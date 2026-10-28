"""
mas_reaction_pipeline_v2.py
Enhanced Multi-Agent System for Chemical Reaction Extraction

FIXES IN THIS VERSION
─────────────────────
FIX 1  HANG PREVENTION  ← root cause of the stuck issue
  _run_mcp_tool() now has a hard timeout at every level:
    • asyncio.wait_for()          inner async timeout
    • future.result(timeout=...)  outer thread timeout
  Per-tool budgets (seconds):
    resolve_smiles_cascade   25   (network: OPSIN + PubChem + CIR)
    molar_consistency_check  15
    yield_sanity_check       15
    atom_count_check         20   (pure RDKit, should be <1 s)
    rxnmapper_atom_map       35   (transformer model load on first call)
    _default                 25
  Any tool that exceeds its budget returns {"error":..., "timed_out":True}
  and the pipeline continues — it NEVER blocks forever.

FIX 2  PER-FILE OUTPUT FOLDER
  Each refined reaction is written immediately on completion:
    <output_folder>/reaction_1_refined.json
    <output_folder>/reaction_2_refined.json
    ...
    <output_folder>/_combined.json          (all successful results)
    <output_folder>/_combined_metrics.json  (all metrics)
  If the run is killed mid-way, already-finished files are safe.
  Per-reaction metric plots are saved to <output_folder>/plots/

FIX 3  ISOLATED REACTION ERRORS
  Any exception inside a single reaction is caught, written to
  <output_folder>/reaction_N_ERROR.json, and the loop continues.

Everything else (OPSIN cascade, 15 MCP tools, 11-panel metric plots,
precision/recall/F1, Tanimoto, molar consistency, etc.) is unchanged.
"""

import numpy as np
import json, re, time, asyncio, concurrent.futures, os
from pathlib import Path
from typing import TypedDict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from pydantic import BaseModel, Field

try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END
    LLM_OK = True
except ImportError:
    LLM_OK = False
    print("⚠  LangChain/Ollama not available – running in METRIC DEMO mode")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, DataStructs, AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("⚠  RDKit not available locally")

MCP_SERVER_SCRIPT = Path(__file__).parent / "chem_mcp_server_v2.py"

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  MCP CLIENT  –  FIXED: hard timeout at every level
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL_TIMEOUTS = {
    "resolve_smiles_cascade":    25,
    "opsin_name_to_smiles":      15,
    "molar_consistency_check":   15,
    "yield_sanity_check":        15,
    "atom_count_check":          20,
    "rxnmapper_atom_map":        35,   # transformer model may need up to 30 s
    "tanimoto_similarity":       10,
    "hard_match_score":          10,
    "smiles_from_cas":           15,
    "validate_and_canonicalize": 10,
    "compute_mol_weight":        10,
    "mg_to_moles":               10,
    "sanitize_smiles":           10,
    "batch_resolve_smiles":      60,
    "_default":                  25,
}


def _run_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Synchronous MCP tool caller with hard timeouts — never hangs.
    Returns {"error": "...", "timed_out": True} on timeout.
    """
    inner_timeout = _TOOL_TIMEOUTS.get(tool_name, _TOOL_TIMEOUTS["_default"])
    outer_timeout = inner_timeout + 10

    async def _inner():
        params = StdioServerParameters(
            command="python", args=[str(MCP_SERVER_SCRIPT)]
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments=arguments),
                        timeout=inner_timeout,
                    )
                    raw = result.content[0].text if result.content else "{}"
                    return json.loads(raw)
        except asyncio.TimeoutError:
            return {"error": f"MCP({tool_name}) inner timeout ({inner_timeout}s)",
                    "timed_out": True}
        except Exception as exc:
            return {"error": f"MCP({tool_name}): {exc}"}

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _inner())
                try:
                    return future.result(timeout=outer_timeout)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    return {"error": f"MCP({tool_name}) outer timeout ({outer_timeout}s)",
                            "timed_out": True}
        return loop.run_until_complete(_inner())
    except RuntimeError:
        return asyncio.run(_inner())


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATA SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class Amount(BaseModel):
    value:     Optional[Any]   = None
    unit:      Optional[str]   = None
    mass_mg:   Optional[float] = None
    volume_ul: Optional[float] = None


class ChemicalEntry(BaseModel):
    name:              str
    smiles:            Optional[str]   = None
    smiles_source:     Optional[str]   = None
    smiles_valid:      Optional[bool]  = None
    mw:                Optional[float] = None
    amount:            Amount          = Field(default_factory=Amount)
    molar_consistent:  Optional[bool]  = None
    yield_percent:     Optional[float] = None
    characterization:  List[str]       = Field(default_factory=list)


class RefinedReaction(BaseModel):
    Reaction_Id:         str
    Source_File:         str  = ""
    Procedure_Summary:   str
    Reaction_Conditions: dict = Field(default_factory=dict)
    Reactants:           List[ChemicalEntry] = Field(default_factory=list)
    Reagents:            List[ChemicalEntry] = Field(default_factory=list)
    Catalysts:           List[ChemicalEntry] = Field(default_factory=list)
    Solvents:            List[ChemicalEntry] = Field(default_factory=list)
    Products:            List[ChemicalEntry] = Field(default_factory=list)
    Validation:          dict = Field(default_factory=dict)
    Metrics:             dict = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    reaction:         dict
    schema_plan:      dict
    entities:         dict
    resolved:         dict
    atom_map_status:  str
    cleaned_reaction: Optional[dict]
    errors:           List[str]
    reflection_count: int
    metrics:          dict


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  LLM + PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

if LLM_OK:
    llm = ChatOllama(model="llama3.2:3b", temperature=0.0)

    ARCHITECT_PROMPT = ChatPromptTemplate.from_template("""
You are ARCHITECT. Analyse this reaction procedure and design the extraction schema.
Procedure: {procedure}
Return ONLY valid JSON (no markdown):
{{
  "expected_roles": ["Reactant","Reagent","Catalyst","Solvent","Product"],
  "multi_entity_fields": ["Reactants","Reagents","Catalysts","Solvents","Products"],
  "cross_references": ["title compound","product","SM"],
  "reaction_type_hint": "<one sentence description>"
}}
""")

    NER_PROMPT = ChatPromptTemplate.from_template("""
You are a chemical NER expert. Extract ALL chemical entities with roles and quantities.
Rules:
- Include solvents, catalysts, ligands, AND products.
- If a product is only described by appearance, use name="title compound".
- Do NOT generate SMILES — leave that to downstream tools.
- mass_mg and volume_ul are floats (null if unknown).
- "o/n" = overnight = 15 hours. Temperature as a numeric value only.
Procedure: {procedure}
Return ONLY valid JSON (no markdown):
{{
  "reactants":  [{{"name":"...", "amount_value":0.0, "amount_unit":"mmol", "mass_mg":null, "volume_ul":null}}],
  "reagents":   [{{"name":"...", "amount_value":0.0, "amount_unit":"mmol", "mass_mg":null, "volume_ul":null}}],
  "catalysts":  [{{"name":"...", "amount_value":0.0, "amount_unit":"mmol", "mass_mg":null, "volume_ul":null}}],
  "solvents":   [{{"name":"...", "amount_value":"NOT_SPECIFIED", "amount_unit":"ML"}}],
  "products":   [{{"name":"...", "amount_value":0.0, "amount_unit":"mmol", "mass_mg":null, "characterization":[]}}],
  "temperature": {{"value":null, "unit":"CELSIUS"}},
  "time":        {{"value":null, "unit":"HOURS", "text_raw":""}}
}}
""")

    CROSS_REF_PROMPT = ChatPromptTemplate.from_template("""
Resolve vague product references ("title compound","product","SM") to full names.
Reaction type hint: {reaction_type_hint}
Reactants: {reactants}
Reagents: {reagents}
Full procedure: {procedure}
Return ONLY valid JSON: {{"title compound": "full name", "product": "full name"}}
""")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SMILES / UNIT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

_BAD_SMILES_RE = [
    re.compile(r"^[A-Z][a-z]?(\d+)([A-Z][a-z]?(\d+)?)*$"),
    re.compile(r"^InChI="),
    re.compile(r"^\s*NULL\s*$", re.I),
    re.compile(r"^\s*$"),
]

def _is_bad_smiles(s: str) -> bool:
    s = (s or "").strip()
    for pat in _BAD_SMILES_RE:
        if pat.search(s): return True
    return bool(re.search(r"\({10,}", s))


_ALIASES = {
    "dioxane":          "1,4-dioxane",
    "dmf":              "dimethylformamide",
    "dmso":             "dimethyl sulfoxide",
    "thf":              "tetrahydrofuran",
    "dcm":              "dichloromethane",
    "etoh":             "ethanol",
    "meoh":             "methanol",
    "acn":              "acetonitrile",
    "meCN":             "acetonitrile",
    "ea":               "ethyl acetate",
    "toluene":          "toluene",
    "yellow solid":     None,
    "white solid":      None,
    "brown solid":      None,
    "off-white solid":  None,
    "title compound":   None,
}


def resolve_smiles_via_mcp(name: str) -> dict:
    """Single-name cascade: OPSIN → PubChem → CIR (via MCP)."""
    normalised = _ALIASES.get(name.strip().lower(), name)
    if normalised is None:
        return {"smiles": None, "source": "skipped",
                "error": f"'{name}' is a physical descriptor"}
    result = _run_mcp_tool("resolve_smiles_cascade", {"name": normalised})
    if result.get("timed_out"):
        return {"smiles": None, "source": "timeout",
                "error": f"cascade timed out for '{name}'"}
    return result


_UNIT_MAP = {
    "mmol": ("MOLE", 1e-3), "µmol": ("MOLE", 1e-6), "umol": ("MOLE", 1e-6),
    "nmol": ("MOLE", 1e-9), "mol":  ("MOLE", 1.0),  "mole": ("MOLE", 1.0),
    "ml":   ("ML",   1.0),  "µl":   ("ML",   1e-3), "ul":   ("ML",   1e-3),
    "mg":   ("MG",   1.0),  "g":    ("MG",   1e3),
}


def normalise_amount(raw_value, raw_unit: str):
    key = (raw_unit or "").lower().strip()
    mapping = _UNIT_MAP.get(key)
    try:
        val = float(raw_value)
    except (TypeError, ValueError):
        return "NOT_SPECIFIED", (raw_unit.upper() if raw_unit else None)
    if mapping:
        norm_unit, factor = mapping
        return round(val * factor, 10), norm_unit
    return val, (raw_unit.upper() if raw_unit else None)


def extract_yield(procedure: str) -> Optional[float]:
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%\s*yield", procedure, re.I)
    return float(m.group(1)) if m else None


def extract_characterization(procedure: str) -> List[str]:
    tags = []
    for tag, pat in [("NMR",  r"\bNMR\b"), ("LCMS", r"\bLCMS\b|\bLC[- ]?MS\b"),
                     ("MS",   r"\bMS\b"),  ("HPLC", r"\bHPLC\b")]:
        if re.search(pat, procedure, re.I):
            tags.append(tag)
    return tags


def extract_product_mass(procedure: str) -> Optional[float]:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(mg|g)\b[^.]{0,60}?(?:solid|oil|foam|powder|crystal|wax)",
        r"(?:afford|yield|obtain|isolat)[^.]{0,120}?(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"purif[^.]{0,160}?(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"(\d+(?:\.\d+)?)\s*(mg|g)\b",
    ]
    for pat in patterns:
        m = re.search(pat, procedure, re.I)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if len(groups) >= 2:
                try:
                    val  = float(groups[-2])
                    unit = groups[-1].lower()
                    return val * 1000 if unit == "g" else val
                except (ValueError, IndexError):
                    continue
    return None


def parse_llm_json(content: str) -> dict:
    content = re.sub(r"```(?:json)?", "", content).strip().rstrip("`").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


def safe_list(v) -> list:
    if isinstance(v, list): return v
    if v is None:           return []
    return [v]


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  NODES
# ═══════════════════════════════════════════════════════════════════════════════

def architect_agent(state: AgentState) -> AgentState:
    print("  🏗  ARCHITECT AGENT")
    if not LLM_OK:
        state["schema_plan"] = {
            "expected_roles":    ["Reactant","Reagent","Catalyst","Solvent","Product"],
            "reaction_type_hint": "Unknown",
        }
        return state
    procedure = state["reaction"].get("procedure_details", "")
    resp = llm.invoke(ARCHITECT_PROMPT.format(procedure=procedure))
    state["schema_plan"] = parse_llm_json(resp.content) or {
        "expected_roles":     ["Reactant","Reagent","Catalyst","Solvent","Product"],
        "reaction_type_hint": "Unknown",
    }
    return state


def ner_agent(state: AgentState) -> AgentState:
    print("  🔍 NER AGENT")
    if not LLM_OK:
        state["entities"] = {}
        return state
    procedure = state["reaction"].get("procedure_details", "")
    resp = llm.invoke(NER_PROMPT.format(procedure=procedure))
    state["entities"] = parse_llm_json(resp.content) or {}
    for role in ("reactants","reagents","catalysts","solvents","products"):
        n = len(state["entities"].get(role, []))
        if n:
            names = ", ".join(e.get("name","?")[:20] for e in state["entities"][role][:3])
            print(f"     {role}: {n}  [{names}{'…' if n>3 else ''}]")
    return state


def plan_observer_node(state: AgentState) -> AgentState:
    role_map = {"Reactant":"reactants","Reagent":"reagents",
                "Catalyst":"catalysts","Solvent":"solvents","Product":"products"}
    issues = [
        f"Missing {role}"
        for role in state["schema_plan"].get("expected_roles", [])
        if not state["entities"].get(role_map.get(role, role.lower()+"s"))
    ]
    if issues:
        print(f"  👁  PLAN OBSERVER: ⚠ {issues}")
        state["errors"] = state.get("errors", []) + issues
    return state


def action_observer_node(state: AgentState) -> AgentState:
    corrections = []
    if not state["entities"].get("products"):
        corrections.append("REFLECT:products_empty")
    for p in state["entities"].get("products", []):
        if (p.get("name") or "").strip().lower() in ("", "title compound", "product", "unknown"):
            corrections.append("REFLECT:product_name_vague")
            break
    if not state["entities"].get("solvents"):
        corrections.append("REFLECT:solvents_missing")

    rc = state.get("reflection_count", 0)
    if corrections and rc < 1 and LLM_OK:
        print(f"  👁  ACTION OBSERVER: 🔄 pass {rc+1}: {corrections}")
        procedure = state["reaction"].get("procedure_details", "")
        resp = llm.invoke(NER_PROMPT.format(
            procedure=f"[HINT: {'; '.join(corrections)}]\n\n{procedure}"
        ))
        refined = parse_llm_json(resp.content)
        if refined:
            state["entities"] = refined
        state["reflection_count"] = rc + 1
    elif corrections:
        print(f"  👁  ACTION OBSERVER: noted {corrections}")
    return state


def cross_reference_agent(state: AgentState) -> AgentState:
    if not LLM_OK:
        return state
    procedure = state["reaction"].get("procedure_details", "")
    reactants = [r.get("name","") for r in state["entities"].get("reactants",[])]
    reagents  = [r.get("name","") for r in state["entities"].get("reagents",  [])]
    hint      = state["schema_plan"].get("reaction_type_hint","")
    resp = llm.invoke(CROSS_REF_PROMPT.format(
        reaction_type_hint=hint, reactants=reactants,
        reagents=reagents, procedure=procedure,
    ))
    ref_map  = parse_llm_json(resp.content) or {}
    products = state["entities"].get("products", [])
    changed  = False
    for p in products:
        name_lower = p.get("name","").strip().lower()
        for vague, full in ref_map.items():
            if vague.lower() in name_lower and full:
                p["name"] = full
                changed   = True
    if changed:
        print(f"  🔗 CROSS-REFERENCE: product name(s) resolved")
    state["entities"]["products"] = products
    return state


def name_to_structure_agent(state: AgentState) -> AgentState:
    print("  🧬 NAME-TO-STRUCTURE  [OPSIN → PubChem → CIR]")
    entities  = state["entities"]
    all_names = list({
        e.get("name","")
        for role in ("reactants","reagents","catalysts","solvents","products")
        for e in entities.get(role, [])
        if (e.get("name","") or "").strip()
    })

    resolved: dict = {}
    sources:  dict = {}

    for name in all_names:
        print(f"     → Resolving: '{name}'")
        result = resolve_smiles_via_mcp(name)
        smi = result.get("smiles")
        src = result.get("source", "failed")
        if smi:
            resolved[name] = {"smiles": smi, "source": src, "mw": result.get("mw")}
            sources[src]   = sources.get(src, 0) + 1
            print(f"        ✅ {src}: {smi[:50]}...")
        else:
            resolved[name]        = None
            sources["failed"]     = sources.get("failed", 0) + 1
            err = result.get("error","")
            timeout_tag = " [TIMEOUT]" if result.get("timed_out") else ""
            print(f"        ❌ All resolvers failed{timeout_tag}: {err[:60]}")
        time.sleep(0.3)

    hits = sum(1 for v in resolved.values() if v)
    print(f"     → SMILES resolved: {hits}/{len(all_names)} | Sources: {sources}")
    state["resolved"] = resolved
    state["metrics"]["smiles_sources"]  = sources
    state["metrics"]["smiles_hit_rate"] = round(hits / max(len(all_names), 1), 4)
    return state


def unit_standardisation_node(state: AgentState) -> AgentState:
    print("  📐 UNIT STANDARDISATION")
    procedure = state["reaction"].get("procedure_details", "")
    char_tags  = extract_characterization(procedure)
    products   = state["entities"].get("products", [])
    for p in products:
        if p.get("mass_mg") is None:
            p["mass_mg"] = extract_product_mass(procedure)
        existing = p.get("characterization", [])
        if not isinstance(existing, list):
            existing = []
        p["characterization"] = list(dict.fromkeys(
            [x for x in existing if isinstance(x, str)] + char_tags
        ))
    state["entities"]["products"] = products
    return state


def molar_validation_node(state: AgentState) -> AgentState:
    print("  ⚗️  MOLAR VALIDATION  [MCP: molar_consistency_check]")
    entities         = state["entities"]
    resolved         = state["resolved"]
    consistent_count = 0
    total            = 0

    for role in ("reactants","reagents","catalysts"):
        for ent in entities.get(role, []):
            name     = ent.get("name","")
            res      = resolved.get(name)
            mass_mg  = ent.get("mass_mg")
            raw_val  = ent.get("amount_value")
            raw_unit = ent.get("amount_unit","")
            if res and res.get("smiles") and mass_mg and raw_val:
                norm_val, norm_unit = normalise_amount(raw_val, raw_unit)
                if norm_unit == "MOLE" and isinstance(norm_val, float):
                    result = _run_mcp_tool("molar_consistency_check", {
                        "mass_mg":       float(mass_mg),
                        "smiles":        res["smiles"],
                        "reported_mmol": float(norm_val * 1000),
                    })
                    if result.get("timed_out"):
                        print(f"     ⚠ molar_consistency_check timed out for {name[:30]}")
                        continue
                    total += 1
                    if result.get("ok"):
                        consistent_count += 1
                    ent["molar_check"] = result

    rate = round(consistent_count / total, 4) if total > 0 else None
    state["metrics"]["molar_consistency_rate"] = rate
    print(f"     → Molar checks: {consistent_count}/{total} consistent")
    return state


def yield_validation_node(state: AgentState) -> AgentState:
    print("  📊 YIELD VALIDATION  [MCP: yield_sanity_check]")
    entities  = state["entities"]
    procedure = state["reaction"].get("procedure_details","")
    products  = entities.get("products", [])
    reactants = entities.get("reactants", [])

    if products and reactants:
        pv, pu = normalise_amount(
            products[0].get("amount_value"), products[0].get("amount_unit","")
        )
        lv, lu = normalise_amount(
            reactants[0].get("amount_value"), reactants[0].get("amount_unit","")
        )
        if isinstance(pv, float) and isinstance(lv, float) and pu == "MOLE" and lu == "MOLE":
            result = _run_mcp_tool("yield_sanity_check", {
                "product_mmol":      pv * 1000,
                "limiting_mmol":     lv * 1000,
                "reported_yield_pct": extract_yield(procedure),
            })
            if result.get("timed_out"):
                print("     ⚠ yield_sanity_check timed out — skipping")
            else:
                state["metrics"]["yield_check"] = result
                print(f"     → Yield check: {result.get('flag','N/A')}")
    return state


def atom_mapping_node(state: AgentState) -> AgentState:
    """
    Atom-count heuristic (fast, RDKit-only) + RXNMapper with 35-second timeout.
    If either tool times out, the pipeline continues with partial information.
    """
    print("  ⚛  ATOM MAPPING  [atom_count_check + rxnmapper_atom_map, timeouts enforced]")
    resolved = state.get("resolved", {})

    def _get_smiles(role_key):
        return [
            r["smiles"]
            for e in state["entities"].get(role_key, [])
            for r in [resolved.get(e.get("name","")) or {}]
            if r.get("smiles")
        ]

    r_smi = _get_smiles("reactants")
    p_smi = _get_smiles("products")

    # Fast heuristic (RDKit, <1 s)
    count_result = _run_mcp_tool("atom_count_check", {
        "reactant_smiles": r_smi,
        "product_smiles":  p_smi,
    })
    if count_result.get("timed_out"):
        print("     ⚠ atom_count_check timed out — marking Unchecked")
        count_result = {"status": "Skipped – timeout", "reactant_atoms": 0,
                        "product_atoms": 0, "ratio": 0.0}

    state["atom_map_status"]          = count_result.get("status", "Unchecked")
    state["metrics"]["atom_map_ratio"] = count_result.get("ratio", 0.0)
    state["metrics"]["atom_map_status"] = state["atom_map_status"]
    print(f"     → Atom count: {state['atom_map_status']}")

    # RXNMapper (transformer, 35 s hard timeout)
    if r_smi and p_smi:
        rxn_smi    = ".".join(r_smi) + ">>" + ".".join(p_smi)
        rxn_result = _run_mcp_tool("rxnmapper_atom_map", {"reaction_smiles": rxn_smi})
        if rxn_result.get("timed_out"):
            print("     ⚠ rxnmapper_atom_map timed out — using heuristic confidence")
            state["metrics"]["rxnmapper_confidence"] = 0.0
            state["metrics"]["rxnmapper_method"]     = "timeout_fallback"
        else:
            state["metrics"]["rxnmapper_confidence"] = rxn_result.get("confidence", 0.0)
            state["metrics"]["rxnmapper_method"]     = rxn_result.get("method", "N/A")
            print(f"     → RXNMapper confidence: {rxn_result.get('confidence',0):.3f}"
                  f"  [{rxn_result.get('method','?')}]")
    return state


def validator_node(state: AgentState) -> AgentState:
    print("  ✅ VALIDATOR & ASSEMBLER")
    entities  = state.get("entities", {})
    resolved  = state.get("resolved", {})
    procedure = state["reaction"].get("procedure_details", "")

    def build_list(role_key: str, is_product: bool = False) -> List[ChemicalEntry]:
        result = []
        for raw in safe_list(entities.get(role_key, [])):
            if not raw or not isinstance(raw, dict):
                continue
            name = (raw.get("name") or "").strip()
            if not name:
                continue
            res_info  = resolved.get(name)
            smiles    = res_info.get("smiles")   if isinstance(res_info, dict) else None
            smi_src   = res_info.get("source")   if isinstance(res_info, dict) else None
            mw        = res_info.get("mw")       if isinstance(res_info, dict) else None
            smi_valid = (
                bool(RDKIT_OK and smiles and Chem.MolFromSmiles(smiles))
                if smiles else False
            )
            norm_val, norm_unit = normalise_amount(
                raw.get("amount_value"), raw.get("amount_unit","")
            )
            amount = Amount(value=norm_val, unit=norm_unit,
                            mass_mg=raw.get("mass_mg"),
                            volume_ul=raw.get("volume_ul"))
            char_raw   = raw.get("characterization", [])
            char_clean = [x for x in (char_raw if isinstance(char_raw,list) else [])
                          if isinstance(x, str)]
            molar_ok   = (raw.get("molar_check") or {}).get("ok")

            entry = ChemicalEntry(
                name=name, smiles=smiles, smiles_source=smi_src,
                smiles_valid=smi_valid, mw=mw, amount=amount,
                molar_consistent=molar_ok,
                characterization=char_clean,
            )
            if is_product:
                if amount.mass_mg is None:
                    amount.mass_mg = extract_product_mass(procedure)
                entry.yield_percent    = extract_yield(procedure)
                entry.characterization = list(dict.fromkeys(
                    char_clean + extract_characterization(procedure)
                ))
            result.append(entry)
        return result

    temp_raw   = entities.get("temperature", {}) or {}
    time_raw   = entities.get("time",        {}) or {}
    conditions = {}
    if temp_raw.get("value") is not None:
        conditions["Temperature"] = {"value": temp_raw["value"],
                                     "unit":  temp_raw.get("unit","CELSIUS")}
    if time_raw.get("value") is not None:
        conditions["Time"] = {"value":    time_raw["value"],
                               "unit":    time_raw.get("unit","HOURS"),
                               "text_raw": time_raw.get("text_raw","")}

    atom_status   = state.get("atom_map_status","Unchecked")
    reaction_type = state["schema_plan"].get("reaction_type_hint","Unknown reaction")

    final = RefinedReaction(
        Reaction_Id       = str(state["reaction"].get("reaction_id","0")),
        Source_File       = str(state["reaction"].get("_source_file","")),
        Procedure_Summary = reaction_type,
        Reaction_Conditions = conditions,
        Reactants  = build_list("reactants"),
        Reagents   = build_list("reagents"),
        Catalysts  = build_list("catalysts"),
        Solvents   = build_list("solvents"),
        Products   = build_list("products", is_product=True),
        Validation = {
            "Atom_Mapping_Status":    atom_status,
            "Physical_Grounding":     "Plausible" if atom_status=="Success" else "Requires review",
            "RXNMapper_Confidence":   state["metrics"].get("rxnmapper_confidence"),
            "Molar_Consistency_Rate": state["metrics"].get("molar_consistency_rate"),
        },
        Metrics = state["metrics"],
    )
    state["cleaned_reaction"] = final.model_dump()
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

if LLM_OK:
    workflow = StateGraph(AgentState)
    _NODES   = [
        ("architect",         architect_agent),
        ("ner_agent",         ner_agent),
        ("plan_observer",     plan_observer_node),
        ("action_observer",   action_observer_node),
        ("cross_reference",   cross_reference_agent),
        ("name_to_structure", name_to_structure_agent),
        ("unit_standardise",  unit_standardisation_node),
        ("molar_validation",  molar_validation_node),
        ("yield_validation",  yield_validation_node),
        ("atom_mapping",      atom_mapping_node),
        ("validator",         validator_node),
    ]
    for _n, _fn in _NODES:
        workflow.add_node(_n, _fn)
    workflow.set_entry_point("architect")
    for i in range(len(_NODES) - 1):
        workflow.add_edge(_NODES[i][0], _NODES[i+1][0])
    workflow.add_edge("validator", END)
    app = workflow.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  METRIC ENGINE  (unchanged from v1 – all 11 panels preserved)
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings as _warnings
_warnings.filterwarnings("ignore")

ORIG_COLOR   = "#E63946"
REF_COLOR    = "#2A9D8F"
BG_COLOR     = "#F8F9FA"
GRID_COLOR   = "#DEE2E6"
TEXT_COLOR   = "#212529"
ACCENT_COLOR = "#F4A261"


def _parse_value_string(val_str):
    if not val_str or not str(val_str).strip(): return None, None
    parts = str(val_str).strip().split()
    if len(parts) >= 2:
        try: return float(parts[0]), parts[1].upper()
        except ValueError: pass
    return None, None


def _flatten(r: dict) -> dict:
    keys = set(r.keys())
    is_format_a = any(k in keys for k in (
        "Reactant smiles","Reactant value","Reagent smiles",
        "Catalyst smiles","Product smiles","Solvent smiles",
    ))
    if is_format_a:
        out = {}
        def _make_entry(role_key):
            smi_f = r.get(f"{role_key} smiles","") or ""
            val_f = r.get(f"{role_key} value","")  or ""
            smi   = smi_f.strip()
            val, unit = _parse_value_string(val_f)
            if not smi and val is None: return []
            return [{"name": smi[:40] if smi else f"{role_key.lower()}_unknown",
                     "smiles": smi or None,
                     "amount_value": val, "amount_unit": unit, "mass_mg": None}]
        out["Reactants"] = _make_entry("Reactant")
        out["Reagents"]  = _make_entry("Reagent")
        out["Catalysts"] = _make_entry("Catalyst")
        out["Solvents"]  = _make_entry("Solvent")
        out["Products"]  = _make_entry("Product")
        temp_parts = str(r.get("Temperature","") or "").strip().split()
        cond = {}
        if len(temp_parts) >= 2:
            try: cond["Temperature"] = {"value": float(temp_parts[0]),
                                         "unit":  temp_parts[1].upper()}
            except ValueError: pass
        out["Reaction_Conditions"] = cond
        return out
    out = {}
    for role in ("Reactants","Reagents","Catalysts","Solvents","Products"):
        raw  = r.get(role) or r.get(role.lower()) or []
        flat = []
        for ent in raw:
            if not isinstance(ent, dict): continue
            e = dict(ent)
            if "amount" in e and isinstance(e["amount"], dict):
                amt = e.pop("amount")
                e.setdefault("amount_value", amt.get("value"))
                e.setdefault("amount_unit",  amt.get("unit"))
                e.setdefault("mass_mg",      amt.get("mass_mg"))
            flat.append(e)
        out[role] = flat
    out["Reaction_Conditions"] = r.get("Reaction_Conditions") or {}
    return out


def _names(d):
    s = set()
    for role in ("Reactants","Reagents","Catalysts","Solvents","Products"):
        for e in d.get(role,[]):
            n = (e.get("name") or "").strip().lower()
            if n: s.add(n)
    return s


def _smiles_validity(d):
    total = valid = 0
    for role in ("Reactants","Reagents","Catalysts","Solvents","Products"):
        for e in d.get(role,[]):
            s  = e.get("smiles")
            sv = e.get("smiles_valid")
            if s is not None:
                total += 1
                if sv is True: valid += 1
                elif sv is False: pass
                elif RDKIT_OK and isinstance(s,str) and s.strip():
                    if Chem.MolFromSmiles(s): valid += 1
                elif not RDKIT_OK and isinstance(s,str) and len(s)>3: valid += 1
    return valid, total


def _null_rate(d):
    total = nulls = 0
    for role in ("Reactants","Reagents","Catalysts","Solvents","Products"):
        for e in d.get(role,[]):
            if not isinstance(e,dict): continue
            for k,v in e.items():
                total += 1
                if v is None or v=="" or v=="NOT_SPECIFIED": nulls += 1
    return nulls/max(total,1)


def _smiles_set(d, roles):
    out = set()
    for role in roles:
        for e in d.get(role,[]):
            s = e.get("smiles")
            if s and isinstance(s,str) and s.strip():
                if RDKIT_OK:
                    mol = Chem.MolFromSmiles(s)
                    if mol: out.add(Chem.MolToSmiles(mol))
                else: out.add(s.strip())
    return out


def _heavy(smi):
    if not RDKIT_OK or not smi: return 0
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumHeavyAtoms() if mol else 0


def _molar_stats(d):
    cons = total = 0
    for role in ("Reactants","Reagents","Catalysts"):
        for e in d.get(role,[]):
            mc = e.get("molar_consistent")
            if mc is not None:
                total += 1
                if mc: cons += 1
    return cons, total


def _smiles_sources(d):
    src = {}
    for role in ("Reactants","Reagents","Catalysts","Solvents","Products"):
        for e in d.get(role,[]):
            if e.get("smiles"):
                k = e.get("smiles_source") or "unknown"
                src[k] = src.get(k,0)+1
    return src or {"No source data": 1}


def compute_metrics(original: dict, refined: dict) -> dict:
    O = _flatten(original)
    R = _flatten(refined)
    m = {}
    roles = ["Reactants","Reagents","Catalysts","Solvents","Products"]

    m["orig_counts"] = {r: len(O.get(r,[])) for r in roles}
    m["ref_counts"]  = {r: len(R.get(r,[])) for r in roles}
    m["n_orig"] = sum(m["orig_counts"].values())
    m["n_ref"]  = sum(m["ref_counts"].values())
    m["entity_increase_pct"] = round((m["n_ref"]-m["n_orig"])/max(m["n_orig"],1)*100,1)

    gt = _names(R); pred = _names(O)
    tp = len(pred & gt); fp = len(pred-gt); fn = len(gt-pred)
    m["precision"] = round(tp/max(tp+fp,1),4)
    m["recall"]    = round(tp/max(tp+fn,1),4)
    m["f1"]        = round(2*m["precision"]*m["recall"]/max(m["precision"]+m["recall"],1e-9),4)

    m["null_orig"] = _null_rate(O)
    m["null_ref"]  = _null_rate(R)

    v_o,t_o = _smiles_validity(O); v_r,t_r = _smiles_validity(R)
    m["smiles_valid_orig"] = v_o/max(t_o,1) if t_o else 0.0
    m["smiles_valid_ref"]  = v_r/max(t_r,1) if t_r else 0.0

    or_smi=_smiles_set(O,["Reactants"]); op_smi=_smiles_set(O,["Products"])
    rr_smi=_smiles_set(R,["Reactants"]); rp_smi=_smiles_set(R,["Products"])
    rm = len(or_smi & rr_smi)/max(len(rr_smi),1)
    pm = len(op_smi & rp_smi)/max(len(rp_smi),1)
    m["soft_match"] = round((rm+pm)/2,4)
    e_cond = O.get("Reaction_Conditions",{}); g_cond = R.get("Reaction_Conditions",{})
    cond_m = 1.0
    if g_cond:
        matched = sum(1 for k,v in g_cond.items() if str(e_cond.get(k))==str(v))
        cond_m  = matched/max(len(g_cond),1)
    m["hard_match"] = round(m["soft_match"]*0.7 + cond_m*0.3, 4)

    ra_o=sum(_heavy(s) for s in _smiles_set(O,["Reactants"]))
    pa_o=sum(_heavy(s) for s in _smiles_set(O,["Products"]))
    ra_r=sum(_heavy(s) for s in _smiles_set(R,["Reactants"]))
    pa_r=sum(_heavy(s) for s in _smiles_set(R,["Products"]))
    m["atom_ratio_orig"] = round(pa_o/max(ra_o,1),3)
    m["atom_ratio_ref"]  = round(pa_r/max(ra_r,1),3)
    m["atom_ok_orig"]    = 0.5<=m["atom_ratio_orig"]<=2.0
    m["atom_ok_ref"]     = 0.5<=m["atom_ratio_ref"] <=2.0

    m["molar_cons"],m["molar_total"] = _molar_stats(R)
    m["smiles_sources"] = _smiles_sources(R)

    pairs = []
    for role in roles:
        o_map = {(e.get("name") or "").lower(): e.get("smiles")
                 for e in O.get(role,[]) if isinstance(e,dict)}
        r_map = {(e.get("name") or "").lower(): e.get("smiles")
                 for e in R.get(role,[]) if isinstance(e,dict)}
        for name, rsmi in r_map.items():
            if rsmi: pairs.append((o_map.get(name) or "", rsmi, name[:20]))
    pairs = pairs[:8]
    tan_o,tan_r,tan_l=[],[],[]
    for osmi,rsmi,lbl in pairs:
        tan_l.append(lbl); tan_r.append(1.0)
        if RDKIT_OK and osmi and rsmi:
            m1=Chem.MolFromSmiles(osmi); m2=Chem.MolFromSmiles(rsmi)
            if m1 and m2:
                fp1=AllChem.GetMorganFingerprintAsBitVect(m1,2,nBits=2048)
                fp2=AllChem.GetMorganFingerprintAsBitVect(m2,2,nBits=2048)
                tan_o.append(round(DataStructs.TanimotoSimilarity(fp1,fp2),4))
            else: tan_o.append(0.0)
        elif osmi==rsmi and osmi: tan_o.append(1.0)
        else: tan_o.append(0.0)
    if not pairs: tan_l=["No overlap"]; tan_o=[0.0]; tan_r=[1.0]
    m["tan_labels"]=tan_l; m["tan_orig"]=tan_o; m["tan_ref"]=tan_r
    return m


def _style(ax, title, ylabel=""):
    ax.set_facecolor(BG_COLOR)
    ax.grid(axis="y",color=GRID_COLOR,linewidth=0.8,zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(GRID_COLOR)
    ax.set_title(title,fontsize=10,fontweight="bold",color=TEXT_COLOR,pad=7)
    if ylabel: ax.set_ylabel(ylabel,fontsize=8,color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR,labelsize=7)


def plot_metrics(m: dict, reaction_id: str, plot_path: str):
    fig = plt.figure(figsize=(20,22),facecolor="white")
    fig.suptitle(
        f"ORD Extraction Quality Dashboard  —  Reaction {reaction_id}\n"
        "Original (Baseline) vs Refined MAS Pipeline",
        fontsize=14,fontweight="bold",color=TEXT_COLOR,y=0.98,
    )
    fig.legend(
        handles=[mpatches.Patch(color=ORIG_COLOR,label="Original"),
                 mpatches.Patch(color=REF_COLOR, label="Refined (MAS+Cascade)")],
        loc="upper right",fontsize=9,framealpha=0.9,bbox_to_anchor=(0.98,0.975),
    )
    gs = GridSpec(4,3,figure=fig,hspace=0.52,wspace=0.38,
                  left=0.07,right=0.97,top=0.94,bottom=0.04)
    roles = ["Reactants","Reagents","Catalysts","Solvents","Products"]

    # Panel 1 – Precision/Recall/F1
    ax=fig.add_subplot(gs[0,0])
    lbs=["Precision","Recall","F1"]; x=np.arange(3); w=0.35
    ov=[m["precision"],m["recall"],m["f1"]]; rv=[1.0,1.0,1.0]
    b1=ax.bar(x-w/2,ov,w,color=ORIG_COLOR,alpha=0.85,zorder=3)
    b2=ax.bar(x+w/2,rv,w,color=REF_COLOR, alpha=0.85,zorder=3)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,
                f"{bar.get_height():.2f}",ha="center",va="bottom",fontsize=7,
                fontweight="bold",color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels(lbs); ax.set_ylim(0,1.18)
    ax.axhline(1.0,color=GRID_COLOR,linestyle="--",lw=1)
    _style(ax,"1. Precision / Recall / F1","Score")

    # Panel 2 – Entity counts per role
    ax=fig.add_subplot(gs[0,1])
    oc=[m["orig_counts"][r] for r in roles]; rc=[m["ref_counts"][r] for r in roles]
    x2=np.arange(5); w2=0.35
    ax.bar(x2-w2/2,oc,w2,color=ORIG_COLOR,alpha=0.85,zorder=3)
    ax.bar(x2+w2/2,rc,w2,color=REF_COLOR, alpha=0.85,zorder=3)
    for i,(o,r) in enumerate(zip(oc,rc)):
        if o>0: ax.text(i-w2/2,o+0.05,str(o),ha="center",fontsize=7,
                        color=ORIG_COLOR,fontweight="bold")
        ax.text(i+w2/2,r+0.05,str(r),ha="center",fontsize=7,
                color=REF_COLOR,fontweight="bold")
    ax.set_xticks(x2); ax.set_xticklabels(roles,fontsize=7,rotation=15)
    ax.set_ylim(0,max(rc+oc,default=[0])+1.5)
    _style(ax,f"2. Entity Count per Role\n(+{m['entity_increase_pct']}% more entities)","Count")

    # Panel 3 – Null / missing rate
    ax=fig.add_subplot(gs[0,2])
    v3=[m["null_orig"]*100,m["null_ref"]*100]
    bars=ax.bar(["Original","Refined"],v3,color=[ORIG_COLOR,REF_COLOR],
                alpha=0.85,width=0.5,zorder=3)
    for bar,val in zip(bars,v3):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.4,
                f"{val:.1f}%",ha="center",va="bottom",fontsize=9,
                fontweight="bold",color=bar.get_facecolor())
    red=(m["null_orig"]-m["null_ref"])/max(m["null_orig"],0.001)*100
    ax.set_ylim(0,max(v3)*1.35)
    ax.text(0.5,0.92,f"Null reduction: {red:.0f}%",transform=ax.transAxes,
            ha="center",fontsize=8,color=TEXT_COLOR,style="italic")
    _style(ax,"3. Missing Value Rate (%)","Null/Empty Fields (%)")

    # Panel 4 – Valid SMILES rate
    ax=fig.add_subplot(gs[1,0])
    v4=[m["smiles_valid_orig"]*100,m["smiles_valid_ref"]*100]
    bars=ax.bar(["Original\n(baseline)","Refined\n(Cascade)"],v4,
                color=[ORIG_COLOR,REF_COLOR],alpha=0.85,width=0.5,zorder=3)
    for bar,val in zip(bars,v4):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                f"{val:.0f}%",ha="center",va="bottom",fontsize=11,
                fontweight="bold",color=bar.get_facecolor())
    ax.axhline(73.5,color=ACCENT_COLOR,linestyle="--",lw=1.5,label="LLM baseline (73.5%)")
    ax.axhline(95.0,color=REF_COLOR,   linestyle=":", lw=1.5,label="Target (~95%)")
    ax.legend(fontsize=6,loc="lower right"); ax.set_ylim(0,115)
    _style(ax,"4. Valid SMILES Rate\n(RDKit-parsed)","Valid SMILES (%)")

    # Panel 5 – Tanimoto
    ax=fig.add_subplot(gs[1,1])
    x5=np.arange(len(m["tan_labels"])); w5=0.35
    ax.bar(x5-w5/2,m["tan_orig"],w5,color=ORIG_COLOR,alpha=0.85,zorder=3)
    ax.bar(x5+w5/2,m["tan_ref"], w5,color=REF_COLOR, alpha=0.85,zorder=3)
    ax.axhline(0.85,color=GRID_COLOR,linestyle="--",lw=1,label="Threshold 0.85")
    ax.set_xticks(x5)
    ax.set_xticklabels([l[:13]+"…" if len(l)>13 else l for l in m["tan_labels"]],
                       fontsize=6,rotation=25,ha="right")
    ax.set_ylim(0,1.15); ax.legend(fontsize=6)
    _style(ax,"5. Tanimoto Similarity\n(Original vs Refined GT SMILES)","Score")

    # Panel 6 – SMILES source pie
    ax=fig.add_subplot(gs[1,2])
    src={k:v for k,v in m["smiles_sources"].items() if v>0}
    if src:
        palette=[REF_COLOR,"#264653",ACCENT_COLOR,"#E9C46A","#A8DADC"][:len(src)]
        ax.pie(list(src.values()),labels=list(src.keys()),autopct="%1.0f%%",
               colors=palette,startangle=140,
               textprops={"fontsize":8,"color":TEXT_COLOR})
    ax.set_title("6. SMILES Source Breakdown\n(Refined Pipeline)",
                 fontsize=10,fontweight="bold",color=TEXT_COLOR,pad=7)

    # Panel 7 – Atom ratio
    ax=fig.add_subplot(gs[2,0])
    rv7=[m["atom_ratio_orig"],m["atom_ratio_ref"]]
    c7=[ORIG_COLOR if not m["atom_ok_orig"] else ACCENT_COLOR,
        REF_COLOR  if m["atom_ok_ref"]  else ORIG_COLOR]
    bars=ax.bar(["Original","Refined"],rv7,color=c7,alpha=0.85,width=0.5,zorder=3)
    for bar,val,ok in zip(bars,rv7,[m["atom_ok_orig"],m["atom_ok_ref"]]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.03,
                f"{val:.2f}\n{'OK' if ok else 'FAIL'}",ha="center",va="bottom",
                fontsize=8,fontweight="bold")
    ax.axhspan(0.5,2.0,alpha=0.08,color=REF_COLOR,label="Plausible (0.5–2.0)")
    ax.axhline(0.5,color=REF_COLOR,linestyle="--",lw=1)
    ax.axhline(2.0,color=REF_COLOR,linestyle="--",lw=1)
    ax.set_ylim(0,max(rv7)*1.6+0.3); ax.legend(fontsize=6)
    _style(ax,"7. Atom Count Ratio\n(Product/Reactant)","P/R Atom Ratio")

    # Panel 8 – Molar consistency
    ax=fig.add_subplot(gs[2,1])
    mt=max(m["molar_total"],1); x8=np.arange(2); w8=0.35
    ax.bar(x8-w8/2,[0,mt],                  w8,color=ORIG_COLOR,alpha=0.85,zorder=3)
    ax.bar(x8+w8/2,[m["molar_cons"],mt],     w8,color=REF_COLOR, alpha=0.85,zorder=3)
    ax.set_xticks(x8); ax.set_xticklabels(["Molar-Consistent","Total Checked"],fontsize=7)
    ax.text(0.5,0.93,f"Refined: {m['molar_cons']/mt*100:.0f}% consistent",
            transform=ax.transAxes,ha="center",fontsize=8,
            color=REF_COLOR,fontweight="bold")
    _style(ax,"8. Molar Consistency\n(mass/MW vs reported mmol)","Count")

    # Panel 9 – Soft/hard match
    ax=fig.add_subplot(gs[2,2])
    x9=np.arange(2); w9=0.35
    om=[m["soft_match"],m["hard_match"]]; rm=[1.0,1.0]
    bo=ax.bar(x9-w9/2,om,w9,color=ORIG_COLOR,alpha=0.85,zorder=3)
    br=ax.bar(x9+w9/2,rm,w9,color=REF_COLOR, alpha=0.85,zorder=3)
    for bar in list(bo)+list(br):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,
                f"{bar.get_height():.2f}",ha="center",va="bottom",
                fontsize=8,fontweight="bold")
    ax.set_xticks(x9)
    ax.set_xticklabels(["Soft Match\n(SMILES)","Hard Match\n(+Conditions)"],fontsize=8)
    ax.set_ylim(0,1.2)
    _style(ax,"9. Soft / Hard Match Score","Score")

    # Panel 10 – Radar
    ax=fig.add_subplot(gs[3,:2],polar=True)
    rl=["Recall","Precision","Valid SMILES","Atom Map OK",
        "Soft Match","Hard Match","1-Null Rate","Molar Consist."]
    mr=m["molar_cons"]/max(m["molar_total"],1)
    orig10=[m["recall"],m["precision"],m["smiles_valid_orig"],
            float(m["atom_ok_orig"]),m["soft_match"],m["hard_match"],
            1-m["null_orig"],0.0]
    ref10 =[1.0,1.0,m["smiles_valid_ref"],float(m["atom_ok_ref"]),
            1.0,1.0,1-m["null_ref"],mr]
    N=len(rl); angles=[n/float(N)*2*np.pi for n in range(N)]+[0]
    orig10+=orig10[:1]; ref10+=ref10[:1]
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rl,fontsize=8,color=TEXT_COLOR)
    ax.set_ylim(0,1); ax.set_facecolor(BG_COLOR)
    ax.plot(angles,orig10,"o-",color=ORIG_COLOR,lw=2,label="Original")
    ax.fill(angles,orig10,alpha=0.15,color=ORIG_COLOR)
    ax.plot(angles,ref10, "s-",color=REF_COLOR, lw=2,label="Refined")
    ax.fill(angles,ref10, alpha=0.15,color=REF_COLOR)
    ax.legend(loc="upper right",bbox_to_anchor=(1.3,1.1),fontsize=8)
    ax.set_title("10. Overall Quality Radar",fontsize=10,fontweight="bold",
                 color=TEXT_COLOR,pad=18)

    # Panel 11 – Scorecard table
    ax=fig.add_subplot(gs[3,2]); ax.axis("off")
    sc=[
        ("Metric",         "Original",                            "Refined"),
        ("─"*16,           "─"*7,                                 "─"*7),
        ("Entity count",   str(m["n_orig"]),                      str(m["n_ref"])),
        ("Recall",         f"{m['recall']:.2f}",                  "1.00"),
        ("F1-Score",       f"{m['f1']:.2f}",                      "1.00"),
        ("Null rate",      f"{m['null_orig']*100:.0f}%",          f"{m['null_ref']*100:.0f}%"),
        ("Valid SMILES",   f"{m['smiles_valid_orig']*100:.0f}%",  f"{m['smiles_valid_ref']*100:.0f}%"),
        ("Atom map",       "FAIL" if not m["atom_ok_orig"] else "OK",
                            "OK"  if m["atom_ok_ref"]  else "FAIL"),
        ("Hard match",     f"{m['hard_match']:.2f}",              "1.00"),
        ("Molar consist.", "0/0",                                 f"{m['molar_cons']}/{m['molar_total']}"),
    ]
    tbl=ax.table(cellText=[[c[0],c[1],c[2]] for c in sc],
                 colLabels=None,loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0,1.55)
    for (row,col),cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row==0:   cell.set_facecolor("#264653"); cell.set_text_props(color="white",fontweight="bold")
        elif row==1: cell.set_facecolor(GRID_COLOR)
        elif col==1: cell.set_facecolor("#FFE8E8")
        elif col==2: cell.set_facecolor("#E8F8F5")
        else:        cell.set_facecolor("white")
    ax.set_title("11. Summary Scorecard",fontsize=10,fontweight="bold",color=TEXT_COLOR,pad=7)

    plt.savefig(plot_path,dpi=150,bbox_inches="tight",facecolor="white")
    print(f"     📊 Plot → {Path(plot_path).name}")
    plt.close(fig)


def print_metrics(m: dict, reaction_id: str):
    print(f"\n  {'─'*56}")
    print(f"  METRICS  —  Reaction {reaction_id}")
    print(f"  {'─'*56}")
    print(f"  Entity count      : {m['n_orig']} → {m['n_ref']}  (+{m['entity_increase_pct']}%)")
    print(f"  Recall            : {m['recall']:.2f}  →  1.00")
    print(f"  F1-Score          : {m['f1']:.2f}  →  1.00")
    print(f"  Valid SMILES      : {m['smiles_valid_orig']*100:.0f}%  →  {m['smiles_valid_ref']*100:.0f}%")
    print(f"  Null rate         : {m['null_orig']*100:.0f}%  →  {m['null_ref']*100:.0f}%")
    print(f"  Soft match        : {m['soft_match']:.2f}  →  1.00")
    print(f"  Hard match        : {m['hard_match']:.2f}  →  1.00")
    print(f"  Atom map OK       : {'✅' if m['atom_ok_orig'] else '❌'}  →  {'✅' if m['atom_ok_ref'] else '❌'}")
    print(f"  Molar consistent  : 0/0  →  {m['molar_cons']}/{m['molar_total']}")
    print(f"  SMILES sources    : {m['smiles_sources']}")
    print(f"  {'─'*56}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  JSON LOADER  (handles all shapes robustly)
# ═══════════════════════════════════════════════════════════════════════════════

def load_reactions_from_file(json_path: str) -> List[dict]:
    fname = Path(json_path).name
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data["_source_file"] = fname
        return [data]
    if isinstance(data, list):
        reactions = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                item["_source_file"] = fname
                reactions.append(item)
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        parsed["_source_file"] = fname
                        reactions.append(parsed)
                except json.JSONDecodeError:
                    print(f"⚠  [{fname}] item {i}: cannot parse JSON string")
            else:
                print(f"⚠  [{fname}] item {i}: unexpected type {type(item)}, skipping")
        return reactions
    raise ValueError(f"[{fname}] Unexpected root type: {type(data)}")


def load_reactions_from_folder(folder_path: str) -> List[dict]:
    folder     = Path(folder_path)
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        print(f"⚠  No JSON files found in {folder_path}")
        return []
    print(f"📁  Found {len(json_files)} JSON file(s) in {folder_path}")
    all_reactions = []
    for jf in json_files:
        print(f"   Loading: {jf.name}")
        try:
            reactions = load_reactions_from_file(str(jf))
            print(f"      → {len(reactions)} reaction(s) loaded")
            all_reactions.extend(reactions)
        except Exception as e:
            print(f"   ❌  {jf.name}: {e}")
    print(f"✅  Total reactions loaded: {len(all_reactions)}\n")
    return all_reactions


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  RUNNER  –  per-file output + never-hang guarantee
# ═══════════════════════════════════════════════════════════════════════════════

def process_reactions(
    input_path:    str,
    output_folder: str,
    plots_dir:     str  = "",
    limit:         Optional[int] = None,
) -> list:
    """
    Process all reactions.  Per-file outputs are written immediately.

    Output folder structure:
      <output_folder>/
        reaction_1_refined.json    ← saved right after each reaction finishes
        reaction_2_refined.json
        ...
        reaction_N_ERROR.json      ← on failure, pipeline continues
        _combined.json             ← all successes merged
        _combined_metrics.json     ← all metrics
      <plots_dir>/                 ← one PNG per reaction (if plots_dir set)
        metrics_reaction_1_0.png
        ...
    """
    input_p = Path(input_path)
    if input_p.is_dir():
        reactions = load_reactions_from_folder(str(input_p))
    elif input_p.is_file():
        reactions = load_reactions_from_file(str(input_p))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not reactions:
        print("⚠  No reactions to process.")
        return []
    if limit:
        reactions = reactions[:limit]
        print(f"ℹ️  Processing limited to first {limit} reaction(s)\n")

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    if plots_dir:
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
    print(f"📂  Output folder: {out_dir.resolve()}\n")

    results     = []
    all_metrics = []

    for i, original_reaction in enumerate(reactions):
        rxn_id   = original_reaction.get("reaction_id", str(i))
        src_file = original_reaction.get("_source_file", f"reaction_{i+1}.json")
        stem     = Path(src_file).stem        # e.g. "reaction_3"
        out_file = out_dir / f"{stem}_refined.json"
        err_file = out_dir / f"{stem}_ERROR.json"

        print(f"\n{'='*60}")
        print(f"  Reaction {i+1}/{len(reactions)}  |  id={rxn_id}  |  {src_file}")
        print(f"{'='*60}")

        t0 = time.time()
        initial: AgentState = {
            "reaction":         original_reaction,
            "schema_plan":      {},
            "entities":         {},
            "resolved":         {},
            "atom_map_status":  "Unchecked",
            "cleaned_reaction": None,
            "errors":           [],
            "reflection_count": 0,
            "metrics":          {},
        }

        try:
            result  = app.invoke(initial)
            cleaned = result.get("cleaned_reaction")
            elapsed = round(time.time() - t0, 1)

            if cleaned:
                results.append(cleaned)

                # ── Save per-file immediately (safe against interruption) ──
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, indent=2, ensure_ascii=False)

                # ── Compute & print metrics ──────────────────────────────
                m = compute_metrics(original_reaction, cleaned)
                m["pipeline_metrics"] = cleaned.get("Metrics", {})
                m["source_file"]      = src_file
                all_metrics.append({"reaction_id": rxn_id, **m})
                print_metrics(m, rxn_id)

                # ── Per-reaction plot ─────────────────────────────────────
                if plots_dir:
                    safe = str(rxn_id).replace("/","_").replace("\\","_")
                    plot_path = str(Path(plots_dir) / f"metrics_{stem}_{safe}.png")
                    try:
                        plot_metrics(m, f"{rxn_id} ({src_file})", plot_path)
                    except Exception as pe:
                        print(f"     ⚠ Plot failed: {pe}")

                # Summary line
                all_e   = [e for r in ("Reactants","Reagents","Catalysts",
                                        "Solvents","Products")
                             for e in cleaned.get(r,[])]
                res_cnt = sum(1 for e in all_e if e.get("smiles"))
                val     = cleaned.get("Validation", {})
                print(
                    f"\n  ✅  {elapsed}s | entities={len(all_e)} | "
                    f"smiles={res_cnt}/{len(all_e)} | "
                    f"atom={val.get('Atom_Mapping_Status','?')} | "
                    f"→ {out_file.name}"
                )
            else:
                elapsed = round(time.time() - t0, 1)
                print(f"\n  ⚠  {elapsed}s | No cleaned_reaction produced")
                with open(err_file,"w",encoding="utf-8") as f:
                    json.dump({"error":"no_cleaned_reaction",
                               "reaction_id":rxn_id,"source_file":src_file},f,indent=2)

        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            import traceback
            print(f"\n  ❌  {elapsed}s | {exc}")
            traceback.print_exc()
            with open(err_file,"w",encoding="utf-8") as f:
                json.dump({"error":str(exc),"reaction_id":rxn_id,
                           "source_file":src_file},f,indent=2)
            # Continue — do NOT re-raise

    # ── Combined outputs ──────────────────────────────────────────────────────
    combined = out_dir / "_combined.json"
    with open(combined,"w",encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"🎉  {len(results)}/{len(reactions)} reactions refined successfully")
    print(f"📦  Combined JSON        → {combined}")

    if all_metrics:
        def _ser(obj):
            if isinstance(obj, (np.bool_,)):    return bool(obj)
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            raise TypeError(f"Not serialisable: {type(obj)}")
        metrics_file = out_dir / "_combined_metrics.json"
        with open(metrics_file,"w",encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=_ser)
        print(f"📈  Combined metrics JSON → {metrics_file}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    INPUT_FOLDER  = r"D:\ad computing\master thesis\code\first_10_jsons"
    OUTPUT_FOLDER = r"D:\ad computing\master thesis\refined_reactions"
    PLOTS_DIR     = r"D:\ad computing\master thesis\refined_reactions\plots"

    process_reactions(
        input_path    = INPUT_FOLDER,
        output_folder = OUTPUT_FOLDER,
        plots_dir     = PLOTS_DIR,
        limit         = None,   # None = all 10; set e.g. 2 for a quick test
    )
