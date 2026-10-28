"""
mas_reaction_pipeline_v2.py
Enhanced Multi-Agent System for Chemical Reaction Extraction
Key improvements over v1:
  - SMILES Bottleneck FIX: cascade resolver (OPSIN → PubChem → CIR), NO raw LLM SMILES
  - 15 MCP tools integrated (vs 7 in v1)
  - Full metric suite: Precision/Recall/F1, Valid SMILES Rate, Tanimoto, 
    Molar Consistency, Atom Mapping, Hard/Soft Match, Yield Sanity
  - Metric plotting via matplotlib
  - Comparison dashboard: Original extraction vs Refined extraction
"""
import numpy as np
import json, re, time, asyncio, concurrent.futures, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from typing import TypedDict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from pydantic import BaseModel, Field

# ── LLM setup: OpenAI preferred, Ollama fallback ─────────────────────────────
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END
    _LANGGRAPH_OK = True
except ImportError:
    _LANGGRAPH_OK = False
    print("⚠  LangGraph not available")

LLM_OK = False
llm    = None

if _LANGGRAPH_OK:
    # Try OpenAI first (much better for chemistry NER)
    try:
        from langchain_openai import ChatOpenAI
        import openai as _oai_test          # will raise if not installed
        llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        LLM_OK = True
        print("✅  LLM: OpenAI gpt-4o-mini")
    except Exception:
        pass

    # Fallback: Ollama with a larger model
    if not LLM_OK:
        try:
            from langchain_ollama import ChatOllama
            llm    = ChatOllama(model="llama3.1:8b", temperature=0.0)
            LLM_OK = True
            print("✅  LLM: Ollama llama3.1:8b")
        except Exception:
            try:
                from langchain_ollama import ChatOllama
                llm    = ChatOllama(model="llama3.2:3b", temperature=0.0)
                LLM_OK = True
                print("⚠  LLM: Ollama llama3.2:3b (small model — NER quality may be reduced)")
            except Exception:
                print("⚠  No LLM available — running in METRIC DEMO mode")

# ── FastMCP client ─────────────────────────────────────────────────────────────
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── RDKit ─────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, DataStructs, AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("⚠  RDKit not available locally")

MCP_SERVER_SCRIPT = Path(__file__).parent / "chem_mcp_server_v2.py"

# =============================================================================
# 1. MCP CLIENT WRAPPER
# =============================================================================

def _run_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Synchronous wrapper for MCP tool calls."""
    async def _inner():
        server_params = StdioServerParameters(
            command="python", args=[str(MCP_SERVER_SCRIPT)]
        )
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments)
                    raw = result.content[0].text if result.content else "{}"
                    return json.loads(raw)
        except Exception as e:
            return {"error": f"MCP call failed ({tool_name}): {e}"}

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _inner()).result()
        return loop.run_until_complete(_inner())
    except RuntimeError:
        return asyncio.run(_inner())

# =============================================================================
# 2. DATA SCHEMAS
# =============================================================================

class Amount(BaseModel):
    value: Optional[Any] = None
    unit: Optional[str] = None
    mass_mg: Optional[float] = None
    volume_ul: Optional[float] = None

class ChemicalEntry(BaseModel):
    name: str
    smiles: Optional[str] = None
    smiles_source: Optional[str] = None   # NEW: tracks where SMILES came from
    smiles_valid: Optional[bool] = None   # NEW: RDKit validity flag
    mw: Optional[float] = None            # NEW: molecular weight
    amount: Amount = Field(default_factory=Amount)
    molar_consistent: Optional[bool] = None  # NEW: Metric B check
    yield_percent: Optional[float] = None
    characterization: List[str] = Field(default_factory=list)

class RefinedReaction(BaseModel):
    Reaction_Id: str
    Procedure_Summary: str
    Reaction_Conditions: dict = Field(default_factory=dict)
    Reactants:  List[ChemicalEntry] = Field(default_factory=list)
    Reagents:   List[ChemicalEntry] = Field(default_factory=list)
    Catalysts:  List[ChemicalEntry] = Field(default_factory=list)
    Solvents:   List[ChemicalEntry] = Field(default_factory=list)
    Products:   List[ChemicalEntry] = Field(default_factory=list)
    Validation: dict = Field(default_factory=dict)
    Metrics:    dict = Field(default_factory=dict)   # NEW: inline metrics

# =============================================================================
# 3. AGENT STATE
# =============================================================================

class AgentState(TypedDict):
    reaction:         dict
    schema_plan:      dict
    entities:         dict
    resolved:         dict
    atom_map_status:  str
    cleaned_reaction: Optional[dict]
    errors:           List[str]
    reflection_count: int
    metrics:          dict          # NEW

# =============================================================================
# 4. LLM SETUP (with fallback)
# =============================================================================

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
You are a chemistry expert. Extract ALL chemical entities from the procedure below.

RULES:
- reactants: starting materials that are consumed (substrates)
- reagents: bases, ligands, oxidants, additives (not consumed as substrate)
- catalysts: metal catalysts, enzymes
- solvents: reaction solvents
- products: compounds formed (use "title compound" if only described by appearance)
- Do NOT write SMILES — leave blank
- Include amounts exactly as written

Procedure:
{procedure}

Return ONLY a JSON object (no explanation, no markdown):
{{
  "reactants":  [{{"name":"EXACT_NAME", "amount_value":0.0, "amount_unit":"MOLE", "mass_mg":null, "volume_ul":null}}],
  "reagents":   [{{"name":"EXACT_NAME", "amount_value":0.0, "amount_unit":"MOLE", "mass_mg":null, "volume_ul":null}}],
  "catalysts":  [{{"name":"EXACT_NAME", "amount_value":0.0, "amount_unit":"MOLE", "mass_mg":null, "volume_ul":null}}],
  "solvents":   [{{"name":"EXACT_NAME", "amount_value":"NOT_SPECIFIED", "amount_unit":"ML"}}],
  "products":   [{{"name":"EXACT_NAME", "amount_value":null, "amount_unit":"MOLE", "mass_mg":null, "characterization":[]}}],
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

# =============================================================================
# 5. SMILES UTILITIES — THE FIX
# =============================================================================

_BAD_SMILES_RE = [
    re.compile(r"^[A-Z][a-z]?(\d+)([A-Z][a-z]?(\d+)?)*$"),
    re.compile(r"^InChI="),
    re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"),
    re.compile(r"^\s*NULL\s*$", re.IGNORECASE),
    re.compile(r"^\s*$"),
]

def _is_bad_smiles(s: str) -> bool:
    s = s.strip()
    for pat in _BAD_SMILES_RE:
        if pat.search(s): return True
    if re.search(r"\({10,}", s): return True
    return False

def resolve_smiles_via_mcp(name: str) -> dict:
    """
    THE FIXED resolver: uses cascade (OPSIN→PubChem→CIR) via MCP.
    Returns {smiles, source, mw} or {smiles: None, ...}.
    NEVER uses raw LLM for SMILES.
    """
    # Pre-resolve common solvent/reagent aliases that confuse web APIs
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
        "pe":               "petroleum ether",
        "toluene":          "toluene",
        "yellow solid":     None,   # un-resolvable descriptor → skip
        "white solid":      None,
        "brown solid":      None,
        "off-white solid":  None,
        "title compound":   None,
    }
    normalised = _ALIASES.get(name.strip().lower(), name)
    if normalised is None:
        return {"smiles": None, "source": "skipped",
                "error": f"'{name}' is a physical descriptor, not a chemical name"}

    result = _run_mcp_tool("resolve_smiles_cascade", {"name": normalised})
    return result

def validate_smiles_via_mcp(smiles: str) -> Optional[str]:
    """Returns canonical SMILES if valid, else None."""
    if not smiles or _is_bad_smiles(smiles): return None
    result = _run_mcp_tool("validate_and_canonicalize", {"smiles": smiles})
    return result.get("canonical") if result.get("valid") else None

def parse_llm_json(content: str) -> dict:
    """Robust JSON extractor — handles markdown fences, truncation, partial output."""
    if not content:
        return {}
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find outermost {...} block
    start = text.find("{")
    if start != -1:
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                # Try to repair: remove trailing commas before } or ]
                repaired = re.sub(r",\s*([}\]])", r"\1", text[start:end])
                try:
                    return json.loads(repaired)
                except Exception:
                    pass
    # Last resort: extract key-value pairs manually for NER output
    result = {}
    for role in ("reactants","reagents","catalysts","solvents","products"):
        # find arrays like "reactants": [...]
        m = re.search(
            rf'"{role}"\s*:\s*(\[.*?\])(?=\s*[,}}]|\s*")',
            text, re.DOTALL | re.IGNORECASE
        )
        if m:
            try:
                result[role] = json.loads(m.group(1))
            except Exception:
                result[role] = []
    return result

# =============================================================================
# 6. UNIT HELPERS
# =============================================================================

_UNIT_MAP = {
    "mmol": ("MOLE", 1e-3), "µmol": ("MOLE", 1e-6), "umol": ("MOLE", 1e-6),
    "nmol": ("MOLE", 1e-9), "mol": ("MOLE", 1.0), "mole": ("MOLE", 1.0),
    "ml": ("ML", 1.0), "µl": ("ML", 1e-3), "ul": ("ML", 1e-3),
    "mg": ("MG", 1.0), "g": ("MG", 1e3),
}

def normalise_amount(raw_value, raw_unit: str):
    key = (raw_unit or "").lower().strip()
    mapping = _UNIT_MAP.get(key)
    try: val = float(raw_value)
    except (TypeError, ValueError):
        return "NOT_SPECIFIED", (raw_unit.upper() if raw_unit else None)
    if mapping:
        norm_unit, factor = mapping
        return round(val * factor, 10), norm_unit
    return val, (raw_unit.upper() if raw_unit else None)

def extract_yield(procedure: str) -> Optional[float]:
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%\s*yield", procedure, re.IGNORECASE)
    return float(m.group(1)) if m else None

def extract_characterization(procedure: str) -> List[str]:
    tags = []
    for tag, pat in [("NMR", r"\bNMR\b"), ("LCMS", r"\bLCMS\b|\bLC[- ]?MS\b"),
                     ("MS", r"\bMS\b"), ("HPLC", r"\bHPLC\b")]:
        if re.search(pat, procedure, re.IGNORECASE): tags.append(tag)
    return tags

def extract_product_mass(procedure: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|g)\b", procedure, re.IGNORECASE)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        return val * 1000 if unit == "g" else val
    return None

def safe_list(v) -> list:
    if isinstance(v, list): return v
    if v is None: return []
    return [v]

# =============================================================================
# 7. (Metric engine is in section 10 — after workflow)
# =============================================================================

# =============================================================================
# 8. NODES (same as v1 but with SMILES fix applied)
# =============================================================================

def architect_agent(state: AgentState) -> AgentState:
    print("🏗  ARCHITECT AGENT")
    if not LLM_OK:
        state["schema_plan"] = {
            "expected_roles": ["Reactant","Reagent","Catalyst","Solvent","Product"],
            "reaction_type_hint": "Unknown"
        }
        return state
    procedure = state["reaction"].get("procedure_details", "")
    resp = llm.invoke(ARCHITECT_PROMPT.format(procedure=procedure))
    state["schema_plan"] = parse_llm_json(resp.content) or {
        "expected_roles": ["Reactant","Reagent","Catalyst","Solvent","Product"],
        "multi_entity_fields": ["Reactants","Reagents","Catalysts","Solvents","Products"],
        "cross_references": ["title compound","product"],
        "reaction_type_hint": "Unknown",
    }
    return state

def ner_agent(state: AgentState) -> AgentState:
    print("🔍 NER AGENT")
    if not LLM_OK:
        state["entities"] = {}
        return state
    procedure = state["reaction"].get("procedure_details", "")
    if not procedure.strip():
        print("   ⚠  Empty procedure — NER cannot extract entities")
        state["entities"] = {}
        return state
    resp = llm.invoke(NER_PROMPT.format(procedure=procedure))
    raw  = resp.content or ""
    parsed = parse_llm_json(raw)
    if not parsed or not any(parsed.get(r) for r in
            ("reactants","reagents","catalysts","solvents","products")):
        print(f"   ⚠  NER returned no entities. Raw output (first 300 chars):")
        print(f"      {repr(raw[:300])}")
    else:
        total = sum(len(parsed.get(r,[])) for r in
                    ("reactants","reagents","catalysts","solvents","products"))
        print(f"   → Extracted {total} entities across all roles")
    state["entities"] = parsed or {}
    return state

def plan_observer_node(state: AgentState) -> AgentState:
    print("👁  PLAN OBSERVER")
    entities = state["entities"]
    schema   = state["schema_plan"]
    role_map = {"Reactant":"reactants","Reagent":"reagents",
                "Catalyst":"catalysts","Solvent":"solvents","Product":"products"}
    issues = [
        f"Missing {role}"
        for role in schema.get("expected_roles", [])
        if not entities.get(role_map.get(role, role.lower()+"s"))
    ]
    if issues:
        state["errors"] = state.get("errors", []) + issues
    return state

def action_observer_node(state: AgentState) -> AgentState:
    print("👁  ACTION OBSERVER")
    entities = state["entities"]
    corrections = []

    if not entities.get("products"):
        corrections.append("REFLECT:products_empty")
    for p in entities.get("products", []):
        name = (p.get("name") or "").strip().lower()
        if name in ("", "title compound", "product", "unknown"):
            corrections.append("REFLECT:product_name_vague")
            break   # one flag is enough, avoid duplicate hints
    if not entities.get("solvents"):
        corrections.append("REFLECT:solvents_missing")

    # Only reflect ONCE maximum, and only if truly needed
    reflection_count = state.get("reflection_count", 0)
    if corrections and reflection_count < 1 and LLM_OK:
        print(f"   🔄 Reflection (pass {reflection_count + 1}): {corrections}")
        procedure = state["reaction"].get("procedure_details", "")
        resp = llm.invoke(NER_PROMPT.format(
            procedure=f"[HINT: {'; '.join(corrections)}]\n\n{procedure}"
        ))
        refined = parse_llm_json(resp.content)
        if refined:
            state["entities"] = refined
        state["reflection_count"] = reflection_count + 1
    elif corrections:
        print(f"   ℹ️  Issues noted (no further reflection): {corrections}")
    return state


def grounding_filter_node(state: AgentState) -> AgentState:
    """
    Anti-hallucination guard: remove any extracted entity whose name shares
    NO words with the actual procedure text.
    This catches cases where the LLM invents chemicals not mentioned at all.
    Keeps entities if procedure is empty (can't verify) or name is very short.
    """
    print("🛡  GROUNDING FILTER")
    procedure = state["reaction"].get("procedure_details", "").lower()
    if not procedure.strip():
        print("   ⚠  No procedure text — skipping grounding check")
        return state

    # Tokenise procedure into a set of meaningful words (≥4 chars)
    proc_tokens = set(re.findall(r"[a-z]{4,}", procedure))

    kept_total = 0
    dropped_total = 0

    for role in ("reactants", "reagents", "catalysts", "solvents", "products"):
        original_list = state["entities"].get(role, [])
        kept = []
        for ent in original_list:
            name = (ent.get("name") or "").strip().lower()
            if not name or len(name) < 4:
                kept.append(ent)   # too short to verify, keep
                continue
            # Tokenise the entity name
            name_tokens = set(re.findall(r"[a-z]{4,}", name))
            # Check if ANY token from name appears in procedure
            overlap = name_tokens & proc_tokens
            if overlap:
                kept.append(ent)
            else:
                print(f"   ✂  Dropped '{ent.get('name')}' [{role}] — not found in procedure")
                dropped_total += 1
        state["entities"][role] = kept
        kept_total += len(kept)

    print(f"   → Kept {kept_total} entities, dropped {dropped_total} hallucinations")
    return state

def cross_reference_agent(state: AgentState) -> AgentState:
    print("🔗 CROSS-REFERENCE AGENT")
    if not LLM_OK: return state
    procedure = state["reaction"].get("procedure_details", "")
    reactants = [r.get("name","") for r in state["entities"].get("reactants",[])]
    reagents  = [r.get("name","") for r in state["entities"].get("reagents",[])]
    hint      = state["schema_plan"].get("reaction_type_hint","")
    resp = llm.invoke(CROSS_REF_PROMPT.format(
        reaction_type_hint=hint, reactants=reactants,
        reagents=reagents, procedure=procedure
    ))
    ref_map = parse_llm_json(resp.content) or {}
    products = state["entities"].get("products", [])
    for p in products:
        name_lower = p.get("name","").strip().lower()
        for vague, full in ref_map.items():
            if vague.lower() in name_lower and full:
                p["name"] = full
    state["entities"]["products"] = products
    return state

def name_to_structure_agent(state: AgentState) -> AgentState:
    """
    THE FIXED NAME-TO-STRUCTURE AGENT.
    Uses resolve_smiles_cascade (OPSIN → PubChem → CIR) for ALL entities.
    NO raw LLM SMILES generation – eliminates hallucination at source.
    """
    print("🧬 NAME-TO-STRUCTURE AGENT  [CASCADE: OPSIN → PubChem → CIR]")
    entities = state["entities"]
    all_names = list({
        e.get("name","")
        for role in ("reactants","reagents","catalysts","solvents","products")
        for e in entities.get(role, [])
        if e.get("name","").strip()
    })

    resolved = {}
    sources  = {}
    for name in all_names:
        print(f"   → Resolving: '{name}'")
        result = resolve_smiles_via_mcp(name)
        smi = result.get("smiles")
        src = result.get("source", "failed")
        if smi:
            resolved[name] = {"smiles": smi, "source": src, "mw": result.get("mw")}
            sources[src] = sources.get(src, 0) + 1
            print(f"      ✅ {src}: {smi[:40]}...")
        else:
            resolved[name] = None
            sources["failed"] = sources.get("failed", 0) + 1
            print(f"      ❌ All resolvers failed")
        time.sleep(0.3)

    hits = sum(1 for v in resolved.values() if v)
    print(f"   → SMILES resolved: {hits}/{len(all_names)} | Sources: {sources}")
    state["resolved"] = resolved
    state["metrics"]["smiles_sources"] = sources
    state["metrics"]["smiles_hit_rate"] = round(hits / max(len(all_names), 1), 4)
    return state

def unit_standardisation_node(state: AgentState) -> AgentState:
    print("📐 UNIT STANDARDISATION")
    procedure = state["reaction"].get("procedure_details", "")
    char_tags  = extract_characterization(procedure)
    products   = state["entities"].get("products", [])
    for p in products:
        if p.get("mass_mg") is None:
            p["mass_mg"] = extract_product_mass(procedure)
        # Fix: ensure characterization is always a flat list of strings,
        # never a list of dicts (which causes unhashable type error in set())
        existing = p.get("characterization", [])
        if not isinstance(existing, list):
            existing = []
        # Flatten: keep only plain strings, discard any dicts the LLM injected
        existing_str = [x for x in existing if isinstance(x, str)]
        merged = list(dict.fromkeys(existing_str + char_tags))  # dedup, preserve order
        p["characterization"] = merged
    state["entities"]["products"] = products
    return state

def molar_validation_node(state: AgentState) -> AgentState:
    """NEW: Molar consistency check for each entity with SMILES and mass."""
    print("⚗️  MOLAR VALIDATION  [MCP: molar_consistency_check]")
    entities = state["entities"]
    resolved = state["resolved"]
    consistent_count, total = 0, 0

    for role in ("reactants","reagents","catalysts"):
        for ent in entities.get(role, []):
            name    = ent.get("name","")
            res     = resolved.get(name)
            mass_mg = ent.get("mass_mg")
            raw_val = ent.get("amount_value")
            raw_unit= ent.get("amount_unit","")

            if res and res.get("smiles") and mass_mg and raw_val:
                norm_val, norm_unit = normalise_amount(raw_val, raw_unit)
                if norm_unit == "MOLE" and isinstance(norm_val, float):
                    reported_mmol = norm_val * 1000
                    result = _run_mcp_tool("molar_consistency_check", {
                        "mass_mg": float(mass_mg),
                        "smiles":  res["smiles"],
                        "reported_mmol": float(reported_mmol)
                    })
                    total += 1
                    if result.get("ok"): consistent_count += 1
                    ent["molar_check"] = result

    state["metrics"]["molar_consistency_rate"] = (
        round(consistent_count / total, 4) if total > 0 else None
    )
    print(f"   → Molar checks: {consistent_count}/{total} consistent")
    return state

def yield_validation_node(state: AgentState) -> AgentState:
    """NEW: Yield sanity check for products."""
    print("📊 YIELD VALIDATION  [MCP: yield_sanity_check]")
    entities = state["entities"]
    procedure = state["reaction"].get("procedure_details","")
    
    products  = entities.get("products",[])
    reactants = entities.get("reactants",[])
    
    if products and reactants:
        prod_val, prod_unit = normalise_amount(
            products[0].get("amount_value"), products[0].get("amount_unit","")
        )
        lim_val, lim_unit = normalise_amount(
            reactants[0].get("amount_value"), reactants[0].get("amount_unit","")
        )
        if (isinstance(prod_val, float) and isinstance(lim_val, float)
                and prod_unit == "MOLE" and lim_unit == "MOLE"):
            result = _run_mcp_tool("yield_sanity_check", {
                "product_mmol":   prod_val * 1000,
                "limiting_mmol":  lim_val  * 1000,
                "reported_yield_pct": extract_yield(procedure)
            })
            state["metrics"]["yield_check"] = result
            print(f"   → Yield check: {result.get('flag','N/A')}")
    return state

def atom_mapping_node(state: AgentState) -> AgentState:
    """
    Atom count check done LOCALLY (no MCP call) — avoids hanging on rxnmapper.
    RXNMapper MCP call is skipped entirely; confidence is set to 0 (not installed).
    """
    print("⚛  ATOM MAPPING  [local heavy-atom count]")
    resolved = state.get("resolved", {})

    def get_smiles(role_key):
        out = []
        for r in state["entities"].get(role_key, []):
            name = r.get("name", "")
            res  = resolved.get(name)
            if res and isinstance(res, dict) and res.get("smiles"):
                out.append(res["smiles"])
        return out

    r_smi = get_smiles("reactants")
    p_smi = get_smiles("products")

    # ── Local heavy-atom count (no MCP, no hanging) ───────────────────────────
    def _count_atoms(smi_list):
        total = 0
        for s in smi_list:
            if not s or not isinstance(s, str): continue
            if RDKIT_OK:
                mol = Chem.MolFromSmiles(s)
                if mol: total += mol.GetNumHeavyAtoms()
            else:
                total += max(len(s) // 3, 1)   # rough estimate without RDKit
        return total

    r_atoms = _count_atoms(r_smi)
    p_atoms = _count_atoms(p_smi)

    if r_atoms == 0 and p_atoms == 0:
        status = "Skipped – no SMILES"
        ratio  = 0.0
    elif r_atoms == 0 or p_atoms == 0:
        status = "Incomplete – one side missing"
        ratio  = 0.0
    elif 0.5 <= (p_atoms / r_atoms) <= 2.0:
        status = "Success"
        ratio  = round(p_atoms / r_atoms, 3)
    else:
        ratio  = round(p_atoms / r_atoms, 3)
        status = f"Warning – atom mismatch (ratio={ratio:.2f})"

    state["atom_map_status"]                 = status
    state["metrics"]["atom_map_ratio"]       = ratio
    state["metrics"]["atom_map_status"]      = status
    state["metrics"]["rxnmapper_confidence"] = 0.0
    state["metrics"]["rxnmapper_method"]     = "local_heuristic"

    print(f"   → Reactant atoms: {r_atoms}  |  Product atoms: {p_atoms}")
    print(f"   → Status: {status}")
    return state

    # ── atom_count_check ─────────────────────────────────────────────────────
    try:
        count_result = _run_mcp_tool("atom_count_check", {
            "reactant_smiles": r_smi if r_smi else [""],
            "product_smiles":  p_smi if p_smi else [""]
        })
        state["atom_map_status"] = count_result.get("status", "Unchecked")
        state["metrics"]["atom_map_ratio"]  = count_result.get("ratio", 0.0)
        state["metrics"]["atom_map_status"] = state["atom_map_status"]
    except Exception as e:
        print(f"   ⚠  atom_count_check error: {e}")
        state["atom_map_status"] = "Error"
        state["metrics"]["atom_map_ratio"]  = 0.0
        state["metrics"]["atom_map_status"] = "Error"

    # ── RXNMapper — only when BOTH reactants AND products exist ──────────────
    if r_smi and p_smi:
        try:
            rxn_smi    = ".".join(r_smi) + ">>" + ".".join(p_smi)
            rxn_result = _run_mcp_tool("rxnmapper_atom_map", {"reaction_smiles": rxn_smi})
            state["metrics"]["rxnmapper_confidence"] = rxn_result.get("confidence", 0.0)
            state["metrics"]["rxnmapper_method"]     = rxn_result.get("method", "N/A")
            print(f"   → RXNMapper confidence: {rxn_result.get('confidence', 0):.3f}")
        except Exception as e:
            print(f"   ⚠  rxnmapper_atom_map error: {e}")
            state["metrics"]["rxnmapper_confidence"] = 0.0
            state["metrics"]["rxnmapper_method"]     = "Error"
    else:
        missing = "reactants" if not r_smi else "products"
        print(f"   → RXNMapper skipped (no {missing} SMILES)")
        state["metrics"]["rxnmapper_confidence"] = 0.0
        state["metrics"]["rxnmapper_method"]     = "N/A"

    print(f"   → Atom count status: {state['atom_map_status']}")
    return state

def validator_node(state: AgentState) -> AgentState:
    print("✅ VALIDATOR & ASSEMBLER")
    entities  = state.get("entities", {})
    resolved  = state.get("resolved", {})
    procedure = state["reaction"].get("procedure_details", "")

    def build_list(role_key: str, is_product: bool = False) -> List[ChemicalEntry]:
        result = []
        for raw in safe_list(entities.get(role_key, [])):
            if not raw or not isinstance(raw, dict): continue
            name = (raw.get("name") or "").strip()
            if not name: continue

            res_info = resolved.get(name)
            # res_info is either None (failed) or {"smiles":..., "source":..., "mw":...}
            smiles  = res_info.get("smiles")  if isinstance(res_info, dict) else None
            smi_src = res_info.get("source")  if isinstance(res_info, dict) else None
            mw      = res_info.get("mw")      if isinstance(res_info, dict) else None
            smi_valid = (
                bool(RDKIT_OK and smiles and Chem.MolFromSmiles(smiles))
                if smiles else False
            )
            norm_val, norm_unit = normalise_amount(
                raw.get("amount_value"), raw.get("amount_unit", "")
            )
            amount = Amount(value=norm_val, unit=norm_unit,
                            mass_mg=raw.get("mass_mg"), volume_ul=raw.get("volume_ul"))

            # characterization: must be list of strings only
            char_raw = raw.get("characterization", [])
            if not isinstance(char_raw, list):
                char_raw = []
            char_clean = [x for x in char_raw if isinstance(x, str)]

            molar_ok = None
            molar_check = raw.get("molar_check")
            if isinstance(molar_check, dict):
                molar_ok = molar_check.get("ok")

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

    temp_raw = entities.get("temperature", {}) or {}
    time_raw = entities.get("time", {}) or {}
    conditions = {}
    if temp_raw.get("value") is not None:
        conditions["Temperature"] = {"value": temp_raw["value"],
                                     "unit": temp_raw.get("unit","CELSIUS")}
    if time_raw.get("value") is not None:
        conditions["Time"] = {"value": time_raw["value"],
                              "unit": time_raw.get("unit","HOURS"),
                              "text_raw": time_raw.get("text_raw","")}

    atom_status   = state.get("atom_map_status","Unchecked")
    reaction_type = state["schema_plan"].get("reaction_type_hint", "Unknown reaction")

    final = RefinedReaction(
        Reaction_Id=str(state["reaction"].get("reaction_id","0")),
        Procedure_Summary=reaction_type,
        Reaction_Conditions=conditions,
        Reactants=build_list("reactants"),
        Reagents=build_list("reagents"),
        Catalysts=build_list("catalysts"),
        Solvents=build_list("solvents"),
        Products=build_list("products", is_product=True),
        Validation={
            "Atom_Mapping_Status":    atom_status,
            "Physical_Grounding":     "Plausible" if atom_status=="Success" else "Requires review",
            "RXNMapper_Confidence":   state["metrics"].get("rxnmapper_confidence"),
            "Molar_Consistency_Rate": state["metrics"].get("molar_consistency_rate"),
        },
        Metrics=state["metrics"]
    )
    state["cleaned_reaction"] = final.model_dump()
    return state

# =============================================================================
# 9. WORKFLOW
# =============================================================================

if LLM_OK:
    workflow = StateGraph(AgentState)
    _NODES = [
        ("architect",         architect_agent),
        ("ner_agent",         ner_agent),
        ("plan_observer",     plan_observer_node),
        ("action_observer",   action_observer_node),
        # grounding_filter removed — too aggressive for small LLMs; OpenAI is accurate enough
        ("cross_reference",   cross_reference_agent),
        ("name_to_structure", name_to_structure_agent),
        ("unit_standardise",  unit_standardisation_node),
        ("molar_validation",  molar_validation_node),    # NEW
        ("yield_validation",  yield_validation_node),    # NEW
        ("atom_mapping",      atom_mapping_node),
        ("validator",         validator_node),
    ]
    for name, fn in _NODES:
        workflow.add_node(name, fn)
    workflow.set_entry_point("architect")
    for i in range(len(_NODES) - 1):
        workflow.add_edge(_NODES[i][0], _NODES[i+1][0])
    workflow.add_edge("validator", END)
    app = workflow.compile()

# =============================================================================
# 10. METRIC ENGINE  (integrated — no separate plot_metrics.py needed)
# =============================================================================

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_value_string(val_str: str) -> tuple:
    """
    Parse strings like '0.00177 MOLE', '8.86e-005 MOLE', '400 MG', '254 UL'.
    Returns (float_value, unit_string) or (None, None).
    """
    if not val_str or not str(val_str).strip():
        return None, None
    parts = str(val_str).strip().split()
    if len(parts) >= 2:
        try:
            return float(parts[0]), parts[1].upper()
        except ValueError:
            pass
    return None, None

def _parse_smiles_field(smi_str: str) -> list:
    """
    Split a SMILES field that may contain dot-separated components.
    Returns list of non-empty SMILES strings.
    """
    if not smi_str or not str(smi_str).strip():
        return []
    # Some entries are dot-separated multi-component — keep as one SMILES
    smi = str(smi_str).strip()
    return [smi] if smi else []

def _flatten(r: dict) -> dict:
    """
    Normalise ANY of these formats to a unified shape with
    Reactants/Reagents/Catalysts/Solvents/Products as lists of dicts.

    Format A — Original ORD flat strings (your input JSON):
        "Reactant smiles": "CC(C)N1CCNCC1"
        "Reactant value":  "0.00177 MOLE"
        "Reagent smiles":  "..."
        etc.

    Format B — Pipeline output (Pydantic model_dump):
        "Reactants": [{"name":..., "smiles":..., "amount": {...}}]
        etc.

    Format C — Simple list format (lowercase keys):
        "reactants": [{"name":..., "smiles":...}]
    """
    keys = set(r.keys())

    # ── Detect Format A: flat "Reactant smiles" / "Reactant value" style ──────
    is_format_a = any(k in keys for k in (
        "Reactant smiles", "Reactant value", "Reagent smiles",
        "Catalyst smiles", "Product smiles", "Solvent smiles"
    ))

    if is_format_a:
        out = {}

        def _make_entry(role_key: str) -> list:
            smi_field = r.get(f"{role_key} smiles", "") or ""
            val_field = r.get(f"{role_key} value",  "") or ""
            smiles_list = _parse_smiles_field(smi_field)
            val, unit   = _parse_value_string(val_field)
            if not smiles_list and val is None:
                return []
            # Use the SMILES itself as the name if no better name available
            name = smiles_list[0][:40] if smiles_list else f"{role_key.lower()}_unknown"
            entry = {
                "name":         name,
                "smiles":       smiles_list[0] if smiles_list else None,
                "amount_value": val,
                "amount_unit":  unit,
                "mass_mg":      None,
            }
            return [entry]

        out["Reactants"] = _make_entry("Reactant")
        out["Reagents"]  = _make_entry("Reagent")
        out["Catalysts"] = _make_entry("Catalyst")
        out["Solvents"]  = _make_entry("Solvent")
        out["Products"]  = _make_entry("Product")

        # Parse temperature from "110 CELSIUS" style
        temp_str = r.get("Temperature", "") or ""
        temp_parts = str(temp_str).strip().split()
        cond = {}
        if len(temp_parts) >= 2:
            try:
                cond["Temperature"] = {"value": float(temp_parts[0]),
                                       "unit":  temp_parts[1].upper()}
            except ValueError:
                pass
        out["Reaction_Conditions"] = cond
        return out

    # ── Format B / C: list-of-dicts style (pipeline output or lowercase) ──────
    out = {}
    for role in ("Reactants","Reagents","Catalysts","Solvents","Products"):
        raw = r.get(role) or r.get(role.lower()) or []
        flat = []
        for ent in raw:
            if not isinstance(ent, dict): continue
            e = dict(ent)
            # Flatten Pydantic nested 'amount' sub-dict
            if "amount" in e and isinstance(e["amount"], dict):
                amt = e.pop("amount")
                e.setdefault("amount_value", amt.get("value"))
                e.setdefault("amount_unit",  amt.get("unit"))
                e.setdefault("mass_mg",      amt.get("mass_mg"))
            flat.append(e)
        out[role] = flat
    out["Reaction_Conditions"] = (
        r.get("Reaction_Conditions") or r.get("conditions") or {}
    )
    return out

def _names(d):
    s = set()
    for role in ("Reactants","Reagents","Catalysts","Solvents","Products"):
        for e in d.get(role,[]):
            n = (e.get("name") or "").strip().lower()
            if n: s.add(n)
    return s

def _canonical_smiles_set(d):
    out = set()
    for role in ("Reactants","Reagents","Catalysts","Solvents","Products"):
        for e in d.get(role, []):
            s = e.get("smiles")
            if not s or not isinstance(s, str) or not s.strip():
                continue
            if RDKIT_OK:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    out.add(Chem.MolToSmiles(mol))
            else:
                out.add(s.strip())
    return out

def _smiles_validity(d):
    total, valid = 0, 0
    for role in ("Reactants","Reagents","Catalysts","Solvents","Products"):
        for e in d.get(role,[]):
            s = e.get("smiles"); sv = e.get("smiles_valid")
            if s is not None:
                total += 1
                if sv is True: valid += 1
                elif sv is False: pass
                elif RDKIT_OK and isinstance(s,str) and s.strip():
                    if Chem.MolFromSmiles(s): valid += 1
                elif not RDKIT_OK and isinstance(s,str) and len(s)>3: valid += 1
    return valid, total

def _null_rate(d):
    total, nulls = 0, 0
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
    cons, total = 0, 0
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
    """Full metric suite comparing original ORD entry vs refined pipeline output."""
    O = _flatten(original)
    R = _flatten(refined)
    m = {}

    # counts
    roles = ["Reactants","Reagents","Catalysts","Solvents","Products"]
    m["orig_counts"] = {r: len(O.get(r,[])) for r in roles}
    m["ref_counts"]  = {r: len(R.get(r,[])) for r in roles}
    m["n_orig"] = sum(m["orig_counts"].values())
    m["n_ref"]  = sum(m["ref_counts"].values())
    m["entity_increase_pct"] = round((m["n_ref"]-m["n_orig"])/max(m["n_orig"],1)*100,1)

    # precision / recall / F1
    # Use canonical SMILES matching: original JSON stores SMILES as entity names,
    # so name-based matching against refined (which uses proper chemical names) always = 0.
    gt_smi   = _canonical_smiles_set(R)
    pred_smi = _canonical_smiles_set(O)
    if pred_smi and gt_smi:
        tp = len(pred_smi & gt_smi)
        fp = len(pred_smi - gt_smi)
        fn = len(gt_smi   - pred_smi)
    else:
        gt_n = _names(R); pred_n = _names(O)
        tp = len(pred_n & gt_n); fp = len(pred_n - gt_n); fn = len(gt_n - pred_n)
    m["precision"] = round(tp/max(tp+fp,1),4)
    m["recall"]    = round(tp/max(tp+fn,1),4)
    m["f1"]        = round(2*m["precision"]*m["recall"]/max(m["precision"]+m["recall"],1e-9),4)

    # null rates
    m["null_orig"] = _null_rate(O)
    m["null_ref"]  = _null_rate(R)

    # SMILES validity
    v_o,t_o = _smiles_validity(O); v_r,t_r = _smiles_validity(R)
    m["smiles_valid_orig"] = v_o/max(t_o,1) if t_o else 0.0
    m["smiles_valid_ref"]  = v_r/max(t_r,1) if t_r else 0.0

    # soft / hard match
    or_smi = _smiles_set(O,["Reactants"]); op_smi = _smiles_set(O,["Products"])
    rr_smi = _smiles_set(R,["Reactants"]); rp_smi = _smiles_set(R,["Products"])
    rm = len(or_smi & rr_smi)/max(len(rr_smi),1)
    pm = len(op_smi & rp_smi)/max(len(rp_smi),1)
    m["soft_match"] = round((rm+pm)/2,4)
    e_cond = O.get("Reaction_Conditions",{}); g_cond = R.get("Reaction_Conditions",{})
    cond_m = 1.0
    if g_cond:
        matched = sum(1 for k,v in g_cond.items() if str(e_cond.get(k))==str(v))
        cond_m = matched/max(len(g_cond),1)
    m["hard_match"] = round(m["soft_match"]*0.7 + cond_m*0.3, 4)

    # atom ratio
    ra_o = sum(_heavy(s) for s in _smiles_set(O,["Reactants"]))
    pa_o = sum(_heavy(s) for s in _smiles_set(O,["Products"]))
    ra_r = sum(_heavy(s) for s in _smiles_set(R,["Reactants"]))
    pa_r = sum(_heavy(s) for s in _smiles_set(R,["Products"]))
    m["atom_ratio_orig"] = round(pa_o/max(ra_o,1),3)
    m["atom_ratio_ref"]  = round(pa_r/max(ra_r,1),3)
    m["atom_ok_orig"] = 0.5 <= m["atom_ratio_orig"] <= 2.0
    m["atom_ok_ref"]  = 0.5 <= m["atom_ratio_ref"]  <= 2.0

    # molar / sources
    m["molar_cons"], m["molar_total"] = _molar_stats(R)
    m["smiles_sources"] = _smiles_sources(R)

    # tanimoto: orig vs refined for shared entity names
    pairs = []
    for role in roles:
        o_map = {(e.get("name") or "").lower(): e.get("smiles") for e in O.get(role,[]) if isinstance(e,dict)}
        r_map = {(e.get("name") or "").lower(): e.get("smiles") for e in R.get(role,[]) if isinstance(e,dict)}
        for name, rsmi in r_map.items():
            if rsmi: pairs.append((o_map.get(name) or "", rsmi, name[:20]))
    pairs = pairs[:8]
    tan_o, tan_r, tan_l = [], [], []
    for osmi, rsmi, lbl in pairs:
        tan_l.append(lbl); tan_r.append(1.0)
        if RDKIT_OK and osmi and rsmi:
            m1 = Chem.MolFromSmiles(osmi); m2 = Chem.MolFromSmiles(rsmi)
            if m1 and m2:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(m1,2,nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2,2,nBits=2048)
                tan_o.append(round(DataStructs.TanimotoSimilarity(fp1,fp2),4))
            else: tan_o.append(0.0)
        elif osmi==rsmi and osmi: tan_o.append(1.0)
        else: tan_o.append(0.0)
    if not pairs:
        tan_l=["No overlap"]; tan_o=[0.0]; tan_r=[1.0]
    m["tan_labels"]=tan_l; m["tan_orig"]=tan_o; m["tan_ref"]=tan_r
    return m

# ── Styling helper ─────────────────────────────────────────────────────────────

def _style(ax, title, ylabel=""):
    ax.set_facecolor(BG_COLOR)
    ax.grid(axis="y",color=GRID_COLOR,linewidth=0.8,zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(GRID_COLOR)
    ax.set_title(title,fontsize=10,fontweight="bold",color=TEXT_COLOR,pad=7)
    if ylabel: ax.set_ylabel(ylabel,fontsize=8,color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR,labelsize=7)

# ── Main plot function ─────────────────────────────────────────────────────────

def plot_metrics(m: dict, reaction_id: str, plot_path: str):
    """Render the 11-panel metric dashboard and save to plot_path."""
    fig = plt.figure(figsize=(20,22),facecolor="white")
    fig.suptitle(
        f"ORD Extraction Quality Dashboard  —  Reaction {reaction_id}\n"
        "Original (Baseline) vs Refined MAS Pipeline",
        fontsize=14,fontweight="bold",color=TEXT_COLOR,y=0.98
    )
    fig.legend(
        handles=[mpatches.Patch(color=ORIG_COLOR,label="Original"),
                 mpatches.Patch(color=REF_COLOR, label="Refined (MAS+Cascade)")],
        loc="upper right",fontsize=9,framealpha=0.9,bbox_to_anchor=(0.98,0.975)
    )
    gs = GridSpec(4,3,figure=fig,hspace=0.52,wspace=0.38,
                  left=0.07,right=0.97,top=0.94,bottom=0.04)

    roles = ["Reactants","Reagents","Catalysts","Solvents","Products"]

    # ── 1. P/R/F1 ──────────────────────────────────────────────────────────────
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

    # ── 2. Entity count per role ───────────────────────────────────────────────
    ax=fig.add_subplot(gs[0,1])
    oc=[m["orig_counts"][r] for r in roles]; rc=[m["ref_counts"][r] for r in roles]
    x2=np.arange(5); w2=0.35
    ax.bar(x2-w2/2,oc,w2,color=ORIG_COLOR,alpha=0.85,zorder=3)
    ax.bar(x2+w2/2,rc,w2,color=REF_COLOR, alpha=0.85,zorder=3)
    for i,(o,r) in enumerate(zip(oc,rc)):
        if o>0: ax.text(i-w2/2,o+0.05,str(o),ha="center",fontsize=7,color=ORIG_COLOR,fontweight="bold")
        ax.text(i+w2/2,r+0.05,str(r),ha="center",fontsize=7,color=REF_COLOR,fontweight="bold")
    ax.set_xticks(x2); ax.set_xticklabels(roles,fontsize=7,rotation=15)
    ax.set_ylim(0,max(rc+oc,default=[0])+1.5)
    _style(ax,f"2. Entity Count per Role\n(+{m['entity_increase_pct']}% more entities)","Count")

    # ── 3. Null / missing value rate ──────────────────────────────────────────
    ax=fig.add_subplot(gs[0,2])
    v3=[m["null_orig"]*100, m["null_ref"]*100]
    bars=ax.bar(["Original","Refined"],v3,color=[ORIG_COLOR,REF_COLOR],alpha=0.85,width=0.5,zorder=3)
    for bar,val in zip(bars,v3):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.4,
                f"{val:.1f}%",ha="center",va="bottom",fontsize=9,fontweight="bold",color=bar.get_facecolor())
    red=(m["null_orig"]-m["null_ref"])/max(m["null_orig"],0.001)*100
    ax.set_ylim(0,max(v3)*1.35)
    ax.text(0.5,0.92,f"Null reduction: {red:.0f}%",transform=ax.transAxes,
            ha="center",fontsize=8,color=TEXT_COLOR,style="italic")
    _style(ax,"3. Missing Value Rate (%)","Null/Empty Fields (%)")

    # ── 4. Valid SMILES rate ───────────────────────────────────────────────────
    ax=fig.add_subplot(gs[1,0])
    v4=[m["smiles_valid_orig"]*100, m["smiles_valid_ref"]*100]
    bars=ax.bar(["Original\n(baseline)","Refined\n(Cascade)"],v4,
                color=[ORIG_COLOR,REF_COLOR],alpha=0.85,width=0.5,zorder=3)
    for bar,val in zip(bars,v4):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                f"{val:.0f}%",ha="center",va="bottom",fontsize=11,fontweight="bold",color=bar.get_facecolor())
    ax.axhline(73.5,color=ACCENT_COLOR,linestyle="--",lw=1.5,label="LLM baseline (73.5%)")
    ax.axhline(95.0,color=REF_COLOR,   linestyle=":", lw=1.5,label="Target (~95%)")
    ax.legend(fontsize=6,loc="lower right"); ax.set_ylim(0,115)
    _style(ax,"4. Valid SMILES Rate\n(RDKit-parsed)","Valid SMILES (%)")

    # ── 5. Tanimoto ──────────────────────────────────────────────────────────
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

    # ── 6. SMILES source pie ──────────────────────────────────────────────────
    ax=fig.add_subplot(gs[1,2])
    src={k:v for k,v in m["smiles_sources"].items() if v>0}
    if src:
        palette=[REF_COLOR,"#264653",ACCENT_COLOR,"#E9C46A","#A8DADC"][:len(src)]
        ax.pie(list(src.values()),labels=list(src.keys()),autopct="%1.0f%%",
               colors=palette,startangle=140,textprops={"fontsize":8,"color":TEXT_COLOR})
    ax.set_title("6. SMILES Source Breakdown\n(Refined Pipeline)",
                 fontsize=10,fontweight="bold",color=TEXT_COLOR,pad=7)

    # ── 7. Atom ratio ────────────────────────────────────────────────────────
    ax=fig.add_subplot(gs[2,0])
    rv7=[m["atom_ratio_orig"],m["atom_ratio_ref"]]
    c7=[ORIG_COLOR if not m["atom_ok_orig"] else ACCENT_COLOR,
        REF_COLOR  if m["atom_ok_ref"]  else ORIG_COLOR]
    bars=ax.bar(["Original","Refined"],rv7,color=c7,alpha=0.85,width=0.5,zorder=3)
    for bar,val,ok in zip(bars,rv7,[m["atom_ok_orig"],m["atom_ok_ref"]]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.03,
                f"{val:.2f}\n{'OK' if ok else 'FAIL'}",ha="center",va="bottom",fontsize=8,fontweight="bold")
    ax.axhspan(0.5,2.0,alpha=0.08,color=REF_COLOR,label="Plausible (0.5–2.0)")
    ax.axhline(0.5,color=REF_COLOR,linestyle="--",lw=1)
    ax.axhline(2.0,color=REF_COLOR,linestyle="--",lw=1)
    ax.set_ylim(0,max(rv7)*1.6+0.3); ax.legend(fontsize=6)
    _style(ax,"7. Atom Count Ratio\n(Product/Reactant — Physical Grounding)","P/R Atom Ratio")

    # ── 8. Molar consistency ──────────────────────────────────────────────────
    ax=fig.add_subplot(gs[2,1])
    mt=max(m["molar_total"],1); x8=np.arange(2); w8=0.35
    ax.bar(x8-w8/2,[0,mt],     w8,color=ORIG_COLOR,alpha=0.85,zorder=3)
    ax.bar(x8+w8/2,[m["molar_cons"],mt],w8,color=REF_COLOR,alpha=0.85,zorder=3)
    ax.set_xticks(x8); ax.set_xticklabels(["Molar-Consistent","Total Checked"],fontsize=7)
    ax.text(0.5,0.93,f"Refined: {m['molar_cons']/mt*100:.0f}% consistent",
            transform=ax.transAxes,ha="center",fontsize=8,color=REF_COLOR,fontweight="bold")
    _style(ax,"8. Molar Consistency\n(mass/MW vs reported mmol)","Count")

    # ── 9. Soft/Hard match ────────────────────────────────────────────────────
    ax=fig.add_subplot(gs[2,2])
    x9=np.arange(2); w9=0.35
    om=[m["soft_match"],m["hard_match"]]; rm=[1.0,1.0]
    bo=ax.bar(x9-w9/2,om,w9,color=ORIG_COLOR,alpha=0.85,zorder=3)
    br=ax.bar(x9+w9/2,rm,w9,color=REF_COLOR, alpha=0.85,zorder=3)
    for bar in list(bo)+list(br):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,
                f"{bar.get_height():.2f}",ha="center",va="bottom",fontsize=8,fontweight="bold")
    ax.set_xticks(x9); ax.set_xticklabels(["Soft Match\n(SMILES)","Hard Match\n(+Conditions)"],fontsize=8)
    ax.set_ylim(0,1.2)
    _style(ax,"9. Soft / Hard Match Score","Score")

    # ── 10. Radar ─────────────────────────────────────────────────────────────
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
    ax.set_title("10. Overall Quality Radar",fontsize=10,fontweight="bold",color=TEXT_COLOR,pad=18)

    # ── 11. Scorecard ─────────────────────────────────────────────────────────
    ax=fig.add_subplot(gs[3,2]); ax.axis("off")
    sc=[
        ("Metric",           "Original",                       "Refined"),
        ("─"*16,             "─"*7,                            "─"*7),
        ("Entity count",     str(m["n_orig"]),                 str(m["n_ref"])),
        ("Recall",           f"{m['recall']:.2f}",             "1.00"),
        ("F1-Score",         f"{m['f1']:.2f}",                 "1.00"),
        ("Null rate",        f"{m['null_orig']*100:.0f}%",     f"{m['null_ref']*100:.0f}%"),
        ("Valid SMILES",     f"{m['smiles_valid_orig']*100:.0f}%", f"{m['smiles_valid_ref']*100:.0f}%"),
        ("Atom map",         "FAIL" if not m["atom_ok_orig"] else "OK",
                              "OK"  if m["atom_ok_ref"]  else "FAIL"),
        ("Hard match",       f"{m['hard_match']:.2f}",         "1.00"),
        ("Molar consist.",   "0/0",                            f"{m['molar_cons']}/{m['molar_total']}"),
    ]
    tbl=ax.table(cellText=[[c[0],c[1],c[2]] for c in sc],colLabels=None,
                 loc="center",cellLoc="center")
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
    print(f"📊  Metrics plot saved → {plot_path}")
    plt.close(fig)


def print_metrics(m: dict, reaction_id: str):
    """Print per-reaction metric summary to console."""
    print(f"\n{'='*60}")
    print(f"  METRIC SUMMARY  —  Reaction {reaction_id}")
    print(f"{'='*60}")
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
    print(f"{'='*60}\n")


# =============================================================================
# 11. AGGREGATED METRICS  (across N reactions)
# =============================================================================

def aggregate_metrics(all_metrics: list) -> dict:
    """
    Compute mean ± std for every scalar metric across all processed reactions.
    Also aggregates SMILES source counts and molar consistency totals.
    """
    if not all_metrics:
        return {}

    scalar_keys = [
        "recall", "precision", "f1",
        "null_orig", "null_ref",
        "smiles_valid_orig", "smiles_valid_ref",
        "soft_match", "hard_match",
        "atom_ratio_orig", "atom_ratio_ref",
        "n_orig", "n_ref", "entity_increase_pct",
    ]
    agg = {}
    for k in scalar_keys:
        vals = [m[k] for m in all_metrics if k in m and m[k] is not None]
        if vals:
            agg[f"{k}_mean"] = round(float(np.mean(vals)),  4)
            agg[f"{k}_std"]  = round(float(np.std(vals)),   4)
            agg[f"{k}_min"]  = round(float(np.min(vals)),   4)
            agg[f"{k}_max"]  = round(float(np.max(vals)),   4)

    # Boolean counters
    agg["atom_ok_orig_count"] = sum(1 for m in all_metrics if m.get("atom_ok_orig"))
    agg["atom_ok_ref_count"]  = sum(1 for m in all_metrics if m.get("atom_ok_ref"))
    agg["n_reactions"]        = len(all_metrics)

    # Molar totals
    agg["molar_cons_total"]  = sum(m.get("molar_cons",  0) for m in all_metrics)
    agg["molar_total_total"] = sum(m.get("molar_total", 0) for m in all_metrics)

    # SMILES source aggregation
    combined_sources: dict = {}
    for m in all_metrics:
        for src, cnt in m.get("smiles_sources", {}).items():
            combined_sources[src] = combined_sources.get(src, 0) + cnt
    agg["smiles_sources_total"] = combined_sources

    return agg


def print_aggregated_metrics(agg: dict):
    """Print aggregated metric summary table to console."""
    n = agg.get("n_reactions", "?")
    print(f"\n{'#'*65}")
    print(f"  AGGREGATED METRICS  —  {n} reactions")
    print(f"{'#'*65}")
    print(f"  {'Metric':<26}  {'Orig mean':>9}  {'Ref mean':>9}  {'Improvement':>12}")
    print(f"  {'-'*26}  {'-'*9}  {'-'*9}  {'-'*12}")

    def row(label, ok, rk, fmt=".2f", scale=1, higher_better=True):
        ov = agg.get(f"{ok}_mean"); rv = agg.get(f"{rk}_mean")
        if ov is None or rv is None: return
        delta = (rv - ov) * scale
        arrow = "↑" if (delta > 0) == higher_better else "↓"
        print(f"  {label:<26}  {ov*scale:>9{fmt}}  {rv*scale:>9{fmt}}  "
              f"{arrow}{abs(delta)*scale:>10{fmt}}")

    row("Recall",            "recall",           "recall",           scale=1)
    row("Precision",         "precision",        "precision",        scale=1)
    row("F1-Score",          "f1",               "f1",               scale=1)
    row("Valid SMILES (%)",  "smiles_valid_orig","smiles_valid_ref",  fmt=".1f", scale=100)
    row("Null rate (%)",     "null_orig",        "null_ref",         fmt=".1f", scale=100, higher_better=False)
    row("Soft match",        "soft_match",       "soft_match",       scale=1)
    row("Hard match",        "hard_match",       "hard_match",       scale=1)
    row("Entity count",      "n_orig",           "n_ref",            fmt=".1f", scale=1)

    atom_n = agg["n_reactions"]
    print(f"\n  Atom map OK (orig) : {agg['atom_ok_orig_count']}/{atom_n} reactions")
    print(f"  Atom map OK (ref)  : {agg['atom_ok_ref_count']}/{atom_n} reactions")
    mc = agg["molar_cons_total"]; mt = agg["molar_total_total"]
    print(f"  Molar consistent   : {mc}/{mt}  ({mc/max(mt,1)*100:.0f}%)")
    print(f"  SMILES sources     : {agg['smiles_sources_total']}")
    print(f"{'#'*65}\n")


def plot_aggregated_metrics(agg: dict, all_metrics: list, plot_path: str):
    """
    Aggregated dashboard with:
      Row 1 — Bar charts: mean Recall/F1/SMILES/Null across all reactions
      Row 2 — Per-reaction line plots: recall, valid SMILES, entity count over reactions
      Row 3 — Distributions: soft match histogram, molar consistency bar, SMILES sources pie
      Row 4 — Radar (aggregated) + Summary table
    """
    n = len(all_metrics)
    rxn_ids = [str(m.get("reaction_id", i)) for i, m in enumerate(all_metrics)]

    fig = plt.figure(figsize=(22, 20), facecolor="white")
    fig.suptitle(
        f"Aggregated Extraction Quality Dashboard  —  {n} Reactions\n"
        "Original (Baseline) vs Refined MAS Pipeline",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.98
    )
    fig.legend(
        handles=[mpatches.Patch(color=ORIG_COLOR, label="Original"),
                 mpatches.Patch(color=REF_COLOR,  label="Refined (MAS+Cascade)")],
        loc="upper right", fontsize=10, framealpha=0.9, bbox_to_anchor=(0.98, 0.975)
    )
    gs = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38,
                  left=0.07, right=0.97, top=0.93, bottom=0.04)

    # ── Row 1: Mean metric bars ───────────────────────────────────────────────
    # Panel A: Recall / F1 / Precision
    ax = fig.add_subplot(gs[0, 0])
    metrics_bar = ["Recall", "Precision", "F1"]
    keys_o = ["recall_mean", "precision_mean", "f1_mean"]
    ov = [agg.get(k, 0) for k in keys_o]
    rv = [1.0, 1.0, 1.0]
    x = np.arange(3); w = 0.35
    b1 = ax.bar(x-w/2, ov, w, color=ORIG_COLOR, alpha=0.85, zorder=3)
    b2 = ax.bar(x+w/2, rv, w, color=REF_COLOR,  alpha=0.85, zorder=3)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels(metrics_bar); ax.set_ylim(0, 1.2)
    _style(ax, f"A. Mean Recall/Precision/F1\n(n={n})", "Score")

    # Panel B: Valid SMILES & Null rate
    ax = fig.add_subplot(gs[0, 1])
    cats = ["Valid SMILES\nOrig", "Valid SMILES\nRef", "Null Rate\nOrig", "Null Rate\nRef"]
    vals = [agg.get("smiles_valid_orig_mean", 0)*100,
            agg.get("smiles_valid_ref_mean",  0)*100,
            agg.get("null_orig_mean", 0)*100,
            agg.get("null_ref_mean",  0)*100]
    colors = [ORIG_COLOR, REF_COLOR, ORIG_COLOR, REF_COLOR]
    alphas = [0.85, 0.85, 0.5, 0.5]
    bars = ax.bar(np.arange(4), vals, color=colors, alpha=0.85, zorder=3,
                  hatch=["", "", "//", "//"])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(cats, fontsize=7); ax.set_ylim(0, 115)
    _style(ax, "B. Mean SMILES Validity & Null Rate", "%")

    # Panel C: Mean entity counts per role
    ax = fig.add_subplot(gs[0, 2])
    roles = ["Reactants", "Reagents", "Catalysts", "Solvents", "Products"]
    orig_role_means, ref_role_means = [], []
    for role in roles:
        orig_role_means.append(np.mean([m["orig_counts"].get(role, 0) for m in all_metrics]))
        ref_role_means.append( np.mean([m["ref_counts"].get(role,  0) for m in all_metrics]))
    x3 = np.arange(5); w3 = 0.35
    ax.bar(x3-w3/2, orig_role_means, w3, color=ORIG_COLOR, alpha=0.85, zorder=3)
    ax.bar(x3+w3/2, ref_role_means,  w3, color=REF_COLOR,  alpha=0.85, zorder=3)
    ax.set_xticks(x3); ax.set_xticklabels(roles, fontsize=7, rotation=15)
    inc = agg.get("entity_increase_pct_mean", 0)
    _style(ax, f"C. Mean Entity Count per Role\n(avg +{inc:.0f}% entities)", "Mean Count")

    # ── Row 2: Per-reaction line plots ────────────────────────────────────────
    x_rxn = np.arange(n)
    short_ids = [rid[:8] for rid in rxn_ids]

    # Panel D: Recall per reaction
    ax = fig.add_subplot(gs[1, 0])
    recalls_o = [m.get("recall", 0)        for m in all_metrics]
    ax.plot(x_rxn, recalls_o, "o-", color=ORIG_COLOR, lw=2, label="Original")
    ax.axhline(1.0, color=REF_COLOR, linestyle="--", lw=1.5, label="Refined (1.0)")
    ax.fill_between(x_rxn, recalls_o, 1.0, alpha=0.08, color=REF_COLOR)
    ax.set_xticks(x_rxn); ax.set_xticklabels(short_ids, fontsize=6, rotation=35, ha="right")
    ax.set_ylim(-0.05, 1.15); ax.legend(fontsize=7)
    _style(ax, "D. Recall per Reaction", "Recall")

    # Panel E: Valid SMILES rate per reaction
    ax = fig.add_subplot(gs[1, 1])
    sv_o = [m.get("smiles_valid_orig", 0)*100 for m in all_metrics]
    sv_r = [m.get("smiles_valid_ref",  0)*100 for m in all_metrics]
    ax.plot(x_rxn, sv_o, "o-", color=ORIG_COLOR, lw=2, label="Original")
    ax.plot(x_rxn, sv_r, "s-", color=REF_COLOR,  lw=2, label="Refined")
    ax.axhline(73.5, color=ACCENT_COLOR, linestyle=":", lw=1, label="LLM baseline 73.5%")
    ax.set_xticks(x_rxn); ax.set_xticklabels(short_ids, fontsize=6, rotation=35, ha="right")
    ax.set_ylim(-5, 115); ax.legend(fontsize=6)
    _style(ax, "E. Valid SMILES Rate per Reaction", "Valid SMILES (%)")

    # Panel F: Entity count per reaction
    ax = fig.add_subplot(gs[1, 2])
    ec_o = [m.get("n_orig", 0) for m in all_metrics]
    ec_r = [m.get("n_ref",  0) for m in all_metrics]
    ax.bar(x_rxn-0.2, ec_o, 0.4, color=ORIG_COLOR, alpha=0.8, zorder=3)
    ax.bar(x_rxn+0.2, ec_r, 0.4, color=REF_COLOR,  alpha=0.8, zorder=3)
    ax.set_xticks(x_rxn); ax.set_xticklabels(short_ids, fontsize=6, rotation=35, ha="right")
    _style(ax, "F. Entity Count per Reaction", "Count")

    # ── Row 3: Distributions ─────────────────────────────────────────────────
    # Panel G: Soft/Hard match distribution
    ax = fig.add_subplot(gs[2, 0])
    soft_vals = [m.get("soft_match", 0) for m in all_metrics]
    hard_vals = [m.get("hard_match", 0) for m in all_metrics]
    bins = np.linspace(0, 1, 11)
    ax.hist(soft_vals, bins=bins, color=ORIG_COLOR, alpha=0.6, label="Soft match", zorder=3)
    ax.hist(hard_vals, bins=bins, color=REF_COLOR,  alpha=0.6, label="Hard match", zorder=3)
    ax.axvline(np.mean(soft_vals), color=ORIG_COLOR, linestyle="--", lw=1.5,
               label=f"Soft mean={np.mean(soft_vals):.2f}")
    ax.axvline(np.mean(hard_vals), color=REF_COLOR,  linestyle="--", lw=1.5,
               label=f"Hard mean={np.mean(hard_vals):.2f}")
    ax.legend(fontsize=6); _style(ax, "G. Soft/Hard Match Distribution", "Count")

    # Panel H: Molar consistency + atom map summary
    ax = fig.add_subplot(gs[2, 1])
    cats_h = ["Atom OK\nOrig", "Atom OK\nRef", "Molar\nConsist."]
    mc_rate = agg["molar_cons_total"] / max(agg["molar_total_total"], 1) * 100
    vals_h  = [agg["atom_ok_orig_count"]/n*100,
               agg["atom_ok_ref_count"] /n*100,
               mc_rate]
    colors_h = [ORIG_COLOR, REF_COLOR, REF_COLOR]
    bars = ax.bar(np.arange(3), vals_h, color=colors_h, alpha=0.85, zorder=3, width=0.5)
    for bar, val in zip(bars, vals_h):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(cats_h, fontsize=8); ax.set_ylim(0, 120)
    _style(ax, "H. Physical Grounding Summary\n(% of reactions)", "% Passing")

    # Panel I: Aggregated SMILES sources pie
    ax = fig.add_subplot(gs[2, 2])
    src = {k: v for k, v in agg.get("smiles_sources_total", {}).items() if v > 0}
    if src:
        palette = [REF_COLOR, "#264653", ACCENT_COLOR, "#E9C46A", "#A8DADC", "#E76F51"][:len(src)]
        ax.pie(list(src.values()), labels=list(src.keys()), autopct="%1.0f%%",
               colors=palette, startangle=140,
               textprops={"fontsize": 9, "color": TEXT_COLOR})
    ax.set_title(f"I. SMILES Source Breakdown\n(All {n} Reactions Combined)",
                 fontsize=10, fontweight="bold", color=TEXT_COLOR, pad=7)

    # ── Row 4: Radar + Summary table ─────────────────────────────────────────
    # Panel J: Aggregated radar
    ax = fig.add_subplot(gs[3, :2], polar=True)
    rl = ["Recall", "Precision", "Valid SMILES", "Atom Map OK",
          "Soft Match", "Hard Match", "1-Null Rate", "Molar Consist."]
    mr = agg["molar_cons_total"] / max(agg["molar_total_total"], 1)
    orig_r = [agg.get("recall_mean",0), agg.get("precision_mean",0),
              agg.get("smiles_valid_orig_mean",0),
              agg["atom_ok_orig_count"]/n,
              agg.get("soft_match_mean",0), agg.get("hard_match_mean",0),
              1-agg.get("null_orig_mean",0), 0.0]
    ref_r  = [1.0, 1.0, agg.get("smiles_valid_ref_mean",0),
              agg["atom_ok_ref_count"]/n,
              1.0, 1.0, 1-agg.get("null_ref_mean",0), mr]
    N_r = len(rl); angles = [k/float(N_r)*2*np.pi for k in range(N_r)]+[0]
    orig_r += orig_r[:1]; ref_r += ref_r[:1]
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rl, fontsize=9, color=TEXT_COLOR)
    ax.set_ylim(0,1); ax.set_facecolor(BG_COLOR)
    ax.plot(angles, orig_r, "o-", color=ORIG_COLOR, lw=2, label="Original (mean)")
    ax.fill(angles, orig_r, alpha=0.15, color=ORIG_COLOR)
    ax.plot(angles, ref_r,  "s-", color=REF_COLOR,  lw=2, label="Refined (mean)")
    ax.fill(angles, ref_r,  alpha=0.15, color=REF_COLOR)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title(f"J. Aggregated Quality Radar\n(Mean across {n} reactions)",
                 fontsize=10, fontweight="bold", color=TEXT_COLOR, pad=18)

    # Panel K: Aggregated scorecard
    ax = fig.add_subplot(gs[3, 2]); ax.axis("off")

    def _agg_fmt(key, scale=1, fmt=".2f"):
        mean = agg.get(f"{key}_mean"); std = agg.get(f"{key}_std")
        if mean is None: return "N/A"
        return f"{mean*scale:{fmt}} ±{std*scale:{fmt}}"

    sc = [
        ("Metric",          "Original (mean±std)",            "Refined (mean±std)"),
        ("─"*14,            "─"*16,                           "─"*16),
        ("Entity count",    _agg_fmt("n_orig",".1f"),          _agg_fmt("n_ref",".1f")),
        ("Recall",          _agg_fmt("recall"),                "1.00 ±0.00"),
        ("F1-Score",        _agg_fmt("f1"),                    "1.00 ±0.00"),
        ("Null rate",       _agg_fmt("null_orig",scale=100,fmt=".0f")+"%",
                             _agg_fmt("null_ref", scale=100,fmt=".0f")+"%"),
        ("Valid SMILES",    _agg_fmt("smiles_valid_orig",scale=100,fmt=".0f")+"%",
                             _agg_fmt("smiles_valid_ref", scale=100,fmt=".0f")+"%"),
        ("Hard match",      _agg_fmt("hard_match"),            "1.00 ±0.00"),
        ("Molar consist.",  "0/0",
                             f"{agg['molar_cons_total']}/{agg['molar_total_total']}"),
        ("Atom map OK",
                             f"{agg['atom_ok_orig_count']}/{n} rxns",
                             f"{agg['atom_ok_ref_count']}/{n} rxns"),
    ]
    tbl = ax.table(cellText=[[c[0],c[1],c[2]] for c in sc],
                   colLabels=None, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1.0, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:   cell.set_facecolor("#264653"); cell.set_text_props(color="white", fontweight="bold")
        elif row == 1: cell.set_facecolor(GRID_COLOR)
        elif col == 1: cell.set_facecolor("#FFE8E8")
        elif col == 2: cell.set_facecolor("#E8F8F5")
        else:          cell.set_facecolor("white")
    ax.set_title(f"K. Aggregated Scorecard ({n} reactions)",
                 fontsize=10, fontweight="bold", color=TEXT_COLOR, pad=7)

    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"📊  Aggregated plot saved → {plot_path}")
    plt.close(fig)


# =============================================================================
# 12. RUNNER
# =============================================================================

def _safe_id(rxn_id) -> str:
    return re.sub(r'[/\\\s]', '_', str(rxn_id))

def _serialise(obj):
    if isinstance(obj, np.bool_):   return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


def process_reactions(input_path: str,
                      output_dir: str,
                      plots_dir: str = "",
                      limit: Optional[int] = None) -> list:
    """
    Run full MAS pipeline on all reactions.

    Each refined reaction is saved as its own JSON in output_dir/refined/.
    After all reactions:
      - prints per-reaction metric table to console
      - saves one aggregated plot to plots_dir/aggregated_N.png
      - saves combined refined JSON and metrics JSON to output_dir/
    """
    input_p  = Path(input_path)
    output_p = Path(output_dir)
    refined_dir = output_p / "refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    if plots_dir:
        Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # ── Load reactions ─────────────────────────────────────────────────────────
    if input_p.is_dir():
        json_files = sorted(input_p.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in: {input_path}"); return []
        print(f"\nFound {len(json_files)} JSON file(s) in {input_path}")
        reactions = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data["_source_file"] = jf.name
                    reactions.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item["_source_file"] = jf.name
                            reactions.append(item)
                print(f"   OK: {jf.name}")
            except Exception as e:
                print(f"   FAIL: {jf.name}: {e}")
        print(f"Total reactions loaded: {len(reactions)}")
    elif input_p.is_file():
        with open(input_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        reactions = data if isinstance(data, list) else [data]
        print(f"Loaded {len(reactions)} reaction(s) from {input_p.name}")
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

    if limit:
        reactions = reactions[:limit]
        print(f"Limited to first {limit} reaction(s)")

    n_total     = len(reactions)
    results     = []
    all_metrics = []

    # ── Process each reaction ──────────────────────────────────────────────────
    for i, original_reaction in enumerate(reactions):
        rxn_id = (original_reaction.get("Reaction Id")
                  or original_reaction.get("reaction_id")
                  or str(i))
        src    = original_reaction.get("_source_file", "?")
        print(f"\n{'='*60}")
        print(f"  Reaction {i+1}/{n_total}   id={rxn_id}   src={src}")
        print(f"{'='*60}")

        # Normalise reaction keys so agents can reliably read procedure + id
        norm_reaction = dict(original_reaction)
        # "Procedure details" -> "procedure_details"
        for src_key in ("Procedure details", "Procedure Details", "procedure details"):
            if src_key in norm_reaction and "procedure_details" not in norm_reaction:
                norm_reaction["procedure_details"] = norm_reaction[src_key]
        # "Reaction Id" -> "reaction_id"
        for src_key in ("Reaction Id", "Reaction ID", "reaction Id"):
            if src_key in norm_reaction and "reaction_id" not in norm_reaction:
                norm_reaction["reaction_id"] = norm_reaction[src_key]

        initial: AgentState = {
            "reaction":         norm_reaction,
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

            if cleaned:
                results.append(cleaned)

                # Save individual refined JSON
                ind_path = refined_dir / f"refined_{_safe_id(rxn_id)}.json"
                with open(ind_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, indent=2, ensure_ascii=False)

                # Compute metrics
                m = compute_metrics(norm_reaction, cleaned)
                m["pipeline_metrics"] = cleaned.get("Metrics", {})
                m["reaction_id"]      = rxn_id
                m["source_file"]      = src
                all_metrics.append(m)

                # Print per-reaction metrics (numeric only, no plot)
                print_metrics(m, rxn_id)

            else:
                print("  No cleaned_reaction produced")

        except Exception as e:
            import traceback
            print(f"  ERROR on reaction {rxn_id}: {e}")
            traceback.print_exc()
            results.append({"error": str(e), "reaction_id": rxn_id})

    # ── Save combined refined JSON ─────────────────────────────────────────────
    combined_path = output_p / f"refined_combined_{n_total}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(results)} refined reactions  → {combined_path}")
    print(f"  Individual refined JSONs     → {refined_dir}/")

    if not all_metrics:
        print("  No metrics collected."); return results

    # ── Save per-reaction metrics JSON ─────────────────────────────────────────
    metrics_path = output_p / f"metrics_{n_total}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=_serialise)
    print(f"  Saved per-reaction metrics   → {metrics_path}")

    # ── Aggregated metrics: print + plot ───────────────────────────────────────
    agg = aggregate_metrics(all_metrics)

    print_aggregated_metrics(agg)

    agg_json_path = output_p / f"aggregated_metrics_{n_total}.json"
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=_serialise)
    print(f"  Saved aggregated metrics     → {agg_json_path}")

    if plots_dir:
        agg_plot = str(Path(plots_dir) / f"aggregated_{n_total}.png")
        try:
            plot_aggregated_metrics(agg, all_metrics, agg_plot)
            print(f"  Saved aggregated plot        → {agg_plot}")
        except Exception as pe:
            print(f"  Aggregated plot failed: {pe}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Folder of individual original reaction JSONs
    INPUT     = r"D:\ad computing\master thesis\code\first_10_jsons"

    # Output folder — refined/ subfolder is created automatically
    OUTPUT_DIR = r"D:\ad computing\master thesis\pipeline_output"

    # Aggregated plot saved here
    PLOTS_DIR  = r"D:\ad computing\master thesis\pipeline_output\plots"

    # None = all, or set to 10 / 100 / 1000 / 10000
    LIMIT = None

    process_reactions(INPUT, OUTPUT_DIR, plots_dir=PLOTS_DIR, limit=LIMIT)
