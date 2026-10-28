"""
chem_mcp_server_v3_publication_ready.py  –  Publication-Ready FastMCP Server
NEW in v3 (publication improvements):
  16. deduplicate_entities        Canonical SMILES deduplication
  17. classify_role_by_chemistry  Rule-based role correction (Pd→catalyst, base→reagent, etc.)
  18. compute_role_accuracy       Ground truth role matching metric

All v2 tools (1-15) preserved.
"""
import json, time, re
import urllib.request, urllib.parse
from typing import Optional, List

import requests
from mcp.server.fastmcp import FastMCP

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, DataStructs, AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

mcp = FastMCP("ChemToolsV3_PubReady")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_BAD_PATTERNS = [
    re.compile(r"^[A-Z][a-z]?(\d+)([A-Z][a-z]?(\d+)?)*$"),
    re.compile(r"^InChI="),
    re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"),
    re.compile(r"^\s*(NULL|NONE|N/A)\s*$", re.I),
    re.compile(r"\({10,}"),
]

def _is_bad(s: str) -> bool:
    s = (s or "").strip()
    if not s: return True
    return any(p.search(s) for p in _BAD_PATTERNS)

def _rdkit_canonicalize(smiles: str) -> Optional[str]:
    if not RDKIT_OK or _is_bad(smiles): return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None

def _pubchem_fetch(name: str) -> dict:
    try:
        enc = urllib.parse.quote(name.strip())
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{enc}/property/IsomericSMILES,CanonicalSMILES,"
            "MolecularWeight,ExactMass/JSON"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "MAS-ChemPipeline/3.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES", "")
        canonical = _rdkit_canonicalize(smiles) or smiles
        return {"smiles": canonical, "cid": props.get("CID"),
                "mw": float(props.get("MolecularWeight") or props.get("ExactMass") or 0)}
    except Exception as e:
        return {"error": str(e)}

def _pubchem_fetch_by_cas(cas: str) -> dict:
    try:
        enc = urllib.parse.quote(cas.strip())
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{enc}/property/IsomericSMILES,MolecularWeight/JSON"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "MAS-ChemPipeline/3.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        smiles = props.get("IsomericSMILES", "")
        return {"smiles": _rdkit_canonicalize(smiles) or smiles,
                "cid": props.get("CID"),
                "mw": float(props.get("MolecularWeight") or 0)}
    except Exception as e:
        return {"error": str(e)}

def _opsin_convert(iupac_name: str) -> Optional[str]:
    try:
        enc  = urllib.parse.quote(iupac_name.strip())
        url  = f"https://opsin.ch.cam.ac.uk/opsin/{enc}.smi"
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "MAS-ChemPipeline/3.0"})
        if resp.status_code == 200:
            smi = resp.text.strip()
            if smi and not _is_bad(smi):
                return _rdkit_canonicalize(smi) or smi
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# V1+V2 TOOLS (preserved from v2)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def pubchem_name_to_smiles(name: str) -> dict:
    """Query PubChem for a chemical name → {smiles, cid, mw} or {error}."""
    return _pubchem_fetch(name)

@mcp.tool()
def validate_and_canonicalize(smiles: str) -> dict:
    """Validate SMILES with RDKit → {valid, canonical, reason}."""
    if _is_bad(smiles):
        return {"valid": False, "canonical": None, "reason": "Formula/InChI/bad nesting"}
    if not RDKIT_OK:
        return {"valid": None, "canonical": None, "reason": "RDKit unavailable"}
    can = _rdkit_canonicalize(smiles)
    if can: return {"valid": True, "canonical": can, "reason": "OK"}
    return {"valid": False, "canonical": None, "reason": "RDKit parse failed"}

@mcp.tool()
def batch_resolve_smiles(names: list[str], delay_s: float = 0.4) -> dict:
    """Resolve list of chemical names to SMILES via PubChem → {results}."""
    results = {}
    for name in names:
        if not name or not name.strip(): continue
        res = _pubchem_fetch(name)
        results[name] = None if ("error" in res or not res.get("smiles")) else {
            "smiles": res["smiles"], "cid": res.get("cid"), "mw": res.get("mw")}
        time.sleep(delay_s)
    return {"results": results}

@mcp.tool()
def compute_mol_weight(smiles: str) -> dict:
    """Compute exact MW from SMILES via RDKit → {mw} or {error}."""
    if not RDKIT_OK: return {"error": "RDKit unavailable"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return {"error": "Invalid SMILES"}
    return {"mw": round(Descriptors.ExactMolWt(mol), 4)}

@mcp.tool()
def mg_to_moles(mass_mg: float, mw: float) -> dict:
    """Convert mg to moles → {moles, mw_used} or {error}."""
    if not mw or mw <= 0: return {"error": f"Invalid MW: {mw}"}
    return {"moles": round(mass_mg / (mw * 1000), 8), "mw_used": mw}

@mcp.tool()
def atom_count_check(reactant_smiles: list[str], product_smiles: list[str]) -> dict:
    """Heavy-atom count heuristic → {status, reactant_atoms, product_atoms, ratio}."""
    if not RDKIT_OK:
        return {"status": "Skipped – RDKit unavailable", "reactant_atoms": 0,
                "product_atoms": 0, "ratio": 0.0}
    def _count(smi_list):
        total = 0
        for s in smi_list:
            if not s: continue
            mol = Chem.MolFromSmiles(s)
            if mol: total += mol.GetNumHeavyAtoms()
        return total
    r, p = _count(reactant_smiles), _count(product_smiles)
    if r == 0 or p == 0:
        status = "Incomplete – missing valid SMILES"
    elif 0.5 <= (p / r) <= 2.0:
        status = "Success"
    else:
        status = f"Warning – atom mismatch (ratio={p/r:.2f})"
    return {"status": status, "reactant_atoms": r,
            "product_atoms": p, "ratio": round(p / r, 3) if r else 0.0}

@mcp.tool()
def sanitize_smiles(smiles: str) -> dict:
    """Sanitize/canonicalize SMILES → {canonical} or {error}."""
    can = _rdkit_canonicalize(smiles)
    if can: return {"canonical": can}
    return {"error": "Cannot parse or sanitize SMILES"}

@mcp.tool()
def opsin_name_to_smiles(iupac_name: str) -> dict:
    """Convert IUPAC name to SMILES using OPSIN web API."""
    smi = _opsin_convert(iupac_name)
    if smi:
        return {"smiles": smi, "source": "OPSIN-WebAPI"}
    return {"error": f"OPSIN web API could not parse: {iupac_name}"}

@mcp.tool()
def rxnmapper_atom_map(reaction_smiles: str) -> dict:
    """Attempt atom-mapping of reaction SMILES using RXNMapper."""
    try:
        from rxnmapper import RXNMapper
        rxn_mapper = RXNMapper()
        results = rxn_mapper.get_attention_guided_atom_maps([reaction_smiles])
        mapped = results[0]
        return {
            "mapped_smiles": mapped.get("mapped_rxn", reaction_smiles),
            "confidence":    round(mapped.get("confidence", 0.0), 4),
            "method":        "rxnmapper"
        }
    except ImportError:
        parts = reaction_smiles.split(">>")
        if len(parts) != 2:
            return {"error": "Invalid reaction SMILES format (expected >>)"}
        reactant_part, product_part = parts
        r_smiles = [s.strip() for s in reactant_part.split(".") if s.strip()]
        p_smiles = [s.strip() for s in product_part.split(".") if s.strip()]
        r_valid = [s for s in r_smiles if _rdkit_canonicalize(s)]
        p_valid = [s for s in p_smiles if _rdkit_canonicalize(s)]
        return {
            "mapped_smiles": reaction_smiles,
            "confidence":    len(r_valid) / max(len(r_smiles), 1) * 0.5 +
                             len(p_valid) / max(len(p_smiles), 1) * 0.5,
            "method":        "validity_heuristic",
            "note":          "Install rxnmapper for full atom mapping"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def tanimoto_similarity(smiles1: str, smiles2: str) -> dict:
    """Compute Morgan fingerprint Tanimoto similarity between two SMILES."""
    if not RDKIT_OK: return {"error": "RDKit unavailable"}
    mol1 = Chem.MolFromSmiles(smiles1) if smiles1 and not _is_bad(smiles1) else None
    mol2 = Chem.MolFromSmiles(smiles2) if smiles2 and not _is_bad(smiles2) else None
    if mol1 is None or mol2 is None:
        return {"tanimoto": 0.0, "valid_pair": False,
                "error": "One or both SMILES invalid"}
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return {"tanimoto": round(DataStructs.TanimotoSimilarity(fp1, fp2), 4),
            "valid_pair": True}

@mcp.tool()
def molar_consistency_check(mass_mg: float, smiles: str,
                             reported_mmol: float) -> dict:
    """Verify reported mmol is consistent with mass_mg and MW from SMILES."""
    if not RDKIT_OK: return {"error": "RDKit unavailable"}
    mol = Chem.MolFromSmiles(smiles) if smiles and not _is_bad(smiles) else None
    if mol is None:
        return {"ok": False, "error": "Invalid SMILES – cannot compute MW"}
    mw = Descriptors.ExactMolWt(mol)
    calc_mmol = round(mass_mg / mw, 4)
    deviation = abs(calc_mmol - reported_mmol) / max(reported_mmol, 1e-9) * 100
    return {
        "ok":             deviation <= 20.0,
        "calculated_mmol": calc_mmol,
        "reported_mmol":  reported_mmol,
        "deviation_pct":  round(deviation, 2),
        "mw_used":        round(mw, 4),
        "flag":           "PASS" if deviation <= 20.0 else "FAIL: >20% mismatch"
    }

@mcp.tool()
def yield_sanity_check(product_mmol: float, limiting_mmol: float,
                        reported_yield_pct: Optional[float] = None) -> dict:
    """Check if yield is chemically plausible (0–100% vs limiting reagent)."""
    if limiting_mmol <= 0:
        return {"ok": False, "error": "Limiting reagent mmol must be > 0"}
    calc_yield = round(product_mmol / limiting_mmol * 100, 2)
    ok = 0.0 <= calc_yield <= 100.0
    flag = "PASS" if ok else f"FAIL: impossible yield {calc_yield:.1f}%"
    result = {"ok": ok, "calculated_yield_pct": calc_yield, "flag": flag}
    if reported_yield_pct is not None:
        deviation = abs(calc_yield - reported_yield_pct)
        result["reported_yield_pct"] = reported_yield_pct
        result["yield_deviation_pct"] = round(deviation, 2)
    return result

@mcp.tool()
def hard_match_score(extracted: dict, ground_truth: dict) -> dict:
    """Compute hard-match and soft-match scores between extracted and GT reaction dicts."""
    def _canon_set(smi_list):
        out = set()
        for s in (smi_list or []):
            c = _rdkit_canonicalize(s) if s else None
            if c: out.add(c)
        return out

    e_r = _canon_set(extracted.get("reactant_smiles", []))
    e_p = _canon_set(extracted.get("product_smiles", []))
    g_r = _canon_set(ground_truth.get("reactant_smiles", []))
    g_p = _canon_set(ground_truth.get("product_smiles", []))

    reactant_match = len(e_r & g_r) / max(len(g_r), 1)
    product_match  = len(e_p & g_p) / max(len(g_p), 1)
    soft_match     = round((reactant_match + product_match) / 2, 4)

    cond_match = 1.0
    e_cond = extracted.get("conditions", {})
    g_cond = ground_truth.get("conditions", {})
    if g_cond:
        matched_cond = sum(1 for k, v in g_cond.items() if e_cond.get(k) == v)
        cond_match = matched_cond / max(len(g_cond), 1)

    hard_match = round(soft_match * 0.7 + cond_match * 0.3, 4)

    return {
        "hard_match":      hard_match,
        "soft_match":      soft_match,
        "breakdown": {
            "reactant_recall": round(reactant_match, 4),
            "product_recall":  round(product_match, 4),
            "condition_match": round(cond_match, 4),
        }
    }

@mcp.tool()
def smiles_from_cas(cas_number: str) -> dict:
    """Resolve CAS number to SMILES via PubChem → {smiles, cid, mw} or {error}."""
    return _pubchem_fetch_by_cas(cas_number)

@mcp.tool()
def resolve_smiles_cascade(name: str) -> dict:
    """
    Cascade: OPSIN web API → PubChem REST → NCI CIR → None
    Returns {smiles, source, mw} or {smiles: null, tried, error}.
    """
    tried = []

    # Step 1: OPSIN
    tried.append("OPSIN-WebAPI")
    smi = _opsin_convert(name)
    if smi:
        mol = Chem.MolFromSmiles(smi) if RDKIT_OK else None
        mw  = round(Descriptors.ExactMolWt(mol), 4) if mol else None
        return {"smiles": smi, "source": "OPSIN-WebAPI", "mw": mw}

    # Step 2: PubChem
    tried.append("PubChem_name")
    time.sleep(0.2)
    res = _pubchem_fetch(name)
    if "error" not in res and res.get("smiles"):
        return {"smiles": res["smiles"], "source": "PubChem", "mw": res.get("mw")}

    # Step 3: CIR
    tried.append("CIR")
    try:
        enc = urllib.parse.quote(name)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{enc}/smiles"
        req = urllib.request.Request(url, headers={"User-Agent": "MAS-ChemPipeline/3.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw_smi = resp.read().decode().strip()
        if raw_smi and not _is_bad(raw_smi):
            can = _rdkit_canonicalize(raw_smi)
            if can:
                mol = Chem.MolFromSmiles(can) if RDKIT_OK else None
                mw = round(Descriptors.ExactMolWt(mol), 4) if mol else None
                return {"smiles": can, "source": "CIR", "mw": mw}
    except Exception:
        pass

    return {"smiles": None, "tried": tried,
            "error": f"No valid SMILES found for '{name}' via any deterministic source"}

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 16 – DEDUPLICATION (Publication Improvement #1)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def deduplicate_entities(entities: list[dict]) -> dict:
    """
    PUBLICATION IMPROVEMENT #1: Deduplicate entities by canonical SMILES.
    
    Input: list of entity dicts with keys: name, smiles, role, amount, etc.
    Returns: {
        "deduplicated": [...],
        "duplicate_count": int,
        "duplicate_rate": float,
        "merged_groups": [[original_names], ...]
    }
    
    Strategy:
    - Group entities by canonical SMILES
    - For each group, keep the most complete entry (most non-null fields)
    - Merge amount/characterization data where appropriate
    - Track which entities were merged
    """
    if not entities:
        return {"deduplicated": [], "duplicate_count": 0, "duplicate_rate": 0.0,
                "merged_groups": []}
    
    # Group by canonical SMILES
    smiles_groups = {}
    no_smiles_group = []
    
    for ent in entities:
        if not isinstance(ent, dict):
            continue
            
        smi = ent.get("smiles")
        if smi and isinstance(smi, str) and smi.strip():
            # Canonicalize
            canonical = _rdkit_canonicalize(smi) if RDKIT_OK else smi.strip()
            if canonical:
                if canonical not in smiles_groups:
                    smiles_groups[canonical] = []
                smiles_groups[canonical].append(ent)
            else:
                no_smiles_group.append(ent)
        else:
            no_smiles_group.append(ent)
    
    # Merge each group
    deduplicated = []
    merged_groups = []
    duplicate_count = 0
    
    for canonical_smi, group in smiles_groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Multiple entities with same SMILES - merge them
            duplicate_count += len(group) - 1
            merged_groups.append([e.get("name", "?") for e in group])
            
            # Pick best entry: most non-null fields
            def _completeness_score(e):
                return sum(1 for v in e.values() 
                          if v is not None and v != "" and v != "NOT_SPECIFIED")
            
            best = max(group, key=_completeness_score)
            
            # Merge characterization data if available
            all_char = []
            for e in group:
                char = e.get("characterization", [])
                if isinstance(char, list):
                    all_char.extend([c for c in char if isinstance(c, str)])
            if all_char:
                best["characterization"] = list(dict.fromkeys(all_char))
            
            # Add merged flag
            best["_merged_from"] = [e.get("name", "?") for e in group]
            deduplicated.append(best)
    
    # Add non-SMILES entities as-is
    deduplicated.extend(no_smiles_group)
    
    duplicate_rate = duplicate_count / max(len(entities), 1)
    
    return {
        "deduplicated": deduplicated,
        "duplicate_count": duplicate_count,
        "unique_entity_ratio": round(len(deduplicated) / max(len(entities), 1), 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "merged_groups": merged_groups
    }

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 17 – ROLE CLASSIFICATION (Publication Improvement #2)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def classify_role_by_chemistry(entity_name: str, entity_smiles: Optional[str],
                                 current_role: str, all_roles: list[str]) -> dict:
    """
    PUBLICATION IMPROVEMENT #2: Rule-based role correction using chemical knowledge.
    
    Prevents common misclassifications:
    - Pd/Cu/Pt compounds → Catalyst
    - Bases (Cs2CO3, K2CO3, NaOH, etc.) → Reagent (not catalyst)
    - Solvents → Solvent (never reactant)
    - Ligands → Reagent
    
    Returns: {
        "suggested_role": str,
        "confidence": float (0-1),
        "reasoning": str,
        "role_changed": bool
    }
    """
    name_lower = (entity_name or "").lower()
    smi = (entity_smiles or "").strip()
    
    # Rule 1: Metal catalysts
    catalyst_metals = ["pd", "palladium", "pt", "platinum", "cu", "copper", 
                       "ni", "nickel", "rh", "rhodium", "ru", "ruthenium",
                       "ir", "iridium", "au", "gold"]
    
    for metal in catalyst_metals:
        if metal in name_lower:
            if "catalyst" not in all_roles:
                return {
                    "suggested_role": "Catalyst",
                    "confidence": 0.95,
                    "reasoning": f"Contains {metal} - typical catalyst metal",
                    "role_changed": current_role != "Catalyst"
                }
    
    # Rule 2: Common bases → Reagent
    bases = {
        "cs2co3": "cesium carbonate",
        "k2co3": "potassium carbonate",
        "na2co3": "sodium carbonate",
        "naoh": "sodium hydroxide",
        "koh": "potassium hydroxide",
        "lioh": "lithium hydroxide",
        "naotbu": "sodium tert-butoxide",
        "kotbu": "potassium tert-butoxide",
        "dbu": "1,8-diazabicyclo[5.4.0]undec-7-ene",
        "diea": "diisopropylethylamine",
        "dipea": "diisopropylethylamine",
        "et3n": "triethylamine",
        "tea": "triethylamine",
    }
    
    for base_formula, base_name in bases.items():
        if base_formula in name_lower.replace(" ", "").replace("-", ""):
            if current_role == "Catalyst":  # Bases often misclassified as catalysts
                return {
                    "suggested_role": "Reagent",
                    "confidence": 0.90,
                    "reasoning": f"{base_name} is a base (reagent), not a catalyst",
                    "role_changed": True
                }
    
    # Rule 3: Common solvents
    solvents = ["dmf", "dmso", "thf", "dioxane", "toluene", "benzene", 
                "acetonitrile", "dcm", "dichloromethane", "chloroform",
                "meoh", "methanol", "etoh", "ethanol", "water", "h2o",
                "acetone", "ether", "diethyl ether"]
    
    for solv in solvents:
        if solv in name_lower.replace(" ", "").replace("-", ""):
            return {
                "suggested_role": "Solvent",
                "confidence": 0.98,
                "reasoning": f"Common solvent: {solv}",
                "role_changed": current_role != "Solvent"
            }
    
    # Rule 4: Ligands → Reagent
    ligands = ["pph3", "triphenylphosphine", "dppe", "dppp", "xantphos",
               "binap", "ligand", "phosphine"]
    
    for lig in ligands:
        if lig in name_lower:
            return {
                "suggested_role": "Reagent",
                "confidence": 0.85,
                "reasoning": f"Ligand compound - classified as reagent",
                "role_changed": current_role != "Reagent"
            }
    
    # Rule 5: SMILES-based detection (if available)
    if RDKIT_OK and smi:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Check for metal atoms
                metal_symbols = {"Pd", "Pt", "Cu", "Ni", "Rh", "Ru", "Ir", "Au", "Fe"}
                atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
                if atoms & metal_symbols:
                    return {
                        "suggested_role": "Catalyst",
                        "confidence": 0.92,
                        "reasoning": f"SMILES contains metal atom(s): {atoms & metal_symbols}",
                        "role_changed": current_role != "Catalyst"
                    }
        except Exception:
            pass
    
    # No rule matched - keep current role
    return {
        "suggested_role": current_role,
        "confidence": 0.5,
        "reasoning": "No specific rule matched - keeping original role",
        "role_changed": False
    }

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 18 – ROLE ACCURACY (Publication Improvement #3)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def compute_role_accuracy(extracted_entities: dict, ground_truth_entities: dict) -> dict:
    """
    PUBLICATION IMPROVEMENT #3: Compute role classification accuracy.
    
    Input format:
    {
      "reactants": [{"name": "...", "smiles": "..."}],
      "reagents": [...],
      "catalysts": [...],
      "solvents": [...],
      "products": [...]
    }
    
    Returns: {
        "role_accuracy": float,
        "correct_classifications": int,
        "total_classifications": int,
        "confusion_matrix": {...},
        "per_role_accuracy": {...}
    }
    
    Matching strategy: Match by canonical SMILES, then by name similarity.
    """
    def _extract_smiles_role_map(entity_dict):
        """Build map: canonical_smiles -> role"""
        smiles_to_role = {}
        name_to_role = {}
        
        for role in ["reactants", "reagents", "catalysts", "solvents", "products"]:
            for ent in entity_dict.get(role, []):
                if not isinstance(ent, dict):
                    continue
                
                smi = ent.get("smiles")
                name = (ent.get("name") or "").strip().lower()
                
                if smi and isinstance(smi, str) and smi.strip():
                    canonical = _rdkit_canonicalize(smi) if RDKIT_OK else smi.strip()
                    if canonical:
                        smiles_to_role[canonical] = role
                
                if name:
                    name_to_role[name] = role
        
        return smiles_to_role, name_to_role
    
    # Build maps
    gt_smi_map, gt_name_map = _extract_smiles_role_map(ground_truth_entities)
    ex_smi_map, ex_name_map = _extract_smiles_role_map(extracted_entities)
    
    # Match and count
    correct = 0
    total = 0
    confusion = {}
    per_role_stats = {
        "reactants": {"correct": 0, "total": 0},
        "reagents": {"correct": 0, "total": 0},
        "catalysts": {"correct": 0, "total": 0},
        "solvents": {"correct": 0, "total": 0},
        "products": {"correct": 0, "total": 0},
    }
    
    # Match by SMILES
    for smi, gt_role in gt_smi_map.items():
        if smi in ex_smi_map:
            total += 1
            ex_role = ex_smi_map[smi]
            
            # Confusion matrix
            key = f"{gt_role}→{ex_role}"
            confusion[key] = confusion.get(key, 0) + 1
            
            # Per-role stats
            per_role_stats[gt_role]["total"] += 1
            
            if ex_role == gt_role:
                correct += 1
                per_role_stats[gt_role]["correct"] += 1
    
    # Match by name (for entities without SMILES)
    for name, gt_role in gt_name_map.items():
        # Skip if already matched by SMILES
        gt_entities = ground_truth_entities.get(gt_role, [])
        has_smiles = any(e.get("smiles") for e in gt_entities 
                        if isinstance(e, dict) and (e.get("name") or "").strip().lower() == name)
        if has_smiles:
            continue
        
        if name in ex_name_map:
            total += 1
            ex_role = ex_name_map[name]
            
            key = f"{gt_role}→{ex_role}"
            confusion[key] = confusion.get(key, 0) + 1
            
            per_role_stats[gt_role]["total"] += 1
            
            if ex_role == gt_role:
                correct += 1
                per_role_stats[gt_role]["correct"] += 1
    
    # Compute per-role accuracy
    per_role_acc = {}
    for role, stats in per_role_stats.items():
        if stats["total"] > 0:
            per_role_acc[role] = round(stats["correct"] / stats["total"], 4)
        else:
            per_role_acc[role] = None
    
    role_accuracy = correct / max(total, 1)
    
    return {
        "role_accuracy": round(role_accuracy, 4),
        "correct_classifications": correct,
        "total_classifications": total,
        "confusion_matrix": confusion,
        "per_role_accuracy": per_role_acc,
        "per_role_stats": per_role_stats
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
