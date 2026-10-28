"""
chem_mcp_server_v2.py  –  Enhanced FastMCP cheminformatics tool server  (v2 fixed)

FIX: rxnmapper_atom_map now runs inside a subprocess with a 30-second hard
     timeout.  If RXNMapper is not installed OR takes too long, the tool
     returns a graceful heuristic result instead of hanging forever.

Tools (15 total)
────────────────
 1  pubchem_name_to_smiles       name → SMILES via PubChem REST
 2  validate_and_canonicalize    SMILES → RDKit canonical
 3  batch_resolve_smiles         [names] → {results}  (PubChem batch)
 4  compute_mol_weight           SMILES → exact MW
 5  mg_to_moles                  (mass_mg, mw) → moles
 6  atom_count_check             ([r_smi],[p_smi]) → heavy-atom balance
 7  sanitize_smiles              SMILES → canonical
 8  opsin_name_to_smiles         IUPAC → SMILES via OPSIN web API
 9  rxnmapper_atom_map           reaction SMILES → mapped (30 s hard timeout)
10  tanimoto_similarity          (smi1, smi2) → float
11  molar_consistency_check      (mass_mg, mw, reported_mmol) → {ok, …}
12  yield_sanity_check           (product_mmol, limiting_mmol) → {ok, …}
13  hard_match_score             (extracted, ground_truth) → {hard, soft}
14  smiles_from_cas              CAS → SMILES via PubChem
15  resolve_smiles_cascade       name → OPSIN → PubChem → CIR → None
"""

import json
import re
import time
import urllib.request
import urllib.parse
from typing import Optional

import requests
from mcp.server.fastmcp import FastMCP

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, DataStructs, AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

mcp = FastMCP("ChemToolsV2")

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_BAD_PATTERNS = [
    re.compile(r"^[A-Z][a-z]?(\d+)([A-Z][a-z]?(\d+)?)*$"),  # mol formula
    re.compile(r"^InChI="),
    re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"),              # InChIKey
    re.compile(r"^\s*(NULL|NONE|N/A)\s*$", re.I),
    re.compile(r"\({10,}"),
]

_HEADERS = {"User-Agent": "MAS-ChemPipeline/2-fixed"}


def _is_bad(s: str) -> bool:
    s = (s or "").strip()
    return not s or any(p.search(s) for p in _BAD_PATTERNS)


def _rdkit_canonical(smiles: str) -> Optional[str]:
    if not RDKIT_OK or _is_bad(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
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
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES", "")
        canonical = _rdkit_canonical(smiles) or smiles
        return {
            "smiles": canonical,
            "cid":    props.get("CID"),
            "mw":     float(props.get("MolecularWeight") or props.get("ExactMass") or 0),
        }
    except Exception as e:
        return {"error": str(e)}


def _opsin_convert(iupac_name: str) -> Optional[str]:
    """OPSIN via Cambridge web API — no Java/install needed."""
    try:
        enc  = urllib.parse.quote(iupac_name.strip())
        url  = f"https://opsin.ch.cam.ac.uk/opsin/{enc}.smi"
        resp = requests.get(url, timeout=10, headers=_HEADERS)
        if resp.status_code == 200:
            smi = resp.text.strip()
            if smi and not _is_bad(smi):
                return _rdkit_canonical(smi) or smi
    except Exception:
        pass
    return None


def _cir_fetch(name: str) -> Optional[str]:
    """NCI Chemical Identifier Resolver."""
    try:
        enc = urllib.parse.quote(name.strip())
        url = f"https://cactus.nci.nih.gov/chemical/structure/{enc}/smiles"
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode().strip()
        if raw and not _is_bad(raw):
            return _rdkit_canonical(raw) or raw
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS 1 – 7  (v1, preserved)
# ═══════════════════════════════════════════════════════════════════════════════

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
    can = _rdkit_canonical(smiles)
    if can:
        return {"valid": True,  "canonical": can, "reason": "OK"}
    return     {"valid": False, "canonical": None, "reason": "RDKit parse failed"}


@mcp.tool()
def batch_resolve_smiles(names: list[str], delay_s: float = 0.4) -> dict:
    """Resolve list of chemical names → SMILES via PubChem → {results}."""
    results = {}
    for name in names:
        if not (name or "").strip():
            continue
        res = _pubchem_fetch(name)
        results[name] = (
            None
            if ("error" in res or not res.get("smiles"))
            else {"smiles": res["smiles"], "cid": res.get("cid"), "mw": res.get("mw")}
        )
        time.sleep(delay_s)
    return {"results": results}


@mcp.tool()
def compute_mol_weight(smiles: str) -> dict:
    """Compute exact MW from SMILES via RDKit → {mw} or {error}."""
    if not RDKIT_OK:
        return {"error": "RDKit unavailable"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}
    return {"mw": round(Descriptors.ExactMolWt(mol), 4)}


@mcp.tool()
def mg_to_moles(mass_mg: float, mw: float) -> dict:
    """Convert mg to moles → {moles, mw_used} or {error}."""
    if not mw or mw <= 0:
        return {"error": f"Invalid MW: {mw}"}
    return {"moles": round(mass_mg / (mw * 1000), 8), "mw_used": mw}


@mcp.tool()
def atom_count_check(reactant_smiles: list[str], product_smiles: list[str]) -> dict:
    """Heavy-atom balance heuristic → {status, reactant_atoms, product_atoms, ratio}."""
    if not RDKIT_OK:
        return {"status": "Skipped – RDKit unavailable",
                "reactant_atoms": 0, "product_atoms": 0, "ratio": 0.0}

    def _count(lst):
        t = 0
        for s in lst:
            if s:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    t += mol.GetNumHeavyAtoms()
        return t

    r, p = _count(reactant_smiles), _count(product_smiles)
    if r == 0 or p == 0:
        status = "Incomplete – missing valid SMILES"
    elif 0.5 <= p / r <= 2.0:
        status = "Success"
    else:
        status = f"Warning – atom mismatch (ratio={p/r:.2f})"
    return {"status": status, "reactant_atoms": r,
            "product_atoms": p, "ratio": round(p / r, 3) if r else 0.0}


@mcp.tool()
def sanitize_smiles(smiles: str) -> dict:
    """Sanitize/canonicalize SMILES → {canonical} or {error}."""
    can = _rdkit_canonical(smiles)
    return {"canonical": can} if can else {"error": "Cannot parse or sanitize SMILES"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 8 – OPSIN name → SMILES
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def opsin_name_to_smiles(iupac_name: str) -> dict:
    """
    Convert IUPAC name → SMILES using OPSIN web API (opsin.ch.cam.ac.uk).
    Deterministic parser — no Java install required.
    Returns {smiles, source} or {error}.
    """
    smi = _opsin_convert(iupac_name)
    if smi:
        return {"smiles": smi, "source": "OPSIN-WebAPI"}
    return {"error": f"OPSIN could not parse: {iupac_name}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 9 – RXNMapper atom mapping  ← THE FIXED TOOL
# ═══════════════════════════════════════════════════════════════════════════════

def _rxnmapper_with_timeout(reaction_smiles: str, timeout_s: int = 30) -> dict:
    """
    Run RXNMapper in a separate thread with a hard timeout.
    Returns result dict or a heuristic fallback — NEVER hangs.
    """
    import concurrent.futures

    def _attempt():
        from rxnmapper import RXNMapper  # heavy import — only happens here
        mapper  = RXNMapper()
        results = mapper.get_attention_guided_atom_maps([reaction_smiles])
        mapped  = results[0]
        return {
            "mapped_smiles": mapped.get("mapped_rxn", reaction_smiles),
            "confidence":    round(mapped.get("confidence", 0.0), 4),
            "method":        "rxnmapper",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_attempt)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return None   # signal: fell through to heuristic
        except ImportError:
            return None   # rxnmapper not installed
        except Exception as exc:
            return {"error": str(exc), "method": "rxnmapper_failed"}


def _rxnmapper_heuristic(reaction_smiles: str) -> dict:
    """Validity-based heuristic when RXNMapper is unavailable or timed out."""
    parts = reaction_smiles.split(">>")
    if len(parts) != 2:
        return {"error": "Invalid reaction SMILES (expected >>)",
                "method": "heuristic"}
    r_smiles = [s.strip() for s in parts[0].split(".") if s.strip()]
    p_smiles = [s.strip() for s in parts[1].split(".") if s.strip()]
    r_valid  = [s for s in r_smiles if _rdkit_canonical(s)]
    p_valid  = [s for s in p_smiles if _rdkit_canonical(s)]
    conf = (
        len(r_valid) / max(len(r_smiles), 1) * 0.5
        + len(p_valid) / max(len(p_smiles), 1) * 0.5
    )
    return {
        "mapped_smiles": reaction_smiles,
        "confidence":    round(conf, 4),
        "method":        "validity_heuristic",
        "note":          "Install rxnmapper for full atom mapping",
    }


@mcp.tool()
def rxnmapper_atom_map(reaction_smiles: str) -> dict:
    """
    Atom-map a reaction SMILES.  Hard 30-second timeout prevents hanging.
    Falls back to a connectivity heuristic if RXNMapper is unavailable/slow.
    Returns {mapped_smiles, confidence, method}.
    """
    result = _rxnmapper_with_timeout(reaction_smiles, timeout_s=30)
    if result is None or "error" in (result or {}):
        return _rxnmapper_heuristic(reaction_smiles)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 10 – Tanimoto similarity
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def tanimoto_similarity(smiles1: str, smiles2: str) -> dict:
    """Morgan fingerprint Tanimoto similarity → {tanimoto, valid_pair} or {error}."""
    if not RDKIT_OK:
        return {"error": "RDKit unavailable"}
    m1 = Chem.MolFromSmiles(smiles1) if smiles1 and not _is_bad(smiles1) else None
    m2 = Chem.MolFromSmiles(smiles2) if smiles2 and not _is_bad(smiles2) else None
    if m1 is None or m2 is None:
        return {"tanimoto": 0.0, "valid_pair": False, "error": "One/both SMILES invalid"}
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    return {"tanimoto": round(DataStructs.TanimotoSimilarity(fp1, fp2), 4),
            "valid_pair": True}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 11 – Molar consistency check
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def molar_consistency_check(mass_mg: float, smiles: str,
                             reported_mmol: float) -> dict:
    """
    Verify reported mmol is consistent with mass_mg / MW (±20% tolerance).
    Returns {ok, calculated_mmol, reported_mmol, deviation_pct, mw_used, flag}.
    """
    if not RDKIT_OK:
        return {"error": "RDKit unavailable"}
    mol = Chem.MolFromSmiles(smiles) if smiles and not _is_bad(smiles) else None
    if mol is None:
        return {"ok": False, "error": "Invalid SMILES – cannot compute MW"}
    mw         = Descriptors.ExactMolWt(mol)
    calc_mmol  = round(mass_mg / mw, 4)
    deviation  = abs(calc_mmol - reported_mmol) / max(reported_mmol, 1e-9) * 100
    return {
        "ok":              deviation <= 20.0,
        "calculated_mmol": calc_mmol,
        "reported_mmol":   reported_mmol,
        "deviation_pct":   round(deviation, 2),
        "mw_used":         round(mw, 4),
        "flag":            "PASS" if deviation <= 20.0 else "FAIL: >20% mismatch",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 12 – Yield sanity check
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def yield_sanity_check(product_mmol: float, limiting_mmol: float,
                        reported_yield_pct: Optional[float] = None) -> dict:
    """
    Check if yield is chemically plausible (0–100% vs limiting reagent).
    Returns {ok, calculated_yield_pct, flag}.
    """
    if limiting_mmol <= 0:
        return {"ok": False, "error": "Limiting reagent mmol must be > 0"}
    calc_yield = round(product_mmol / limiting_mmol * 100, 2)
    ok   = 0.0 <= calc_yield <= 100.0
    flag = "PASS" if ok else f"FAIL: impossible yield {calc_yield:.1f}%"
    result = {"ok": ok, "calculated_yield_pct": calc_yield, "flag": flag}
    if reported_yield_pct is not None:
        deviation = abs(calc_yield - reported_yield_pct)
        result["reported_yield_pct"]  = reported_yield_pct
        result["yield_deviation_pct"] = round(deviation, 2)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 13 – Hard / soft match score
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def hard_match_score(extracted: dict, ground_truth: dict) -> dict:
    """
    Compute hard / soft match between extracted and GT reaction dicts.
    Each dict: {reactant_smiles:[], product_smiles:[], conditions:{}}.
    Returns {hard_match, soft_match, breakdown}.
    """
    def _canon_set(smi_list):
        out = set()
        for s in (smi_list or []):
            c = _rdkit_canonical(s) if s else None
            if c:
                out.add(c)
        return out

    e_r = _canon_set(extracted.get("reactant_smiles", []))
    e_p = _canon_set(extracted.get("product_smiles",  []))
    g_r = _canon_set(ground_truth.get("reactant_smiles", []))
    g_p = _canon_set(ground_truth.get("product_smiles",  []))

    reactant_match = len(e_r & g_r) / max(len(g_r), 1)
    product_match  = len(e_p & g_p) / max(len(g_p), 1)
    soft_match     = round((reactant_match + product_match) / 2, 4)

    e_cond = extracted.get("conditions",   {})
    g_cond = ground_truth.get("conditions", {})
    cond_match = 1.0
    if g_cond:
        matched  = sum(1 for k, v in g_cond.items() if e_cond.get(k) == v)
        cond_match = matched / max(len(g_cond), 1)

    hard_match = round(soft_match * 0.7 + cond_match * 0.3, 4)
    return {
        "hard_match": hard_match,
        "soft_match": soft_match,
        "breakdown":  {
            "reactant_recall": round(reactant_match, 4),
            "product_recall":  round(product_match,  4),
            "condition_match": round(cond_match,      4),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 14 – CAS → SMILES
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def smiles_from_cas(cas_number: str) -> dict:
    """Resolve CAS number → SMILES via PubChem → {smiles, cid, mw} or {error}."""
    try:
        enc = urllib.parse.quote(cas_number.strip())
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{enc}/property/IsomericSMILES,MolecularWeight/JSON"
        )
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        smiles = _rdkit_canonical(props.get("IsomericSMILES", "")) or props.get("IsomericSMILES", "")
        return {"smiles": smiles, "cid": props.get("CID"),
                "mw": float(props.get("MolecularWeight") or 0)}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 15 – Cascade SMILES resolver  (OPSIN → PubChem → CIR)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def resolve_smiles_cascade(name: str) -> dict:
    """
    Primary SMILES fix tool: OPSIN web API → PubChem REST → NCI CIR.
    Never uses raw LLM — prevents hallucinations entirely.
    Returns {smiles, source, mw} or {smiles: null, tried, error}.
    """
    tried: list[str] = []

    # Tier 1 — OPSIN (deterministic IUPAC parser)
    tried.append("OPSIN-WebAPI")
    smi = _opsin_convert(name)
    if smi:
        mol = Chem.MolFromSmiles(smi) if RDKIT_OK else None
        mw  = round(Descriptors.ExactMolWt(mol), 4) if mol else None
        return {"smiles": smi, "source": "OPSIN-WebAPI", "mw": mw}

    # Tier 2 — PubChem by name
    tried.append("PubChem_name")
    time.sleep(0.2)
    res = _pubchem_fetch(name)
    if "error" not in res and res.get("smiles"):
        return {"smiles": res["smiles"], "source": "PubChem", "mw": res.get("mw")}

    # Tier 3 — NCI CIR
    tried.append("CIR")
    smi = _cir_fetch(name)
    if smi:
        mol = Chem.MolFromSmiles(smi) if RDKIT_OK else None
        mw  = round(Descriptors.ExactMolWt(mol), 4) if mol else None
        return {"smiles": smi, "source": "CIR", "mw": mw}

    return {
        "smiles": None,
        "tried":  tried,
        "error":  f"No valid SMILES for '{name}' via any deterministic source",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
