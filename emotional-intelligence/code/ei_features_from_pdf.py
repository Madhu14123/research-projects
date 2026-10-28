"""
EI Candidate Features — Full Implementation from PDF Specification
==================================================================
116 features across 5 dimensions, computed from:
  - Facial AU timeseries (OpenFace 2.0, 17 AUs)
  - Stage-coded annotation files (6 problem-solving stages)
  - SAM self-report (10 checkpoints × valence + arousal)
  - Exam scores (criterion variable)

Validation pipeline (4 phases as specified in PDF):
  Phase 1 : Measurement model  — VIF (formative) / McDonald's ω (reflective)
  Phase 2 : Criterion validity — bootstrap BCa CIs + permutation FDR
  Phase 3 : Feature selection  — ElasticNet + bootstrap stability ≥ 75 %
  Phase 4 : Report             — pilot framing, caveats, output files

Usage:
    python ei_features_from_pdf.py
    python ei_features_from_pdf.py --data_dir "D:/A_emotions in exam/data"
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re, zipfile, argparse, warnings, logging, traceback
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import savgol_filter, welch, argrelmax
from scipy.stats import (pearsonr, spearmanr,
                          entropy as sp_entropy,
                          kurtosis, skew, linregress,
                          norm as sp_norm)
from scipy.spatial.distance import cosine

from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (same data layout as existing pipeline)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = r"D:\A_emotions in exam\data"
RESULTS_DIR      = "results_pdf_features"

STAGE_ZIP  = "stage coded data-20260321T120851Z-1-001.zip"
FACIAL_ZIP = "facial expressions- data-20260321T182549Z-1-001.zip"
EXAM_FILE  = "Exam Scores.xlsx"
SAM_FILE   = "SAM Responses every 3 minutes.xlsx"
STAGE_DIR  = "stage_coded_data"
FACIAL_DIR = "facial_expressions"

FPS = 25           # nominal frame rate
SAM_N = 10         # SAM checkpoints per session

# AU → emotion composite mapping (FACS-based, consistent with existing pipeline)
EMOTION_AU = {
    "Anger"    : ["AU04_r","AU05_r","AU07_r","AU23_r"],
    "Fear"     : ["AU01_r","AU04_r","AU05_r"],
    "Sadness"  : ["AU01_r","AU04_r","AU15_r"],
    "Disgust"  : ["AU04_r","AU07_r","AU09_r","AU10_r","AU17_r","AU20_r"],
    "Pride"    : ["AU06_r","AU07_r","AU12_r"],
    "Happiness": ["AU06_r","AU12_r"],
    "Stress"   : ["AU01_r","AU06_r","AU12_r","AU15_r"],
}
ALL_EMOTIONS   = list(EMOTION_AU.keys())   # length 7
POSITIVE_EMOS  = ["Happiness","Pride"]
NEGATIVE_EMOS  = ["Anger","Fear","Sadness","Disgust","Stress"]
ALL_AU_COLS    = [f"AU{str(i).zfill(2)}_r"
                  for i in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]]

STAGE_MAP = {
    "preparation":"Preparation","ideation":"Ideation",
    "production" :"Production", "prod"    :"Production",
    "evaluation" :"Evaluation", "refinement":"Refinement",
    "block"      :"Block",      "sam"     :"SAM",
}
PRODUCTIVE_STAGES  = {"Production","Evaluation","Refinement","Prod","prod"}
BLOCK_STAGES       = {"Block","Block ","block"}
ALL_STAGES_ORDERED = ["Preparation","Ideation","Production",
                       "Evaluation","Refinement","Block"]
STAGE_ORDINAL      = {"Preparation":1,"Ideation":2,"Production":3,
                       "Evaluation":4,"Refinement":5,"Block":0}
STAGE_COMPLEXITY   = {"Preparation":2,"Ideation":3,"Production":4,
                       "Evaluation":4,"Refinement":3,"Block":5}
STAGE_OPTIMAL_STRESS = {"Preparation":0.5,"Ideation":1.0,"Production":1.5,
                          "Evaluation":2.0,"Refinement":1.0,"Block":3.0}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_zip(zp: Path, od: Path):
    if od.exists() and any(od.iterdir()):
        return
    od.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, "r") as z:
        z.extractall(od)


def pid_from(fn: str) -> str | None:
    m = re.search(r"(P\d+)", fn, re.I)
    return m.group(1).upper() if m else None


def find_files(d: Path) -> dict:
    out = {}
    for f in Path(d).rglob("*"):
        if f.suffix.lower() in {".csv",".xlsx",".xls"}:
            p = pid_from(f.name)
            if p:
                out.setdefault(p, f)
    return out


def clean_stage(raw) -> str:
    if pd.isna(raw):
        return "Unknown"
    return STAGE_MAP.get(str(raw).strip().lower(), str(raw).strip())


def load_facial(path: Path) -> pd.DataFrame:
    df = (pd.read_csv(path) if path.suffix == ".csv"
          else pd.read_excel(path))
    df.columns = df.columns.str.strip()
    if "cleaned time" in df.columns:
        df = df.rename(columns={"cleaned time":"time_s"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp":"time_s"})
    r_cols = [c for c in df.columns if c.endswith("_r")]
    keep   = (["time_s"]
              + [c for c in ["confidence","success"] if c in df.columns]
              + r_cols)
    df = df[keep].copy()
    if "confidence" in df.columns:
        conf_thr = 0.80
        filtered = df[df["confidence"] >= conf_thr]
        # If too strict, relax to 50% to avoid empty DataFrames
        if len(filtered) < 100:
            filtered = df[df["confidence"] >= 0.50]
        df = filtered if len(filtered) >= 50 else df
    if "success" in df.columns:
        filtered2 = df[df["success"] == 1]
        df = filtered2 if len(filtered2) >= 50 else df
    return df.reset_index(drop=True)


def _parse_secs(v) -> float:
    try:
        parts = str(v).split(":")
        return sum(float(x) * 60 ** (len(parts)-i-1)
                   for i, x in enumerate(parts))
    except:
        return np.nan


def load_stage(path: Path) -> pd.DataFrame:
    df = (pd.read_excel(path) if path.suffix in {".xlsx",".xls"}
          else pd.read_csv(path))
    df.columns = df.columns.str.strip()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "start" in cl and "start" not in col_map: col_map["start"] = c
        if "end"   in cl and "end"   not in col_map: col_map["end"]   = c
        if ("code" in cl or "stage" in cl) and "stage" not in col_map:
            col_map["stage"] = c
    df = df.rename(columns={v: k for k, v in col_map.items()})
    for col in ["start","end"]:
        if col not in df.columns:
            continue
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
        elif df[col].dtype == object:
            df[col] = df[col].apply(_parse_secs)
    if "stage" not in df.columns:
        return pd.DataFrame()
    df["stage"] = df["stage"].apply(clean_stage)
    df = df[df["stage"] != "SAM"].copy()
    return df.dropna(subset=["start","end","stage"]).reset_index(drop=True)


def load_exam(path: Path) -> dict:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    scores = {}
    for _, row in df.iterrows():
        m = re.search(r"P(\d+)", str(row.iloc[0]).upper())
        if not m:
            continue
        try:
            denom = float(row.iloc[2]) if len(row) > 2 else 26.0
            scores[f"P{m.group(1)}"] = float(row.iloc[1]) / denom
        except:
            pass
    return scores


def load_sam(path: Path) -> dict:
    df = pd.read_excel(path, header=None)
    hrow = 0
    for i, row in df.iterrows():
        if any("participant" in str(v).lower() for v in row.values):
            hrow = i
            break
    df.columns = df.iloc[hrow]
    df = df.iloc[hrow+1:].reset_index(drop=True)
    out = {}
    for _, row in df.iterrows():
        m = re.search(r"P(\d+)", str(row.iloc[0]).upper())
        if not m:
            continue
        pid  = f"P{m.group(1)}"
        vals = row.iloc[1:].tolist()
        def _f(v):
            try:
                return float(v) if not pd.isna(v) else np.nan
            except:
                return np.nan
        out[pid] = {
            "valence": [_f(v) for v in vals[0::2]],
            "arousal": [_f(v) for v in vals[1::2]],
        }
    return out


def compute_emotions(df: pd.DataFrame) -> pd.DataFrame:
    for em, aus in EMOTION_AU.items():
        avail = [a for a in aus if a in df.columns]
        df[em] = df[avail].sum(axis=1) if avail else 0.0
    return df


def align_stages(facial: pd.DataFrame, stage: pd.DataFrame) -> pd.DataFrame:
    facial = facial.copy()
    facial["stage"] = "Unknown"
    for _, row in stage.iterrows():
        m = ((facial["time_s"] >= row["start"]) &
             (facial["time_s"] <  row["end"]))
        facial.loc[m, "stage"] = row["stage"]
    return facial


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def sg(series, window=75, poly=3):
    """Savitzky-Golay smoothing. Handles both 1-D and 2-D arrays (per-column)."""
    arr = np.array(series, dtype=float)
    if arr.ndim == 2:
        return np.column_stack([sg(arr[:, i], window, poly)
                                 for i in range(arr.shape[1])])
    n = len(arr)
    w = min(window, n // 2 * 2 + 1)
    if w % 2 == 0:
        w += 1
    w = max(5, w)
    if n < w:
        return arr
    return savgol_filter(arr, w, poly)


def ds1hz(arr, fps=FPS):
    """Downsample to 1 Hz by block-averaging."""
    step = int(fps)
    return np.array([arr[i:i+step].mean()
                     for i in range(0, len(arr) - step + 1, step)])


def fill_sam(lst) -> np.ndarray:
    arr = np.array(lst, dtype=float)
    if np.all(np.isnan(arr)):
        return np.full(len(arr), 3.0)
    arr[np.isnan(arr)] = np.nanmean(arr)
    return arr


def ar1(s) -> float:
    s = np.asarray(s, dtype=float)
    if len(s) < 3:
        return 0.0
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])


def mssd(s) -> float:
    return float(np.mean(np.diff(np.asarray(s, dtype=float))**2))


def sample_entropy(s, m=2, r_fac=0.2, max_n=300) -> float:
    """Sample entropy — vectorised, capped at max_n points for speed."""
    s = np.asarray(s, dtype=float)
    if len(s) > max_n:
        # subsample evenly
        idx = np.round(np.linspace(0, len(s)-1, max_n)).astype(int)
        s   = s[idx]
    r = r_fac * s.std() + 1e-9
    n = len(s)
    if n < 2*m+1:
        return 0.0
    # Vectorised count using broadcasting
    def _cnt(m_):
        # Build template matrix (n-m_ × m_)
        templates = np.array([s[i:i+m_] for i in range(n-m_)])
        # Chebyshev distance between all pairs
        diff = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=2)
        np.fill_diagonal(diff, np.inf)
        return int((diff < r).sum())
    cm  = _cnt(m)
    cm1 = _cnt(m+1)
    if cm == 0:
        return 0.0
    return float(-np.log(cm1/(cm+1e-9)+1e-9))


def icc_1_1(mat) -> float:
    """ICC(1,1) — rows=epochs, cols=emotion composites."""
    try:
        k, n  = mat.shape
        grand = mat.mean()
        ss_tot   = ((mat - grand)**2).sum()
        row_m    = mat.mean(axis=1)
        ss_bet   = n * ((row_m - grand)**2).sum()
        ss_wit   = ss_tot - ss_bet
        ms_b = ss_bet / (k - 1)
        ms_w = ss_wit / (k * (n - 1))
        return float((ms_b - ms_w) / (ms_b + (n-1)*ms_w + 1e-9))
    except:
        return 0.0


def stage_windows(fused: pd.DataFrame, stage_set: set) -> list:
    in_s  = fused["stage"].isin(stage_set)
    segs, start = [], None
    for i, v in enumerate(in_s):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append(fused.iloc[start:i])
            start = None
    if start is not None:
        segs.append(fused.iloc[start:])
    return segs


def window_mean(fused, col, t_end, win=180):
    m = (fused["time_s"] >= t_end - win) & (fused["time_s"] < t_end)
    return float(fused.loc[m, col].mean()) if m.sum() > 5 else np.nan


def window_mean_au(fused, t_end, win=180):
    m     = (fused["time_s"] >= t_end - win) & (fused["time_s"] < t_end)
    au_c  = [c for c in ALL_AU_COLS if c in fused.columns]
    if m.sum() < 5 or not au_c:
        return np.nan
    return float(fused.loc[m, au_c].values.mean())


# ─────────────────────────────────────────────────────────────────────────────
# D1 : EMOTION PERCEPTION  (22 features)
# ─────────────────────────────────────────────────────────────────────────────

def compute_d1(fused: pd.DataFrame) -> dict:
    f   = {}
    emo = fused[ALL_EMOTIONS].values          # (N,7)
    au  = fused[[c for c in ALL_AU_COLS if c in fused.columns]].values

    # --- Emodiversity ---
    dom    = np.argmax(emo, axis=1)
    counts = np.bincount(dom, minlength=7).astype(float)
    p      = counts / (counts.sum() + 1e-9)

    # Simpson diversity  1 − Σ pᵢ²
    f["c1_simpson_diversity"]   = float(1 - np.sum(p**2))

    # Gini coefficient of frame counts
    s = np.sort(counts); n = len(s)
    f["c1_gini_emodiversity"]   = float(
        (2*np.sum(np.arange(1,n+1)*s) - (n+1)*s.sum()) /
        (n*s.sum() + 1e-9))

    # Emotion richness: # composites with mean > 0.5
    f["c1_emotion_richness"]    = float(np.sum(emo.mean(axis=0) > 0.5))

    # --- Emotional granularity via 1−ICC (30-s epochs) ---
    epoch_len = 30*FPS
    n_ep      = len(fused) // epoch_len
    if n_ep >= 3:
        ep_mat = np.array([emo[i*epoch_len:(i+1)*epoch_len].mean(axis=0)
                           for i in range(n_ep)])
        f["c1_emotion_differentiation_ICC"] = float(1 - icc_1_1(ep_mat))
        neg_idx = [ALL_EMOTIONS.index(e) for e in NEGATIVE_EMOS
                   if e in ALL_EMOTIONS]
        f["c1_neg_emotion_differentiation"] = float(1 - icc_1_1(ep_mat[:,neg_idx]))
    else:
        f["c1_emotion_differentiation_ICC"] = 0.0
        f["c1_neg_emotion_differentiation"] = 0.0

    # --- Temporal dynamics ---
    # Emotion transition entropy (7×7 Markov)
    T = np.zeros((7,7));
    for t in range(len(dom)-1):
        T[dom[t], dom[t+1]] += 1
    Tp   = T / (T.sum(axis=1, keepdims=True) + 1e-9)
    flat = Tp.flatten(); flat = flat[flat>0]
    f["c1_emotion_transition_entropy"] = float(-np.sum(flat*np.log(flat+1e-9)))

    # Emotion inertia (AR1 at 1 Hz, mean over 7 composites)
    f["c1_emotion_inertia_composite"] = float(np.mean(
        [ar1(ds1hz(emo[:,i])) for i in range(7)]))

    # Sample entropy on mean AU signal
    if au.shape[1] > 0:
        f["c1_sample_entropy_AU"] = sample_entropy(
            ds1hz(au.mean(axis=1)), m=2, r_fac=0.2)
    else:
        f["c1_sample_entropy_AU"] = 0.0

    # Emotion instability (MSSD at 1 Hz)
    f["c1_emotion_instability_MSSD"] = float(np.mean(
        [mssd(ds1hz(emo[:,i])) for i in range(7)]))

    # --- Emotion co-occurrence & blending ---
    thr    = 1.0
    active = (emo >= thr).astype(int)
    pos_m  = np.array([0,0,0,0,1,1,0], dtype=float)   # Happiness=4, Pride=5
    neg_m  = 1 - pos_m
    pos_a  = (active * pos_m).sum(axis=1) > 0
    neg_a  = (active * neg_m[:7]).sum(axis=1) > 0
    f["c1_mixed_emotion_rate"]  = float((pos_a & neg_a).mean())
    patterns = set(tuple(r) for r in active)
    f["c1_emotion_blend_count"] = float(len(patterns) / 127.0)

    # --- Cross-stage perception ---
    sp = {}
    for st in ALL_STAGES_ORDERED:
        m = fused["stage"] == st
        if m.sum() > 0:
            sp[st] = emo[m.values].mean(axis=0)
    if len(sp) >= 2:
        profiles = list(sp.values())
        dists = [cosine(profiles[i], profiles[j]) + 1e-9
                 for i,j in combinations(range(len(profiles)), 2)]
        f["c1_cross_stage_profile_differentiation"] = float(np.mean(dists))
        ents = [float(sp_entropy(pr/(pr.sum()+1e-9) + 1e-9))
                for pr in profiles]
        f["c1_stage_emodiversity_variance"] = float(np.std(ents))
    else:
        f["c1_cross_stage_profile_differentiation"] = 0.0
        f["c1_stage_emodiversity_variance"]         = 0.0

    # --- Frequency-domain ---
    if au.shape[1] > 0 and len(fused) > 2*FPS:
        sp_ents, hf_rats = [], []
        for i in range(au.shape[1]):
            freq, psd = welch(au[:,i], fs=FPS, nperseg=min(256,len(fused)//2))
            pn        = psd / (psd.sum() + 1e-9)
            sp_ents.append(float(sp_entropy(pn + 1e-9)))
            hf_rats.append(float(psd[(freq>=1.0)&(freq<=5.0)].sum() /
                                  (psd.sum() + 1e-9)))
        f["c1_spectral_entropy"]      = float(np.mean(sp_ents))
        f["c1_high_freq_power_ratio"] = float(np.mean(hf_rats))
    else:
        f["c1_spectral_entropy"]      = 0.0
        f["c1_high_freq_power_ratio"] = 0.0

    # Emotion kurtosis (5-s windows)
    win5 = 5*FPS
    kvs  = [np.mean([kurtosis(emo[s:s+win5, i]) for i in range(7)])
            for s in range(0, len(fused)-win5, win5)]
    f["c1_emotion_kurtosis"]    = float(np.mean(kvs)) if kvs else 0.0
    f["c1_skewness_balance"]    = float(np.std([skew(emo[:,i]) for i in range(7)]))

    # --- AU co-activation network ---
    if au.shape[1] > 1:
        act_au = (au > 0.5).astype(float)
        f["c1_AU_activation_breadth"] = float(act_au.sum(axis=1).mean())

        n_au    = au.shape[1]
        cooc    = (act_au.T @ act_au) / (len(fused) + 1e-9)
        np.fill_diagonal(cooc, 0)
        n_edges = (cooc > 0.1).sum() / 2
        f["c1_AU_network_density"]    = float(n_edges / (n_au*(n_au-1)/2 + 1e-9))

        adj = (cooc > 0.1).astype(float)
        np.fill_diagonal(adj, 0)
        cc_v = []
        for i in range(n_au):
            nbrs = np.where(adj[i] > 0)[0]
            if len(nbrs) < 2: continue
            sub = adj[np.ix_(nbrs, nbrs)]
            cc_v.append(sub.sum() / (len(nbrs)*(len(nbrs)-1) + 1e-9))
        f["c1_AU_network_clustering"] = float(np.mean(cc_v)) if cc_v else 0.0

        # Expression distinctiveness (MI with stage)
        sc = fused["stage"].astype("category").cat.codes.values
        mi_v = []
        for i in range(au.shape[1]):
            bins  = np.digitize(au[:,i], [0.5, 2.0])
            joint = np.zeros((3, int(sc.max()+1)), dtype=float)
            for b, s in zip(bins, sc):
                if s >= 0: joint[b, s] += 1
            joint /= (joint.sum() + 1e-9)
            px, py = joint.sum(axis=1), joint.sum(axis=0)
            mi = sum(joint[r,c]*np.log(joint[r,c]/(px[r]*py[c]+1e-9)+1e-9)
                     for r in range(joint.shape[0])
                     for c in range(joint.shape[1])
                     if joint[r,c] > 0)
            mi_v.append(mi)
        f["c1_expression_distinctiveness"] = float(np.mean(mi_v))

        # Expression ambiguity H(AU|Stage)
        ch_v = []
        for i in range(au.shape[1]):
            bins = np.digitize(au[:,i], [0.5, 2.0])
            ch   = 0.0
            for st in ALL_STAGES_ORDERED:
                m2 = (fused["stage"] == st).values
                if m2.sum() < 5: continue
                bc = np.bincount(bins[m2], minlength=3).astype(float)
                bc /= (bc.sum() + 1e-9)
                ch += float(sp_entropy(bc+1e-9)) * (m2.sum()/len(fused))
            ch_v.append(ch)
        f["c1_expression_ambiguity"] = float(np.mean(ch_v))
    else:
        for k in ["c1_AU_activation_breadth","c1_AU_network_density",
                  "c1_AU_network_clustering","c1_expression_distinctiveness",
                  "c1_expression_ambiguity"]:
            f[k] = 0.0

    return f   # 22 features


# ─────────────────────────────────────────────────────────────────────────────
# D2 : EMOTION REGULATION  (22 features)
# ─────────────────────────────────────────────────────────────────────────────

def compute_d2(fused: pd.DataFrame,
               sam_valence: list, sam_arousal: list) -> dict:
    f      = {}
    stress = sg(fused["Stress"].values).astype(float)
    s1hz   = ds1hz(stress)
    emo    = fused[ALL_EMOTIONS].values
    pm, ps = stress.mean(), stress.std() + 1e-9

    # --- Recovery dynamics (affective chronometry) ---
    peaks = argrelmax(stress, order=FPS*2)[0]
    peaks = peaks[stress[peaks] > pm + ps]

    lats, comps, slopes = [], [], []
    for pk in peaks:
        pre_b  = stress[max(0,pk-FPS*10):pk].mean() if pk > FPS*10 else pm
        target = pre_b + 0.5*ps
        for j in range(pk, min(pk+FPS*60, len(stress))):
            if stress[j] <= target:
                lats.append((j-pk)/FPS)
                comps.append(float(np.clip(
                    (stress[pk]-stress[j])/(stress[pk]-pre_b+1e-9), 0, 2)))
                if j > pk:
                    sl, *_ = linregress(np.arange(j-pk+1), stress[pk:j+1])
                    slopes.append(-sl)
                break
        else:
            lats.append(60.0); comps.append(0.0)

    f["c2_stress_recovery_latency"] = float(np.mean(lats))    if lats   else 30.0
    f["c2_recovery_completeness"]   = float(np.mean(comps))   if comps  else 0.0
    f["c2_peak_to_baseline_slope"]  = float(np.mean(slopes))  if slopes else 0.0

    # --- Emotional inertia ---
    f["c2_stress_inertia"] = ar1(s1hz)
    neg_idx = [ALL_EMOTIONS.index(e) for e in NEGATIVE_EMOS if e in ALL_EMOTIONS]
    f["c2_negative_emotion_inertia"] = ar1(ds1hz(emo[:,neg_idx].mean(axis=1)))

    # --- Affective variability / instability ---
    f["c2_stress_instability_MSSD"] = mssd(s1hz)
    f["c2_emotion_variability_SD"]  = float(s1hz.std())

    # --- Regulation flexibility ---
    ent_st, stress_st = [], []
    for st in ALL_STAGES_ORDERED:
        m = fused["stage"] == st
        if m.sum() < 5: continue
        seg   = emo[m.values]
        act_m = (seg >= 1.0).astype(float).mean(axis=0)
        pv    = act_m / (act_m.sum() + 1e-9)
        ent_st.append(float(sp_entropy(pv + 1e-9)))
        stress_st.append(float(fused.loc[m.values, "Stress"].mean()))

    f["c2_emotion_repertoire_diversity"]       = float(np.mean(ent_st)) if ent_st else 0.0
    f["c2_cross_stage_regulation_variability"] = float(np.std(stress_st)) if len(stress_st)>1 else 0.0

    dom_s = np.argmax(emo, axis=1)
    f["c2_regulation_strategy_switching"] = float(
        np.sum(np.diff(dom_s) != 0) / (len(dom_s)/FPS + 1e-9))

    # --- Stress dampening / habituation ---
    if len(peaks) >= 2:
        sl2, *_ = linregress(np.arange(len(peaks)), stress[peaks])
        f["c2_stress_peak_attenuation"] = float(-sl2)
    else:
        f["c2_stress_peak_attenuation"] = 0.0

    n = len(s1hz)
    f["c2_stress_habituation_ratio"] = float(
        s1hz[n//2:].mean() / (s1hz[:n//2].mean() + 1e-9)) if n >= 4 else 1.0

    # --- AU-level suppression ---
    if "AU23_r" in fused.columns:
        au23  = fused["AU23_r"].values
        hi_m  = stress > (pm + ps)
        lo_m  = stress < pm
        f["c2_lip_press_index"] = float(
            au23[hi_m].mean() - au23[lo_m].mean()
            if hi_m.sum()>5 and lo_m.sum()>5 else 0.0)
    else:
        f["c2_lip_press_index"] = 0.0

    neg_au = [c for c in ["AU04_r","AU05_r","AU01_r","AU15_r","AU17_r"]
              if c in fused.columns]
    if neg_au and len(peaks) >= 2:
        nau   = fused[neg_au].values.sum(axis=1)
        ep    = len(peaks)//2
        er    = nau[peaks[:ep]].mean() / (stress[peaks[:ep]].mean() + 1e-9)
        lr    = nau[peaks[ep:]].mean() / (stress[peaks[ep:]].mean() + 1e-9)
        f["c2_negative_expression_dampening"] = float(lr/(er+1e-9))
    else:
        f["c2_negative_expression_dampening"] = 1.0

    # --- Temporal regulation patterns ---
    chal = {"Block","Evaluation"}
    pre_sv = []
    for _, grp in fused.groupby(
            (fused["stage"] != fused["stage"].shift()).cumsum()):
        if grp["stage"].iloc[0] in chal:
            t0  = grp["time_s"].iloc[0]
            pre = fused[(fused["time_s"] >= t0-30) &
                        (fused["time_s"] <  t0)]["Stress"]
            if len(pre) > 5:
                pre_sv.append(float(pre.mean()))
    f["c2_anticipatory_regulation"] = float(
        pm - np.mean(pre_sv) if pre_sv else 0.0)

    n3 = n // 3
    f["c2_early_vs_late_regulation_ratio"] = float(
        s1hz[n-n3:].mean() / (s1hz[:n3].mean() + 1e-9)) if n3 > 0 else 1.0

    # --- Cross-stage regulation ---
    bsl = [-(linregress(np.arange(len(sg(ep["Stress"].values))),
                        sg(ep["Stress"].values))[0])
           for ep in stage_windows(fused, BLOCK_STAGES)
           if len(ep) >= FPS*3]
    f["c2_block_stage_regulation_efficiency"] = float(np.mean(bsl)) if bsl else 0.0

    prod_m = fused["stage"].isin(PRODUCTIVE_STAGES)
    f["c2_production_stress_floor"] = float(
        fused.loc[prod_m,"Stress"].median() if prod_m.sum()>5 else pm)

    # --- Cognitive reappraisal proxies ---
    if "AU04_r" in fused.columns and "AU12_r" in fused.columns:
        hi_s = stress > (pm + ps)
        if hi_s.sum() > 0:
            au04v = fused["AU04_r"].values; au12v = fused["AU12_r"].values
            f["c2_reappraisal_proxy_coactivation"] = float(
                ((au04v[hi_s]>1.0)&(au12v[hi_s]>1.0)).mean())
        else:
            f["c2_reappraisal_proxy_coactivation"] = 0.0

        v_a    = fill_sam(sam_valence)
        chk_t  = np.linspace(fused["time_s"].min(), fused["time_s"].max(), len(v_a))
        disc   = []
        for i, t in enumerate(chk_t):
            wm = (fused["time_s"]>=t-90)&(fused["time_s"]<t)
            if wm.sum() < 5: continue
            fv = (fused.loc[wm,"AU12_r"].mean() -
                  fused.loc[wm,"AU04_r"].mean())
            disc.append(abs(v_a[i] - fv))
        if len(disc) >= 3:
            sl3, *_ = linregress(np.arange(len(disc)), disc)
            f["c2_SAM_facial_discrepancy_reduction"] = float(-sl3)
        else:
            f["c2_SAM_facial_discrepancy_reduction"] = 0.0
    else:
        f["c2_reappraisal_proxy_coactivation"]    = 0.0
        f["c2_SAM_facial_discrepancy_reduction"]  = 0.0

    # neg-to-pos transition speed
    d_sm = np.argmax(sg(emo,window=51), axis=1) if len(fused)>51 else np.argmax(emo,axis=1)
    neg_i = [ALL_EMOTIONS.index(e) for e in NEGATIVE_EMOS if e in ALL_EMOTIONS]
    pos_i = [ALL_EMOTIONS.index(e) for e in POSITIVE_EMOS if e in ALL_EMOTIONS]
    in_n  = np.isin(d_sm, neg_i)
    in_p  = np.isin(d_sm, pos_i)
    ttimes, st = [], None
    for i in range(len(in_n)):
        if in_n[i] and st is None:   st = i
        elif in_p[i] and st is not None:
            ttimes.append((i-st)/FPS); st = None
    f["c2_neg_to_pos_transition_speed"] = float(np.mean(ttimes)) if ttimes else 30.0

    # valence-arousal regulation coupling
    v_arr = fill_sam(sam_valence); a_arr = fill_sam(sam_arousal)
    if len(v_arr) >= 3:
        dv     = np.diff(v_arr)
        s_ds   = ds1hz(stress)
        ci     = np.linspace(0, len(s_ds)-1, len(v_arr), dtype=int)
        ds_st  = np.diff(s_ds[ci])
        if len(dv) == len(ds_st):
            r, _ = pearsonr(dv, -ds_st)
            f["c2_valence_arousal_regulation_coupling"] = float(r)
        else:
            f["c2_valence_arousal_regulation_coupling"] = 0.0
    else:
        f["c2_valence_arousal_regulation_coupling"] = 0.0

    return f   # 22 features


# ─────────────────────────────────────────────────────────────────────────────
# D3 : ADAPTIVE PERSISTENCE  (25 features)
# ─────────────────────────────────────────────────────────────────────────────

def compute_d3(fused: pd.DataFrame, stage_df: pd.DataFrame) -> dict:
    f      = {}
    stress = fused["Stress"].values.astype(float)
    happy  = fused["Happiness"].values.astype(float)
    anger  = fused["Anger"].values.astype(float) if "Anger" in fused.columns else np.zeros(len(fused))
    stages = fused["stage"].values

    # Build per-second stage sequence
    ss, st_times = [], []
    for _, row in stage_df.sort_values("start").iterrows():
        n_s = max(1, int(row["end"]-row["start"]))
        ss.extend([row["stage"]]*n_s)
        st_times.extend(np.linspace(row["start"],row["end"],n_s).tolist())

    blocks_df  = stage_df[stage_df["stage"].isin(BLOCK_STAGES)].reset_index(drop=True)
    sorted_stg = stage_df.sort_values("start").reset_index(drop=True)

    # --- Block-to-productive transition dynamics ---
    post_b, blk_dur, prod_post = [], [], []
    for i, row in sorted_stg.iterrows():
        if row["stage"] not in BLOCK_STAGES: continue
        blk_dur.append(row["end"]-row["start"])
        if i+1 < len(sorted_stg):
            nx = sorted_stg.iloc[i+1]["stage"]
            post_b.append(nx)
            if nx in PRODUCTIVE_STAGES:
                prod_post.append(sorted_stg.iloc[i+1]["end"] -
                                 sorted_stg.iloc[i+1]["start"])

    if post_b:
        pc = {}
        for st in post_b: pc[st] = pc.get(st,0)+1
        tot = sum(pc.values())
        pv  = np.array(list(pc.values()))/tot
        f["c3_block_exit_diversity"]     = float(sp_entropy(pv+1e-9))
        f["c3_block_to_production_rate"] = float(pc.get("Production",0)/(tot+1e-9))
    else:
        f["c3_block_exit_diversity"]     = 0.0
        f["c3_block_to_production_rate"] = 0.0

    if len(blk_dur) >= 2:
        sl, *_ = linregress(np.arange(len(blk_dur)), blk_dur)
        f["c3_recovery_acceleration"] = float(-sl)
    else:
        f["c3_recovery_acceleration"] = 0.0

    # --- Productive struggle ---
    pb_s, upb_s = [], []
    blk_rows = sorted_stg[sorted_stg["stage"].isin(BLOCK_STAGES)]
    for i, row in sorted_stg.iterrows():
        if row["stage"] not in BLOCK_STAGES: continue
        bm  = (fused["time_s"]>=row["start"])&(fused["time_s"]<row["end"])
        bsv = float(stress[bm.values].mean()) if bm.sum()>0 else 0.0
        if i+1<len(sorted_stg) and sorted_stg.iloc[i+1]["stage"] in PRODUCTIVE_STAGES:
            pb_s.append(bsv)
        else:
            upb_s.append(bsv)

    f["c3_productive_confusion_index"] = float(
        np.mean(pb_s)/(np.mean(upb_s)+1e-9)
        if pb_s and upb_s else 1.0)

    if blk_dur:
        q25 = np.percentile(blk_dur,25); smed = np.median(stress)
        res = 0; tot2 = 0
        for i,(dur,row) in enumerate(zip(blk_dur, blk_rows.itertuples())):
            bm = (fused["time_s"]>=row.start)&(fused["time_s"]<row.end)
            bs = float(stress[bm.values].mean()) if bm.sum()>0 else 0.0
            if dur<=q25 and bs>smed:
                tot2+=1
                ii = sorted_stg.index.get_loc(row.Index) if row.Index in sorted_stg.index else i
                if ii+1<len(sorted_stg) and sorted_stg.iloc[ii+1]["stage"] in PRODUCTIVE_STAGES:
                    res+=1
        f["c3_confusion_resolution_rate"] = float(res/(tot2+1e-9))
    else:
        f["c3_confusion_resolution_rate"] = 0.0

    # --- Advanced Markov features ---
    enc   = {s:i for i,s in enumerate(ALL_STAGES_ORDERED)}
    e_seq = [enc[s] for s in ss if s in enc]

    if len(e_seq) >= 10:
        T = np.zeros((6,6))
        for t in range(len(e_seq)-1): T[e_seq[t],e_seq[t+1]]+=1
        rs  = T.sum(axis=1,keepdims=True)
        Tp2 = np.divide(T,rs,where=rs>0)
        try:
            ev, evec = np.linalg.eig(Tp2.T)
            stat = np.abs(evec[:,np.argmin(np.abs(ev-1))])
            stat /= stat.sum()
            ps_i = [enc[s] for s in PRODUCTIVE_STAGES if s in enc]
            f["c3_productive_absorbing_probability"] = float(stat[ps_i].sum())
        except:
            f["c3_productive_absorbing_probability"] = 0.0

        bst = enc["Block"]
        if len(e_seq) >= 3:
            cbb=0; cbb_e=0
            for t in range(len(e_seq)-2):
                if e_seq[t]==bst and e_seq[t+1]==bst:
                    cbb+=1
                    if e_seq[t+2]!=bst: cbb_e+=1
            f["c3_second_order_block_escape"] = float(cbb_e/(cbb+1e-9))
        else:
            f["c3_second_order_block_escape"] = 0.0

        ords = np.array([STAGE_ORDINAL.get(ALL_STAGES_ORDERED[e],0) for e in e_seq])
        nb   = ords[ords>0]
        f["c3_forward_stage_momentum"] = float(np.diff(nb).mean()) if len(nb)>1 else 0.0
    else:
        f["c3_productive_absorbing_probability"] = 0.0
        f["c3_second_order_block_escape"]        = 0.0
        f["c3_forward_stage_momentum"]           = 0.0

    # --- Engagement sustainability ---
    streaks=[]; cur=0
    for st in stages:
        if st in PRODUCTIVE_STAGES: cur+=1
        elif (st in BLOCK_STAGES or st not in PRODUCTIVE_STAGES) and cur>0:
            streaks.append(cur/FPS); cur=0
    if cur>0: streaks.append(cur/FPS)

    f["c3_longest_productive_streak"]     = float(max(streaks)) if streaks else 0.0
    f["c3_productive_streak_variability"] = float(
        np.std(streaks)/(np.mean(streaks)+1e-9)) if len(streaks)>1 else 0.0

    tot_fr = len(fused); thr3 = tot_fr//3
    ep   = (fused.iloc[:thr3]["stage"].isin(PRODUCTIVE_STAGES)).sum()
    lp   = (fused.iloc[2*thr3:]["stage"].isin(PRODUCTIVE_STAGES)).sum()
    f["c3_late_task_engagement_ratio"] = float(lp/(ep+1e-9))

    # --- Frustration tolerance ---
    bsl2 = [-(linregress(np.arange(len(sg(ep2["Stress"].values))),
                         sg(ep2["Stress"].values))[0])
            for ep2 in stage_windows(fused, BLOCK_STAGES)
            if len(ep2) >= FPS*2]
    f["c3_block_negative_affect_damping"] = float(np.mean(bsl2)) if bsl2 else 0.0

    bm2 = fused["stage"].isin(BLOCK_STAGES); nbm2 = ~bm2
    f["c3_block_anger_suppression"] = float(
        anger[bm2.values].mean()/(anger[nbm2.values].mean()+1e-9)
        if bm2.sum()>5 and nbm2.sum()>5 else 1.0)

    if bm2.sum() > 5:
        f["c3_stress_inertia_during_blocks"] = ar1(ds1hz(stress[bm2.values]))
    else:
        f["c3_stress_inertia_during_blocks"] = 0.0

    # --- Academic buoyancy ---
    brat, cvar = [], []
    for _, row in blocks_df.iterrows():
        t0,t1   = row["start"],row["end"]
        pre_m   = (fused["time_s"]>=t0-60)&(fused["time_s"]<t0)
        post_m  = (fused["time_s"]>=t1)&(fused["time_s"]<t1+60)
        if pre_m.sum()>5 and post_m.sum()>5:
            brat.append(float(happy[post_m.values].mean()) /
                        (float(happy[pre_m.values].mean())+1e-9))
        ep_s = stress[(fused["time_s"]>=t0).values&(fused["time_s"]<t1).values]
        if len(ep_s)>5: cvar.append(float(ep_s.var()))

    f["c3_bounce_back_index"]        = float(np.mean(brat)) if brat else 1.0
    ov = stress.var()+1e-9
    f["c3_adversity_composure_index"] = float(1-np.mean(cvar)/ov) if cvar else 0.0

    # --- Grit & time-on-task ---
    tot_t  = stage_df["end"].max()-stage_df["start"].min()+1e-9
    prod_t = stage_df[stage_df["stage"].isin(PRODUCTIVE_STAGES)].apply(
        lambda r: r["end"]-r["start"], axis=1).sum()
    blk_t  = stage_df[stage_df["stage"].isin(BLOCK_STAGES)].apply(
        lambda r: r["end"]-r["start"], axis=1).sum()

    f["c3_total_productive_time_ratio"]  = float(prod_t/tot_t)
    last3 = fused["time_s"] >= fused["time_s"].max()-180
    f["c3_task_completion_persistence"]  = float(
        fused.loc[last3,"stage"].isin(PRODUCTIVE_STAGES).mean()
        if last3.sum()>0 else 0.0)
    f["c3_effective_time_efficiency"]    = float(prod_t/(tot_t-blk_t+1e-9))

    # --- Stage cycling quality ---
    sl2 = sorted_stg["stage"].tolist()
    pet = sum(1 for i in range(len(sl2)-2)
              if sl2[i]=="Production" and sl2[i+1]=="Evaluation"
              and sl2[i+2]=="Refinement")
    f["c3_productive_iteration_index"] = float(pet/(len(sl2)+1e-9))

    reg_pat = {"Production→Preparation","Evaluation→Ideation",
               "Refinement→Ideation","Refinement→Production",
               "Evaluation→Production"}
    reg_c = sum(1 for i in range(len(sl2)-1)
                if f"{sl2[i]}→{sl2[i+1]}" in reg_pat)
    f["c3_regressive_transition_ratio"] = float(reg_c/(len(sl2)+1e-9))

    bts = []
    for _, row in blocks_df.iterrows():
        bm3 = (fused["time_s"]>=row["start"])&(fused["time_s"]<row["end"])
        if bm3.sum()<FPS*3: continue
        traj = happy[bm3.values]-stress[bm3.values]
        sl4,*_ = linregress(np.arange(len(traj)),sg(traj))
        bts.append(sl4)
    f["c3_block_emotional_trajectory_slope"] = float(np.mean(bts)) if bts else 0.0

    ph = []
    smh = float(happy.mean())
    for _, row in blocks_df.iterrows():
        pm3 = (fused["time_s"]>=row["end"])&(fused["time_s"]<row["end"]+120)
        if pm3.sum()>5: ph.append(float(happy[pm3.values].mean()))
    f["c3_post_block_engagement_quality"] = float(
        np.mean(ph)/(smh+1e-9)) if ph else 1.0

    if len(prod_post)>=3 and len(blk_dur)>=3:
        nc = min(len(blk_dur),len(prod_post))
        r,_ = pearsonr(blk_dur[:nc], prod_post[:nc])
        f["c3_struggle_benefit_index"] = float(r)
    else:
        f["c3_struggle_benefit_index"] = 0.0

    return f   # 25 features


# ─────────────────────────────────────────────────────────────────────────────
# D4 : AFFECTIVE SELF-AWARENESS  (25 features)
# ─────────────────────────────────────────────────────────────────────────────

def compute_d4(fused: pd.DataFrame,
               sam_valence: list, sam_arousal: list) -> dict:
    f      = {}
    v_arr  = fill_sam(sam_valence)
    a_arr  = fill_sam(sam_arousal)
    nchk   = len(v_arr)
    tmin,tmax = fused["time_s"].min(), fused["time_s"].max()
    chk_t  = np.linspace(tmin, tmax, nchk)

    def wm(col, t): return window_mean(fused, col, t)

    # facial valence index per checkpoint
    neg_ec = [e for e in NEGATIVE_EMOS if e in fused.columns]
    fv_lst = []
    for t in chk_t:
        h  = wm("Happiness", t)
        ns = [v for v in [wm(e,t) for e in neg_ec] if not np.isnan(v)]
        fv_lst.append(h - np.mean(ns) if (not np.isnan(h) and ns) else np.nan)
    fv_arr = np.array(fv_lst, dtype=float)

    # --- SAM-AU concordance ---
    h_chk = np.array([wm("Happiness",t) for t in chk_t])
    vb    = ~(np.isnan(v_arr)|np.isnan(fv_arr))
    vbh   = vb & ~np.isnan(h_chk)
    f["c4_valence_happiness_concordance"] = float(
        pearsonr(v_arr[vbh], h_chk[vbh])[0]) if vbh.sum()>=3 else 0.0

    neg_mn = np.array([np.nanmean([wm(e,t) for e in neg_ec[:4]])
                       for t in chk_t])
    inv_v  = 6 - v_arr
    vbn    = ~(np.isnan(inv_v)|np.isnan(neg_mn))
    f["c4_valence_negemo_concordance"] = float(
        pearsonr(inv_v[vbn], neg_mn[vbn])[0]) if vbn.sum()>=3 else 0.0

    au_chk = np.array([window_mean_au(fused,t) for t in chk_t])
    vba    = ~(np.isnan(a_arr)|np.isnan(au_chk))
    f["c4_arousal_AU_concordance"] = float(
        pearsonr(a_arr[vba], au_chk[vba])[0]) if vba.sum()>=3 else 0.0

    s_chk = np.array([wm("Stress",t) for t in chk_t])
    vbs   = ~(np.isnan(v_arr)|np.isnan(s_chk))
    f["c4_valence_stress_inverse_concordance"] = float(
        pearsonr(v_arr[vbs], -s_chk[vbs])[0]) if vbs.sum()>=3 else 0.0

    # --- SAM trajectory ---
    if nchk >= 4:
        tidx   = np.arange(nchk, dtype=float)
        coeffs = np.polyfit(tidx, v_arr, 2)
        f["c4_valence_curvature"]      = float(coeffs[0])
        dv                             = np.diff(v_arr)
        f["c4_valence_turning_points"] = float(np.sum(np.diff(np.sign(dv))!=0))
        sl5,*_ = linregress(tidx, a_arr)
        f["c4_arousal_trend"]          = float(sl5)
        f["c4_arousal_stability"]      = float(-a_arr.std())
    else:
        f["c4_valence_curvature"]      = 0.0
        f["c4_valence_turning_points"] = 0.0
        f["c4_arousal_trend"]          = 0.0
        f["c4_arousal_stability"]      = 0.0

    # --- Interoceptive accuracy ---
    z_sam = (v_arr  - v_arr.mean())             / (v_arr.std()           + 1e-9)
    z_fv  = (fv_arr - np.nanmean(fv_arr))        / (np.nanstd(fv_arr)   + 1e-9)
    vbz   = ~(np.isnan(z_sam)|np.isnan(z_fv))
    f["c4_facial_valence_accuracy"] = float(
        pearsonr(z_sam[vbz], z_fv[vbz])[0]) if vbz.sum()>=3 else 0.0

    if vbz.sum() >= 4:
        ds, df2 = np.diff(z_sam), np.diff(z_fv)
        vd      = ~(np.isnan(ds)|np.isnan(df2))
        f["c4_change_detection_alignment"] = float(
            pearsonr(ds[vd], df2[vd])[0]) if vd.sum()>=3 else 0.0
    else:
        f["c4_change_detection_alignment"] = 0.0

    bci = [i for i,t in enumerate(chk_t)
           if fused[(fused["time_s"]>=t-90)&(fused["time_s"]<t)
                    ]["stage"].isin(BLOCK_STAGES).mean() > 0.3]
    if len(bci) >= 2:
        vbl = v_arr[bci]; nbl = neg_mn[bci]
        vb2 = ~(np.isnan(vbl)|np.isnan(nbl))
        f["c4_block_stage_awareness"] = float(
            pearsonr(-vbl[vb2], nbl[vb2])[0]) if vb2.sum()>=2 else 0.0
    else:
        f["c4_block_stage_awareness"] = 0.0

    f["c4_positive_reporting_bias"]    = float(np.nanmean(z_sam-z_fv)) if vbz.sum()>=3 else 0.0
    f["c4_absolute_calibration_error"] = float(
        -np.nanmean(np.abs(z_sam[vbz]-z_fv[vbz]))) if vbz.sum()>=3 else 0.0

    # --- Arousal-performance alignment ---
    f["c4_arousal_optimality"]          = float(-abs(a_arr.mean()-3.0))
    f["c4_arousal_midrange_proportion"] = float(((a_arr>=2)&(a_arr<=4)).mean())

    # --- Emotional self-efficacy ---
    dips = np.where(np.diff(v_arr)<0)[0]
    rmag = [max(0.0, v_arr[d+1]-v_arr[d]) for d in dips if d+1<nchk]
    f["c4_valence_recovery_magnitude"] = float(np.mean(rmag)) if rmag else 0.0

    if "Pride" in fused.columns:
        pm2 = fused["stage"].isin(PRODUCTIVE_STAGES)
        gmp = float(fused["Pride"].mean())+1e-9
        f["c4_pride_during_production"] = float(
            fused.loc[pm2,"Pride"].mean()/gmp if pm2.sum()>5 else 1.0)
    else:
        f["c4_pride_during_production"] = 1.0

    aps = []
    for st,et in [("Ideation","Happiness"),("Evaluation","Stress"),
                  ("Refinement","Happiness"),("Block","Anger")]:
        m3 = fused["stage"]==st
        if m3.sum()>5 and et in fused.columns:
            z = ((fused.loc[m3,et].mean()-fused[et].mean()) /
                 (fused[et].std()+1e-9))
            aps.append(float(z))
    f["c4_stage_emotion_appropriateness"] = float(np.mean(aps)) if aps else 0.0

    em2 = fused["stage"]=="Evaluation"
    if em2.sum()>5:
        et2    = fused.loc[em2,"time_s"].mean()
        nc_idx = np.argmin(np.abs(chk_t-et2))
        f["c4_evaluation_anxiety_alignment"] = float(
            fused.loc[em2,"Stress"].mean()*a_arr[nc_idx]/25.0
            if 0<=nc_idx<nchk else 0.0)
    else:
        f["c4_evaluation_anxiety_alignment"] = 0.0

    # --- Alexithymia-reversed ---
    fvr  = np.nanmax(fv_arr)-np.nanmin(fv_arr)+1e-9
    samr = v_arr.max()-v_arr.min()+1e-9
    f["c4_emotional_range_concordance"]    = float(1-abs(1-samr/fvr))

    vmed  = np.nanmedian(v_arr)
    lv_t  = chk_t[v_arr<vmed]
    if len(lv_t)>=3:
        lv_e = []
        for t in lv_t:
            wn = (fused["time_s"]>=t-90)&(fused["time_s"]<t)
            if wn.sum()>5:
                lv_e.append([fused.loc[wn,e].mean()
                              for e in NEGATIVE_EMOS[:4] if e in fused.columns])
        if len(lv_e)>=3:
            mat = np.array(lv_e)
            f["c4_negative_emotion_differentiation"] = float(1-icc_1_1(mat))
        else:
            f["c4_negative_emotion_differentiation"] = 0.0
    else:
        f["c4_negative_emotion_differentiation"] = 0.0

    f["c4_sam_response_range"]           = float(samr)
    f["c4_anchoring_resistance"]         = float(1-ar1(v_arr))
    rva = abs(pearsonr(v_arr,a_arr)[0]) if nchk>=3 else 0.0
    f["c4_valence_arousal_discrimination"] = float(1-rva)

    # Temporal coherence
    if vbz.sum()>=4:
        ds2 = np.diff(z_sam); df3 = np.diff(z_fv)
        vd2 = ~(np.isnan(ds2)|np.isnan(df3))
        if vd2.sum()>=3:
            xc = [pearsonr(ds2[vd2],df3[vd2])[0]]
            if len(ds2)>2:
                xc.append(pearsonr(ds2[vd2][:-1],df3[vd2][1:])[0])
                xc.append(pearsonr(ds2[vd2][1:], df3[vd2][:-1])[0])
            f["c4_temporal_coherence"] = float(np.nanmax(xc))
        else:
            f["c4_temporal_coherence"] = 0.0
    else:
        f["c4_temporal_coherence"] = 0.0

    return f   # 25 features


# ─────────────────────────────────────────────────────────────────────────────
# D5 : EMOTION-COGNITION ALIGNMENT  (22 features)
# ─────────────────────────────────────────────────────────────────────────────

def compute_d5(fused: pd.DataFrame, sam_arousal: list) -> dict:
    f      = {}
    a_arr  = fill_sam(sam_arousal)
    happy  = fused["Happiness"].values.astype(float)
    stress = fused["Stress"].values.astype(float)
    anger  = (fused["Anger"].values.astype(float)
              if "Anger" in fused.columns else np.zeros(len(fused)))
    emo    = fused[ALL_EMOTIONS].values

    au04 = fused["AU04_r"].values if "AU04_r" in fused.columns else np.zeros(len(fused))
    au07 = fused["AU07_r"].values if "AU07_r" in fused.columns else np.zeros(len(fused))
    au01 = fused["AU01_r"].values if "AU01_r" in fused.columns else np.zeros(len(fused))
    au02 = fused["AU02_r"].values if "AU02_r" in fused.columns else np.zeros(len(fused))
    au12 = fused["AU12_r"].values if "AU12_r" in fused.columns else np.zeros(len(fused))
    au14 = fused["AU14_r"].values if "AU14_r" in fused.columns else np.zeros(len(fused))

    id_m  = (fused["stage"]=="Ideation").values
    pr_m  = fused["stage"].isin(PRODUCTIVE_STAGES).values
    ev_m  = (fused["stage"]=="Evaluation").values
    bl_m  = fused["stage"].isin(BLOCK_STAGES).values
    ref_m = (fused["stage"]=="Refinement").values

    hm = happy.mean()+1e-9

    # --- Stage-appropriate emotion features ---
    f["c5_ideation_positive_affect"] = float(
        (happy[id_m].mean()-hm)/hm if id_m.sum()>5 else 0.0)
    f["c5_production_concentration"] = float(
        (au04[pr_m]+au07[pr_m]).mean() if pr_m.sum()>5 else 0.0)
    f["c5_evaluation_critical_engagement"] = float(
        (au04[ev_m]+au14[ev_m]).mean() if ev_m.sum()>5 else 0.0)

    ideal = {
        "Ideation"   : np.array([0.3,0.2,0.1,0.1,0.8,0.5,0.4]),
        "Production" : np.array([0.5,0.3,0.2,0.1,0.3,0.5,0.8]),
        "Evaluation" : np.array([0.5,0.4,0.2,0.3,0.4,0.4,1.0]),
        "Refinement" : np.array([0.2,0.2,0.1,0.1,0.7,0.8,0.4]),
        "Block"      : np.array([0.8,0.7,0.5,0.6,0.1,0.1,1.5]),
    }
    sims = []
    for st, iv in ideal.items():
        m2 = (fused["stage"]==st).values
        if m2.sum()<5: continue
        obs  = emo[m2].mean(axis=0)
        obsn = obs/(np.linalg.norm(obs)+1e-9)
        ivn  = iv/(np.linalg.norm(iv)+1e-9)
        sims.append(float(np.dot(obsn,ivn)))
    f["c5_stage_emotion_profile_fit"] = float(np.mean(sims)) if sims else 0.0

    # --- Emotion-stage coupling (η²) ---
    eta2 = []
    for i,em_n in enumerate(ALL_EMOTIONS):
        grps = [emo[(fused["stage"]==st).values,i]
                for st in ALL_STAGES_ORDERED
                if (fused["stage"]==st).sum()>5]
        if len(grps)<2: continue
        grand   = np.concatenate(grps).mean()
        ss_b    = sum(len(g)*(g.mean()-grand)**2 for g in grps)
        ss_tot2 = sum(((g-grand)**2).sum() for g in grps)
        eta2.append(ss_b/(ss_tot2+1e-9))
    f["c5_emotion_stage_differentiation"] = float(np.mean(eta2)) if eta2 else 0.0

    def sm(arr, stset):
        m2 = fused["stage"].isin(stset).values
        return float(arr[m2].mean()) if m2.sum()>5 else 0.0

    f["c5_cognitive_effort_deployment"] = float(
        sm(au04,{"Production","Evaluation"}) -
        sm(au04,{"Preparation","Refinement"}))

    # --- Interest & curiosity ---
    int_id  = (au01[id_m]+au02[id_m]+au07[id_m]).mean() if id_m.sum()>5 else 0.0
    int_pr  = (au01[pr_m]+au02[pr_m]+au07[pr_m]).mean() if pr_m.sum()>5 else 0.0
    f["c5_interest_during_ideation"] = float(int_id - int_pr)

    # --- Flow state ---
    if pr_m.sum() > 5*FPS:
        au7p  = au07[pr_m]; hp = happy[pr_m]; sp = stress[pr_m]
        def zs(a): return (a-a.mean())/(a.std()+1e-9)
        flow  = (zs(au7p)+zs(hp)-zs(sp)-zs(au7p.std()*np.ones_like(au7p)))/4.0
        f["c5_flow_proxy_production"] = float(flow.mean())
        w5    = 5*FPS
        fwins = [flow[i:i+w5].mean() for i in range(0,len(flow)-w5,w5)]
        f["c5_flow_stability"] = float(
            -(np.std(fwins)/(np.mean(np.abs(fwins))+1e-9))
            if len(fwins)>1 else 0.0)
    else:
        f["c5_flow_proxy_production"] = 0.0
        f["c5_flow_stability"]        = 0.0

    # --- Boekaerts dual-processing ---
    pmh,pms = happy.mean(), stress.mean()
    w5 = 5*FPS
    mc=0; cc=0; tw=0
    for i in range(0,len(fused)-w5,w5):
        sh=happy[i:i+w5].mean(); ss2=stress[i:i+w5].mean()
        sa=anger[i:i+w5].mean() if len(anger)>0 else 0.0
        tw+=1
        if sh>pmh and ss2<pms and au12[i:i+w5].mean()>0.5: mc+=1
        elif ss2>pms and sh<pmh and sa>pms*0.5:            cc+=1
    f["c5_mastery_mode_ratio"] = float(mc/(mc+cc+1e-9))

    mc2=0; cc2=0
    chal_m = fused["stage"].isin({"Production","Evaluation"}).values
    for i in range(0,len(fused)-w5,w5):
        if not chal_m[i:i+w5].mean()>0.5: continue
        sh=happy[i:i+w5].mean(); ss2=stress[i:i+w5].mean()
        if sh>pmh and ss2<pms:     mc2+=1
        elif ss2>pms and sh<pmh:   cc2+=1
    f["c5_mastery_during_challenge"] = float(mc2/(mc2+cc2+1e-9))

    # --- Confusion-resolution ---
    au4_75 = np.percentile(au04,75); au1_md = np.median(au01)
    cres=0; ctot=0
    for i in range(len(fused)-FPS*18):
        w3 = slice(i,i+FPS*3)
        if au04[w3].mean()>au4_75 and au01[w3].mean()>au1_md:
            ctot+=1
            fut = slice(i+FPS*3,i+FPS*18)
            if (au04[fut].mean()<au4_75*0.75 and
                    au07[fut].mean()>np.median(au07)):
                cres+=1
    f["c5_confusion_resolution_rate"]  = float(cres/(ctot+1e-9))
    f["c5_block_constructive_confusion"] = float(
        (au04[bl_m]+au01[bl_m]).mean()-anger[bl_m].mean()
        if bl_m.sum()>5 else 0.0)

    # --- Stage transition quality ---
    sc_ch  = fused["stage"]!=fused["stage"].shift()
    ch_idx = fused.index[sc_ch].tolist()
    tdists = []
    for ci in ch_idx[1:]:
        loc = fused.index.get_loc(ci)
        if loc<15 or loc>len(fused)-15: continue
        pre  = emo[max(0,loc-15):loc].mean(axis=0)
        post = emo[loc:min(len(fused),loc+15)].mean(axis=0)
        if pre.sum()>0 and post.sum()>0:
            tdists.append(float(np.linalg.norm(pre-post)))
    f["c5_transition_emotional_preparation"] = float(-np.mean(tdists)) if tdists else 0.0

    if pr_m.sum()>5 and ev_m.sum()>5:
        pride = (fused["Pride"].values if "Pride" in fused.columns
                 else (au12+au14)/2)
        pt = fused.loc[pr_m,"time_s"]; et2 = fused.loc[ev_m,"time_s"]
        if len(pt)>0 and len(et2)>0:
            pl = ((fused["time_s"]>=pt.max()-30) &
                  fused["stage"].isin(PRODUCTIVE_STAGES))
            ef = ((fused["time_s"]<=et2.min()+30) &
                  (fused["stage"]=="Evaluation"))
            f["c5_pride_after_production"] = float(
                (pride[ef.values].mean() if ef.sum()>0 else 0.0) -
                (pride[pl.values].mean() if pl.sum()>0 else 0.0))
        else:
            f["c5_pride_after_production"] = 0.0
    else:
        f["c5_pride_after_production"] = 0.0

    # --- Task-appropriate arousal ---
    od = []
    for st,opt in STAGE_OPTIMAL_STRESS.items():
        m2 = (fused["stage"]==st).values
        if m2.sum()<5: continue
        oz  = (opt-stress.mean())/(stress.std()+1e-9)
        obsz= (stress[m2].mean()-stress.mean())/(stress.std()+1e-9)
        od.append(abs(obsz-oz))
    f["c5_stress_appropriate_modulation"] = float(-np.mean(od)) if od else 0.0

    chk_t2    = np.linspace(fused["time_s"].min(),fused["time_s"].max(),len(a_arr))
    comp_chk  = []
    for t in chk_t2:
        wn = (fused["time_s"]>=t-90)&(fused["time_s"]<t)
        if wn.sum()>5:
            ds = fused.loc[wn,"stage"].mode()
            comp_chk.append(STAGE_COMPLEXITY.get(ds.iloc[0],3) if len(ds)>0 else np.nan)
        else:
            comp_chk.append(np.nan)
    ca  = np.array(comp_chk,dtype=float)
    vld = ~(np.isnan(ca)|np.isnan(a_arr))
    f["c5_arousal_SAM_task_match"] = float(
        spearmanr(a_arr[vld],ca[vld])[0]) if vld.sum()>=3 else 0.0

    # --- Cross-stage differentiation ---
    hi = happy[id_m].mean() if id_m.sum()>5 else 0.0
    hp2= happy[pr_m].mean() if pr_m.sum()>5 else 0.0
    ac_pr = (au04[pr_m]+au07[pr_m]).mean() if pr_m.sum()>5 else 0.0
    ac_id = (au04[id_m]+au07[id_m]).mean() if id_m.sum()>5 else 0.0
    f["c5_positive_creative_vs_focused_production"] = float(
        ((hi-hp2)+(ac_pr-ac_id))/2.0)

    neg_emo = fused[[e for e in NEGATIVE_EMOS if e in fused.columns]].values.mean(axis=1)
    f["c5_negative_affect_block_specificity"] = float(
        neg_emo[bl_m].mean()-neg_emo[~bl_m].mean()
        if bl_m.sum()>5 else 0.0)

    w10    = 10*FPS
    cpw    = [(emo[i:i+w10]>1.0).sum(axis=1).mean()
              for i in range(0,len(fused)-w10,w10)]
    f["c5_emotion_complexity_index"] = float(np.mean(cpw)) if cpw else 0.0

    # SAM-AU coherence
    neg_ec = [e for e in NEGATIVE_EMOS if e in fused.columns]
    fvs = [float(fused.loc[(fused["time_s"]>=t-90)&(fused["time_s"]<t),
                            "Happiness"].mean() -
                 fused.loc[(fused["time_s"]>=t-90)&(fused["time_s"]<t),
                            (["Sadness"] if "Sadness" in fused.columns
                             else neg_ec[:1])].values.mean())
           if ((fused["time_s"]>=t-90)&(fused["time_s"]<t)).sum()>5
           else np.nan for t in chk_t2]
    fvstr = [float(fused.loc[(fused["time_s"]>=t-90)&(fused["time_s"]<t),
                              ["Stress"]+
                              (["Fear"] if "Fear" in fused.columns else [])
                              ].values.mean())
             if ((fused["time_s"]>=t-90)&(fused["time_s"]<t)).sum()>5
             else np.nan for t in chk_t2]
    sv  = fill_sam(list(a_arr))
    fvsa= np.array(fvs,dtype=float); fvsta=np.array(fvstr,dtype=float)
    coh = []
    v1  = ~np.isnan(fvsa);  v2 = ~np.isnan(fvsta)
    if v1.sum()>=3: coh.append(pearsonr(sv[v1],fvsa[v1])[0])
    if v2.sum()>=3: coh.append(pearsonr(a_arr[v2],fvsta[v2])[0])
    f["c5_SAM_AU_coherence"] = float(np.mean(coh)) if coh else 0.0

    s75   = np.percentile(stress,75); a7m = np.median(au07); smd=np.median(stress)
    rsp   = []; in_ep=False; epst=0
    for i in range(len(fused)-FPS*3):
        if not in_ep:
            if (stress[i:i+FPS*3].mean()>s75 or
                    anger[i:i+FPS*3].mean()>s75):
                in_ep=True; epst=i
        else:
            if au07[i]>a7m and stress[i]<smd:
                rsp.append((i-epst)/FPS); in_ep=False
    f["c5_engagement_recovery_speed"] = float(
        -np.mean(rsp) if rsp else -60.0)

    return f   # 22 features


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def process_all(data_dir: Path) -> tuple:
    for zname, dname in [(STAGE_ZIP, STAGE_DIR), (FACIAL_ZIP, FACIAL_DIR)]:
        zp = data_dir / zname
        od = data_dir / dname
        if zp.exists():
            extract_zip(zp, od)

    stage_files  = find_files(data_dir / STAGE_DIR)
    facial_files = find_files(data_dir / FACIAL_DIR)
    exam_data    = load_exam(data_dir / EXAM_FILE)
    sam_data     = load_sam(data_dir / SAM_FILE)

    common = sorted(
        set(stage_files) & set(facial_files) & set(exam_data) & set(sam_data),
        key=lambda p: int(re.search(r"\d+", p).group()),
    )
    log.info("Participants with all four data sources: %d", len(common))

    all_feats = {}
    for pid in common:
        try:
            facial = load_facial(facial_files[pid])
            stage  = load_stage(stage_files[pid])
            if facial.empty or stage.empty:
                log.warning("[%s] empty data — skipped", pid)
                continue
            facial = compute_emotions(facial)
            fused  = align_stages(facial, stage)
            fused  = fused[np.isfinite(fused["time_s"].values)].copy()
            if len(fused) < 100:
                log.warning("[%s] too few rows (%d) — skipped", pid, len(fused))
                continue

            sam = sam_data[pid]
            d1  = compute_d1(fused)
            d2  = compute_d2(fused, sam["valence"], sam["arousal"])
            d3  = compute_d3(fused, stage)
            d4  = compute_d4(fused, sam["valence"], sam["arousal"])
            d5  = compute_d5(fused, sam["arousal"])

            all_feats[pid] = {**d1, **d2, **d3, **d4, **d5}
            log.info("[%s] ✓  %d features", pid, len(all_feats[pid]))

        except Exception as e:
            log.error("[%s] FAILED: %s", pid, e)
            traceback.print_exc()

    return all_feats, exam_data


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 : MEASUREMENT MODEL
# ─────────────────────────────────────────────────────────────────────────────

def compute_vif(X: pd.DataFrame) -> pd.Series:
    from numpy.linalg import inv
    Xs   = StandardScaler().fit_transform(X.fillna(0))
    corr = np.corrcoef(Xs.T)
    try:
        vif = np.diag(inv(corr))
    except np.linalg.LinAlgError:
        vif = np.full(X.shape[1], np.nan)
    return pd.Series(vif, index=X.columns)


def mcdonalds_omega(X: pd.DataFrame) -> float:
    try:
        from numpy.linalg import eigh
        Xs        = StandardScaler().fit_transform(X.dropna())
        cov       = np.cov(Xs.T)
        evals,evecs = eigh(cov)
        lam       = evecs[:,-1] * np.sqrt(evals[-1])
        omega     = lam.sum()**2 / (lam.sum()**2 + (np.diag(cov)-lam**2).sum())
        return float(np.clip(omega,-1,1))
    except:
        return np.nan


def phase1_measurement_model(fm: pd.DataFrame) -> pd.DataFrame:
    dim_pref = {"D1":"c1_","D2":"c2_","D3":"c3_","D4":"c4_","D5":"c5_"}
    rows = []
    for dim, pref in dim_pref.items():
        cols = [c for c in fm.columns if c.startswith(pref)]
        if not cols:
            continue
        sub = fm[cols].dropna()
        if sub.shape[1] < 2:
            rows.append({"Dimension":dim,"Type":"insufficient","Action":"need more features"})
            continue
        corr = sub.corr().values
        np.fill_diagonal(corr, np.nan)
        mir  = float(np.nanmean(corr))
        row  = {"Dimension":dim,"N_features":len(cols),"Mean_inter_r":round(mir,3)}
        if mir < 0.15:
            row["Type"]    = "FORMATIVE"
            row["Metric"]  = "VIF"
            vif            = compute_vif(sub)
            row["VIF_max"] = round(float(vif.max()),2)
            row["VIF_mean"]= round(float(vif.mean()),2)
            row["Pass"]    = bool(vif.max()<5.0)
            row["Action"]  = ("RETAIN — VIF acceptable"
                              if row["Pass"] else "DROP collinear (VIF>5)")
        else:
            row["Type"]    = "REFLECTIVE"
            row["Metric"]  = "McDonald's ω"
            row["Omega"]   = round(mcdonalds_omega(sub),3)
            row["Pass"]    = bool(row["Omega"]>0.6)
            row["Action"]  = ("RETAIN — ω acceptable"
                              if row["Pass"] else "REVIEW — ω<0.6")
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 : CRITERION VALIDITY
# ─────────────────────────────────────────────────────────────────────────────

def bca_ci(x, y, n_boot=2000, alpha=0.05):
    rng   = np.random.default_rng(42)
    n     = len(x)
    obs_r = pearsonr(x,y)[0]
    boot  = np.array([pearsonr(x[ix:=rng.integers(0,n,n)],y[ix])[0]
                      for _ in range(n_boot)])
    z0v   = np.clip(np.sum(boot<obs_r)/n_boot, 1e-6, 1-1e-6)
    z0    = float(np.log(z0v/(1-z0v)))
    jk    = np.array([pearsonr(np.delete(x,i),np.delete(y,i))[0]
                      for i in range(n)])
    jkm   = jk.mean()
    a     = float(np.sum((jkm-jk)**3) /
                  (6*(np.sum((jkm-jk)**2)**1.5)+1e-9))
    def ap(za):
        return sp_norm.cdf(z0+(z0+za)/(1-a*(z0+za)+1e-9))
    lo = float(np.percentile(boot, 100*ap(sp_norm.ppf(alpha/2))))
    hi = float(np.percentile(boot, 100*ap(sp_norm.ppf(1-alpha/2))))
    return obs_r, lo, hi


def perm_p(x, y, n_perm=5000):
    rng   = np.random.default_rng(0)
    obs_r = abs(pearsonr(x,y)[0])
    pr    = np.array([abs(pearsonr(x,rng.permutation(y))[0])
                      for _ in range(n_perm)])
    return float((pr>=obs_r).mean())


def bh_fdr(pv: list) -> np.ndarray:
    n   = len(pv)
    idx = np.argsort(pv)
    ps  = np.array(pv)[idx]
    adj = np.minimum(1.0, ps*n/(np.arange(1,n+1)))
    for i in range(n-2,-1,-1):
        adj[i] = min(adj[i],adj[i+1])
    res       = np.empty(n)
    res[idx]  = adj
    return res


def phase2_criterion_validity(fm: pd.DataFrame, exam: dict,
                               n_boot=2000, n_perm=5000) -> pd.DataFrame:
    pids = sorted(fm.index, key=lambda p: int(re.search(r"\d+",p).group()))
    y    = np.array([exam[p] for p in pids])
    rows = []
    for col in fm.columns:
        x = fm.loc[pids, col].fillna(0).values
        if x.std() < 1e-9:
            continue
        try:
            r, lo, hi  = bca_ci(x, y, n_boot)
            pp         = perm_p(x, y, n_perm)
            rho,_      = spearmanr(x, y)
            rows.append({"Feature":col,
                         "Pearson_r": round(r,4),
                         "BCa_lo":    round(lo,4),
                         "BCa_hi":    round(hi,4),
                         "Perm_p":    round(pp,4),
                         "Spearman_r":round(rho,4),
                         "CI_excl_0": bool(lo>0 or hi<0)})
        except Exception as e:
            log.warning("Validity failed for %s: %s", col, e)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["FDR_p"]       = bh_fdr(df["Perm_p"].tolist())
    df["Significant"] = (df["FDR_p"]<0.05) | df["CI_excl_0"]
    df["Strength"]    = df["Pearson_r"].abs().apply(
        lambda r: "large" if r>=0.50 else ("medium" if r>=0.30 else "small"))
    df["Dimension"]   = df["Feature"].str[:2].str.upper()
    return df.sort_values("Pearson_r", key=abs, ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 : FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_stability(X, y, names, n_boot=500, thr=0.75):
    rng   = np.random.default_rng(42)
    n, p  = X.shape
    sel   = np.zeros(p)
    for b in range(n_boot):
        idx  = rng.integers(0, n, n)
        Xb   = StandardScaler().fit_transform(X[idx])
        yb   = y[idx]
        try:
            m = ElasticNetCV(cv=min(5,n), l1_ratio=[0.5,0.7,0.9],
                              max_iter=2000, random_state=b%1000)
            m.fit(Xb, yb)
            sel += (np.abs(m.coef_)>1e-9).astype(float)
        except:
            pass
    stab = sel/n_boot
    df   = pd.DataFrame({"Feature":names,"Stability":stab})
    df   = df.sort_values("Stability", ascending=False)
    df["Selected"] = df["Stability"] >= thr
    return df


def phase3_feature_selection(fm: pd.DataFrame, exam: dict,
                              out_dir: Path) -> tuple:
    pids   = sorted(fm.index, key=lambda p: int(re.search(r"\d+",p).group()))
    y      = np.array([exam[p] for p in pids])
    X      = fm.loc[pids].fillna(0).values
    names  = list(fm.columns)

    log.info("Bootstrap stability selection (%d feats, n=%d) ...",
             len(names), len(pids))
    stab_df = bootstrap_stability(X, y, names, n_boot=500)
    stab_df.to_csv(out_dir/"stability_selection.csv", index=False)

    sel = stab_df[stab_df["Selected"]]["Feature"].tolist()
    if not sel:
        log.warning("No feature cleared 75%% threshold — using top 4.")
        sel = stab_df.head(4)["Feature"].tolist()
    sel = sel[:4]          # 10:1 rule (n=42)
    log.info("Stable features: %s", sel)

    Xs   = StandardScaler().fit_transform(
               X[:, [names.index(f) for f in sel]])
    preds = []
    for i in range(len(pids)):
        Xtr = np.delete(Xs,i,0); ytr = np.delete(y,i)
        m   = RidgeCV(alphas=[0.01,0.1,1,10])
        m.fit(Xtr,ytr)
        preds.append(float(m.predict(Xs[i:i+1])[0]))
    preds  = np.array(preds)
    r2     = r2_score(y, preds)
    log.info("LOOCV R² (stable features, ≤4): %.4f", r2)
    return stab_df, sel, r2, y, preds


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_features(cv_df: pd.DataFrame, out_dir: Path, top_n=20):
    top = cv_df.head(top_n).copy()
    top = top.iloc[::-1]                   # flip for horizontal bar
    fig, ax = plt.subplots(figsize=(10,8))
    colors = ["#e74c3c" if s else "#95a5a6" for s in top["Significant"]]
    bars   = ax.barh(top["Feature"], top["Pearson_r"].abs(), color=colors)
    for bar,r in zip(bars, top["Pearson_r"]):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f"{r:+.3f}", va="center", fontsize=7)
    ax.axvline(0.30, ls="--", color="#f39c12", lw=1, label="medium effect")
    ax.axvline(0.50, ls="--", color="#e74c3c", lw=1, label="large effect")
    ax.set_xlabel("|Pearson r| with exam score")
    ax.set_title(f"Top {top_n} candidate features — criterion validity\n"
                 "Red = significant (FDR<0.05 or BCa CI excludes 0)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir/"top_features_criterion_validity.png", dpi=150)
    plt.close(fig)


def plot_loocv(y_true, y_pred, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", s=60, zorder=3)
    lims = [min(y_true.min(),y_pred.min())-0.05,
            max(y_true.max(),y_pred.max())+0.05]
    ax.plot(lims, lims, "r--", lw=1.5, label="perfect prediction")
    r,_  = pearsonr(y_true, y_pred)
    ax.set_title(f"LOOCV: predicted vs actual exam score\nr = {r:.3f},  R² = {r2_score(y_true,y_pred):.3f}")
    ax.set_xlabel("Actual exam score (normalised)")
    ax.set_ylabel("LOOCV predicted")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir/"loocv_scatter.png", dpi=150)
    plt.close(fig)


def plot_stability(stab_df: pd.DataFrame, out_dir: Path):
    top = stab_df.head(30).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(9,8))
    colors  = ["#2ecc71" if s else "#bdc3c7" for s in top["Selected"]]
    ax.barh(top["Feature"], top["Stability"], color=colors)
    ax.axvline(0.75, ls="--", color="red", lw=1.5, label="75% threshold")
    ax.set_xlabel("Bootstrap selection stability")
    ax.set_title("Feature stability (500 bootstrap resamples)\nGreen = selected (≥75%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir/"stability_selection.png", dpi=150)
    plt.close(fig)


def plot_dimension_summary(cv_df: pd.DataFrame, out_dir: Path):
    dims = {"D1":"Emotion Perception","D2":"Emotion Regulation",
            "D3":"Adaptive Persistence","D4":"Affective Self-Awareness",
            "D5":"Emotion-Cognition Alignment"}
    fig, axes = plt.subplots(1,5,figsize=(18,5),sharey=False)
    for ax,(dk,dn) in zip(axes,dims.items()):
        sub = cv_df[cv_df["Dimension"]==dk].copy()
        if sub.empty:
            ax.set_title(f"{dk}\n{dn}\n(no data)"); continue
        sub = sub.sort_values("Pearson_r",key=abs,ascending=False).head(10).iloc[::-1]
        colors = ["#e74c3c" if s else "#bdc3c7" for s in sub["Significant"]]
        ax.barh(sub["Feature"].str.replace(f"c{dk[1]}_","",regex=False),
                sub["Pearson_r"].abs(), color=colors)
        ax.axvline(0.42, ls="--", color="grey", lw=0.8)
        ax.set_title(f"{dk} — {dn}", fontsize=8)
        ax.set_xlabel("|r|", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    plt.suptitle("Top features per dimension (red=significant)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir/"dimension_summary.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 : REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(fm, mm_df, cv_df, stab_df, sel, r2, out_dir):
    n_sig = int(cv_df["Significant"].sum()) if not cv_df.empty else 0

    lines = [
        "="*72,
        "  EI CANDIDATE FEATURES — PDF SPECIFICATION  (Full Implementation)",
        f"  n={fm.shape[0]} participants  ×  {fm.shape[1]} features",
        "  NOTE: n<150 — exploratory/pilot framing only",
        "="*72,"",
        "─"*72,
        "  PHASE 1 — MEASUREMENT MODEL CLASSIFICATION",
        "─"*72,
        f"  {'Dim':<4} {'Type':<12} {'Metric':<18} {'Score':>8}  Action",
    ]
    for _, r in mm_df.iterrows():
        score = (f"ω={r.get('Omega','—')}" if r.get("Type")=="REFLECTIVE"
                 else f"VIF_max={r.get('VIF_max','—')}")
        lines.append(f"  {r['Dimension']:<4} {r.get('Type','—'):<12} "
                     f"{r.get('Metric','—'):<18} {str(score):>10}  {r.get('Action','—')}")
    lines += ["",
        "─"*72,
        f"  PHASE 2 — CRITERION VALIDITY  (n_boot=2000, n_perm=5000, BH-FDR)",
        "─"*72,
        f"  Significant after FDR / CI excludes 0:  {n_sig} / {len(cv_df)} features",
        "",
        f"  {'Feature':<44} {'r':>6} {'BCa 95% CI':>18} {'FDR_p':>8} {'Size':>7}",
        "  "+"─"*68,
    ]
    for _, r in cv_df.head(25).iterrows():
        mk = " *" if r["Significant"] else ""
        lines.append(
            f"  {r['Feature']:<44} {r['Pearson_r']:>+6.3f} "
            f"[{r['BCa_lo']:>+6.3f},{r['BCa_hi']:>+6.3f}] "
            f"{r['FDR_p']:>8.4f} {r['Strength']:>7}{mk}")
    lines += ["",
        "─"*72,
        "  PHASE 3 — BOOTSTRAP STABILITY SELECTION  (500 resamples)",
        "─"*72,
        f"  Stable features selected (≥75%, cap=4):  {sel}",
        f"  LOOCV R² with ≤4 stable features:        {r2:.4f}",
        "",
        "─"*72,
        "  PILOT STUDY CAVEATS",
        "─"*72,
        "  • n=42 gives 80% power only for |r|≥0.42 — medium-large effects",
        "  • Winner's curse likely — treat all findings as exploratory",
        "  • Formative constructs (D1,D2,D5): VIF is correct, not Cronbach α",
        "  • Replicate with n≥150 before formal psychometric claims",
        "  • Pre-register for replication with a-priori power analysis",
        "",
        "─"*72,
        "  OUTPUT FILES",
        "─"*72,
        "  all_features_116.csv           feature matrix (42 × 116)",
        "  measurement_model.csv          Phase 1 results",
        "  criterion_validity.csv         Phase 2 results (BCa, perm, FDR)",
        "  stability_selection.csv        Phase 3 results",
        "  top_features_criterion_validity.png",
        "  loocv_scatter.png",
        "  stability_selection.png",
        "  dimension_summary.png",
        "="*72,
    ]
    txt = "\n".join(lines)
    print("\n"+txt)
    (out_dir/"pilot_report.txt").write_text(txt, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main(data_dir: str):
    dp      = Path(data_dir)
    out_dir = dp / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("="*60)
    log.info("  EI FEATURES FROM PDF  —  116 features × 42 participants")
    log.info("="*60)

    # ── Extract features ──────────────────────────────────────────────────
    all_feats, exam = process_all(dp)
    if len(all_feats) < 5:
        log.error("Too few participants (%d). Stopping.", len(all_feats))
        return

    fm = pd.DataFrame(all_feats).T
    fm.to_csv(out_dir/"all_features_116.csv")
    log.info("Feature matrix: %d × %d", *fm.shape)

    # ── Phase 1 ───────────────────────────────────────────────────────────
    log.info("\n── PHASE 1: Measurement Model ──────────────────────────────")
    mm_df = phase1_measurement_model(fm)
    mm_df.to_csv(out_dir/"measurement_model.csv", index=False)
    log.info("\n%s", mm_df.to_string(index=False))

    # ── Phase 2 ───────────────────────────────────────────────────────────
    log.info("\n── PHASE 2: Criterion Validity ─────────────────────────────")
    cv_df = phase2_criterion_validity(fm, exam, n_boot=2000, n_perm=5000)
    cv_df.to_csv(out_dir/"criterion_validity.csv", index=False)

    # ── Phase 3 ───────────────────────────────────────────────────────────
    log.info("\n── PHASE 3: Feature Selection ──────────────────────────────")
    stab_df, sel, r2, y_true, y_pred = phase3_feature_selection(fm, exam, out_dir)

    # ── Plots ─────────────────────────────────────────────────────────────
    if not cv_df.empty:
        plot_top_features(cv_df, out_dir)
        plot_dimension_summary(cv_df, out_dir)
    plot_loocv(y_true, y_pred, out_dir)
    plot_stability(stab_df, out_dir)

    # ── Phase 4 report ────────────────────────────────────────────────────
    write_report(fm, mm_df, cv_df, stab_df, sel, r2, out_dir)
    log.info("All outputs → %s", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()
    main(args.data_dir)
