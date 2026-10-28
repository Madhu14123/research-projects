"""
EI Candidate Feature Engineering — 116 Features + Revised Validation
=====================================================================
Implements the full candidate feature menu from the process-based behavioral
EI measurement framework across five dimensions:

  D1: Emotion Perception       — 22 features (granularity, diversity, complexity)
  D2: Emotion Regulation       — 22 features (recovery, inertia, flexibility)
  D3: Adaptive Persistence     — 25 features (buoyancy, grit, productive struggle)
  D4: Affective Self-Awareness — 25 features (SAM-AU concordance, calibration)
  D5: Emotion-Cognition Align  — 22 features (facilitation, flow, cognitive load)

Revised validation pipeline (replaces Cronbach's alpha for formative constructs):
  Phase 1: Measurement model classification — VIF for formative, McDonald's ω reflective
  Phase 2: Criterion validity — bootstrap BCa CIs, permutation p-values, BH-FDR
  Phase 3: Feature selection — ElasticNet + bootstrap stability selection (≥75%)
  Phase 4: Transparent reporting with pilot framing

References: Barrett (2004), Davidson (1998), Gross (1998/2015), Duckworth et al. (2007),
  Mauss et al. (2005), Boekaerts (2011), Csikszentmihalyi (1990), Mayer & Salovey (1997),
  Diamantopoulos & Winklhofer (2001), Meinshausen & Bühlmann (2010)

Results → results_candidate_features/

Usage:
    python ei_candidate_features.py
    python ei_candidate_features.py --data_dir "D:/A_emotions in exam/data"
"""

import re, zipfile, argparse, warnings, logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch, argrelmax
from scipy.stats import (pearsonr, spearmanr, entropy as scipy_entropy,
                          kurtosis, skew, linregress)
from scipy.spatial.distance import cosine
from statsmodels.stats.stattools import durbin_watson
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

# ============================================================================
# CONFIGURATION  (same paths as reference script)
# ============================================================================

DEFAULT_DATA_DIR = r"D:\A_emotions in exam\data"
RESULTS_DIR      = "results_candidate_features"

STAGE_ZIP  = "stage coded data-20260321T120851Z-1-001.zip"
FACIAL_ZIP = "facial expressions- data-20260321T182549Z-1-001.zip"
EXAM_FILE  = "Exam Scores.xlsx"
SAM_FILE   = "SAM Responses every 3 minutes.xlsx"
STAGE_DIR  = "stage_coded_data"
FACIAL_DIR = "facial_expressions"

FPS = 25

# AU → emotion composite mapping (identical to reference)
EMOTION_AU = {
    "Anger":     ["AU04_r","AU05_r","AU07_r","AU23_r"],
    "Fear":      ["AU01_r","AU04_r","AU05_r"],
    "Sadness":   ["AU01_r","AU04_r","AU15_r"],
    "Disgust":   ["AU04_r","AU07_r","AU09_r","AU10_r","AU17_r","AU20_r"],
    "Pride":     ["AU06_r","AU07_r","AU12_r"],
    "Happiness": ["AU06_r","AU12_r"],
    "Stress":    ["AU01_r","AU06_r","AU12_r","AU15_r"],
}
ALL_EMOTIONS    = list(EMOTION_AU.keys())
POSITIVE_EMOS   = ["Happiness", "Pride"]
NEGATIVE_EMOS   = ["Anger", "Fear", "Sadness", "Disgust", "Stress"]
ALL_AU_COLS     = [f"AU{str(i).zfill(2)}_r" for i in
                   [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]]

STAGE_MAP = {
    "preparation": "Preparation", "ideation": "Ideation",
    "production":  "Production",  "prod":     "Production",
    "evaluation":  "Evaluation",  "refinement":"Refinement",
    "block":       "Block",       "sam":       "SAM",
}
PRODUCTIVE_STAGES = {"Production", "Evaluation", "Refinement", "prod", "Prod"}
BLOCK_STAGES      = {"Block", "Block ", "block"}
ALL_STAGES_ORDERED = ["Preparation","Ideation","Production","Evaluation","Refinement","Block"]
STAGE_ORDINAL      = {"Preparation":1,"Ideation":2,"Production":3,
                       "Evaluation":4,"Refinement":5,"Block":0}
# Complexity assigned to stages for Yerkes-Dodson matching
STAGE_COMPLEXITY   = {"Preparation":2,"Ideation":3,"Production":4,
                       "Evaluation":4,"Refinement":3,"Block":5}
# Theoretically optimal stress per stage (0-5 scale)
STAGE_OPTIMAL_STRESS = {"Preparation":0.5,"Ideation":1.0,"Production":1.5,
                          "Evaluation":2.0,"Refinement":1.0,"Block":3.0}

# ============================================================================
# DATA LOADING  (identical helpers to reference script)
# ============================================================================

def extract_zip(zp, od):
    if od.exists() and any(od.iterdir()): return
    od.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, "r") as z:
        z.extractall(od)

def pid_from(fn):
    m = re.search(r"(P\d+)", fn, re.I)
    return m.group(1).upper() if m else None

def find_files(d):
    m = {}
    for f in Path(d).rglob("*"):
        if f.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            p = pid_from(f.name)
            if p: m.setdefault(p, f)
    return m

def clean_stage(raw):
    if pd.isna(raw): return "Unknown"
    return STAGE_MAP.get(str(raw).strip().lower(), str(raw).strip())

def load_facial(path):
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df.columns = df.columns.str.strip()
    if "cleaned time" in df.columns: df = df.rename(columns={"cleaned time": "time_s"})
    elif "timestamp"  in df.columns: df = df.rename(columns={"timestamp":    "time_s"})
    r_cols = [c for c in df.columns if c.endswith("_r")]
    df = df[["time_s"] + [c for c in ["confidence","success"] if c in df.columns] + r_cols].copy()
    if "confidence" in df.columns: df = df[df["confidence"] >= 0.80]
    if "success"    in df.columns: df = df[df["success"]    == 1]
    return df.reset_index(drop=True)

def load_stage(path):
    df = pd.read_excel(path) if path.suffix in {".xlsx",".xls"} else pd.read_csv(path)
    df.columns = df.columns.str.strip()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "start" in cl and "start" not in col_map: col_map["start"] = c
        if "end"   in cl and "end"   not in col_map: col_map["end"]   = c
        if ("code" in cl or "stage" in cl) and "stage" not in col_map: col_map["stage"] = c
    df = df.rename(columns={v: k for k, v in col_map.items()})
    for col in ["start","end"]:
        if col not in df.columns: continue
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
        elif df[col].dtype == object:
            def _s(v):
                try:
                    p = str(v).split(":")
                    return sum(float(x)*60**(len(p)-i-1) for i,x in enumerate(p))
                except: return np.nan
            df[col] = df[col].apply(_s)
    if "stage" not in df.columns: return pd.DataFrame()
    df["stage"] = df["stage"].apply(clean_stage)
    df = df[df["stage"] != "SAM"].copy()
    return df.dropna(subset=["start","end","stage"]).reset_index(drop=True)

def load_exam(path):
    df = pd.read_excel(path); df.columns = df.columns.str.strip()
    scores = {}
    for _, row in df.iterrows():
        m = re.search(r"P(\d+)", str(row.iloc[0]).upper())
        if not m: continue
        try: scores[f"P{m.group(1)}"] = float(row.iloc[1]) / (float(row.iloc[2]) if len(row) > 2 else 26.)
        except: pass
    return scores

def load_sam(path):
    df = pd.read_excel(path, header=None)
    hrow = 0
    for i, row in df.iterrows():
        if any("participant" in str(v).lower() for v in row.values):
            hrow = i; break
    df.columns = df.iloc[hrow]; df = df.iloc[hrow+1:].reset_index(drop=True)
    out = {}
    for _, row in df.iterrows():
        m = re.search(r"P(\d+)", str(row.iloc[0]).upper())
        if not m: continue
        pid = f"P{m.group(1)}"; vals = row.iloc[1:].tolist()
        def _f(v):
            try: return float(v) if not pd.isna(v) else np.nan
            except: return np.nan
        out[pid] = {"valence": [_f(v) for v in vals[0::2]],
                    "arousal":  [_f(v) for v in vals[1::2]]}
    return out

def compute_emotions(df):
    for em, aus in EMOTION_AU.items():
        avail = [a for a in aus if a in df.columns]
        df[em] = df[avail].sum(axis=1) if avail else 0.0
    return df

def align_stages(facial, stage):
    facial = facial.copy(); facial["stage"] = "Unknown"
    for _, row in stage.iterrows():
        mask = (facial["time_s"] >= row["start"]) & (facial["time_s"] < row["end"])
        facial.loc[mask, "stage"] = row["stage"]
    return facial

def sg(series, window=75, poly=3):
    n = len(series); w = min(window, n // 2 * 2 + 1)
    if w % 2 == 0: w += 1
    w = max(5, w)
    return savgol_filter(series, w, poly) if n >= w else np.array(series, dtype=float)

def downsample_1hz(arr, fps=FPS):
    """Downsample frame-level array to 1 Hz by averaging blocks of FPS frames."""
    n = len(arr); step = int(fps)
    return np.array([arr[i:i+step].mean() for i in range(0, n - step + 1, step)])

def _fill_sam(lst):
    arr = np.array(lst, dtype=float)
    if np.all(np.isnan(arr)): return np.full(len(arr), 3.0)
    arr[np.isnan(arr)] = np.nanmean(arr)
    return arr

def stage_windows(fused, stage_name_set):
    """Return list of sub-DataFrames, one per contiguous stage episode."""
    in_stage = fused["stage"].isin(stage_name_set)
    episodes = []
    start = None
    for i, v in enumerate(in_stage):
        if v and start is None: start = i
        elif not v and start is not None:
            episodes.append(fused.iloc[start:i])
            start = None
    if start is not None: episodes.append(fused.iloc[start:])
    return episodes

def icc(mat):
    """ICC(1,1) across rows (epochs × composites). Returns scalar."""
    try:
        k, n = mat.shape
        grand = mat.mean()
        ss_total = ((mat - grand)**2).sum()
        row_means = mat.mean(axis=1)
        ss_between = n * ((row_means - grand)**2).sum()
        ss_within = ss_total - ss_between
        ms_b = ss_between / (k - 1)
        ms_w = ss_within / (k * (n - 1))
        return float((ms_b - ms_w) / (ms_b + (n-1)*ms_w + 1e-9))
    except: return 0.0

def ar1(series):
    """Lag-1 autocorrelation of a 1D array."""
    if len(series) < 3: return 0.0
    return float(np.corrcoef(series[:-1], series[1:])[0,1])

def mssd(series):
    """Mean Squared Successive Differences."""
    d = np.diff(series)
    return float(np.mean(d**2))

def sample_entropy(series, m=2, r_factor=0.2):
    """Sample entropy (SampEn). Fast implementation."""
    series = np.array(series, dtype=float)
    r = r_factor * series.std() + 1e-9
    n = len(series)
    if n < 2 * m + 1: return 0.0
    def _count(m):
        count = 0
        for i in range(n - m):
            for j in range(i+1, n - m):
                if np.max(np.abs(series[i:i+m] - series[j:j+m])) < r:
                    count += 1
        return count
    c_m  = _count(m)
    c_m1 = _count(m+1)
    if c_m == 0: return 0.0
    return float(-np.log(c_m1 / (c_m + 1e-9) + 1e-9))

# ============================================================================
# D1: EMOTION PERCEPTION  — 22 features
# ============================================================================

def compute_d1(fused: pd.DataFrame) -> dict:
    feats = {}
    emo   = fused[ALL_EMOTIONS].values                  # (N, 7)
    au    = fused[[c for c in ALL_AU_COLS if c in fused.columns]].values

    # ── Emodiversity ─────────────────────────────────────────────────────────
    dom = np.argmax(emo, axis=1)
    counts = np.bincount(dom, minlength=7).astype(float)
    p = counts / (counts.sum() + 1e-9)

    # Shannon entropy (existing, kept for reference)
    feats["c1_shannon_entropy"] = float(scipy_entropy(p + 1e-9))

    # Simpson diversity index (1 - Σpᵢ²)
    feats["c1_simpson_diversity"] = float(1 - np.sum(p**2))

    # Gini coefficient of frame counts
    s = np.sort(counts)
    n = len(s)
    gini = float((2 * np.sum(np.arange(1, n+1) * s) - (n+1)*s.sum()) /
                 (n * s.sum() + 1e-9))
    feats["c1_gini_emodiversity"] = gini

    # Emotion richness: # composites with mean > 0.5
    feats["c1_emotion_richness"] = float(np.sum(emo.mean(axis=0) > 0.5))

    # ── Emotional granularity via 1 - ICC ─────────────────────────────────
    epoch_len = 30 * FPS  # 30-second epochs
    n_epochs  = len(fused) // epoch_len
    if n_epochs >= 3:
        epoch_mat = np.array([
            emo[i*epoch_len:(i+1)*epoch_len].mean(axis=0)
            for i in range(n_epochs)
        ])                                              # (n_epochs, 7)
        feats["c1_emotion_differentiation_ICC"] = float(1 - icc(epoch_mat))

        # Negative granularity (Anger, Fear, Sadness, Disgust, Stress = cols 0-4)
        neg_cols = [ALL_EMOTIONS.index(e) for e in NEGATIVE_EMOS if e in ALL_EMOTIONS]
        neg_mat  = epoch_mat[:, neg_cols]
        feats["c1_neg_emotion_differentiation"] = float(1 - icc(neg_mat))
    else:
        feats["c1_emotion_differentiation_ICC"] = 0.0
        feats["c1_neg_emotion_differentiation"] = 0.0

    # ── Temporal dynamics ──────────────────────────────────────────────────
    # Emotion transition entropy (7×7 Markov)
    trans = np.zeros((7, 7), dtype=float)
    for t in range(len(dom)-1):
        trans[dom[t], dom[t+1]] += 1
    trans_p = trans / (trans.sum(axis=1, keepdims=True) + 1e-9)
    flat = trans_p.flatten()
    flat = flat[flat > 0]
    feats["c1_emotion_transition_entropy"] = float(-np.sum(flat * np.log(flat + 1e-9)))

    # Emotion inertia (AR1 at 1 Hz, averaged across 7 composites)
    ar1_vals = []
    for i in range(7):
        ds = downsample_1hz(emo[:, i])
        ar1_vals.append(ar1(ds))
    feats["c1_emotion_inertia_composite"] = float(np.mean(ar1_vals))

    # Sample entropy on concatenated AU signal
    if au.shape[1] > 0:
        au_flat = au.mean(axis=1)
        feats["c1_sample_entropy_AU"] = sample_entropy(
            downsample_1hz(au_flat), m=2, r_factor=0.2)
    else:
        feats["c1_sample_entropy_AU"] = 0.0

    # MSSD (emotion instability, 1 Hz)
    mssd_vals = []
    for i in range(7):
        ds = downsample_1hz(emo[:, i])
        mssd_vals.append(mssd(ds))
    feats["c1_emotion_instability_MSSD"] = float(np.mean(mssd_vals))

    # ── Emotion co-occurrence and blending ────────────────────────────────
    threshold = 1.0
    active = (emo >= threshold).astype(int)
    valence_mask = np.array([0,0,0,0,1,1,0], dtype=float)  # Happiness=4, Pride=5
    pos_active = (active * valence_mask).sum(axis=1) > 0
    neg_mask   = 1 - valence_mask
    neg_active = (active * neg_mask[:7]).sum(axis=1) > 0
    feats["c1_mixed_emotion_rate"] = float((pos_active & neg_active).mean())

    # Emotion blend count (unique binary patterns)
    patterns = set(tuple(r) for r in active)
    feats["c1_emotion_blend_count"] = float(len(patterns) / 127.0)

    # ── Cross-stage perception ────────────────────────────────────────────
    stage_profiles = {}
    for st in ALL_STAGES_ORDERED:
        mask = fused["stage"] == st
        if mask.sum() > 0:
            stage_profiles[st] = emo[mask.values].mean(axis=0)
    if len(stage_profiles) >= 2:
        profiles = list(stage_profiles.values())
        dists = [cosine(profiles[i], profiles[j]) + 1e-9
                 for i, j in combinations(range(len(profiles)), 2)]
        feats["c1_cross_stage_profile_differentiation"] = float(np.mean(dists))

        entropies = []
        for prof in profiles:
            p_stage = prof / (prof.sum() + 1e-9)
            entropies.append(float(scipy_entropy(p_stage + 1e-9)))
        feats["c1_stage_emodiversity_variance"] = float(np.std(entropies))
    else:
        feats["c1_cross_stage_profile_differentiation"] = 0.0
        feats["c1_stage_emodiversity_variance"]         = 0.0

    # ── Frequency-domain features ──────────────────────────────────────────
    if au.shape[1] > 0 and len(fused) > 2 * FPS:
        spectral_entropies = []
        high_freq_ratios   = []
        for i in range(au.shape[1]):
            f, psd = welch(au[:, i], fs=FPS, nperseg=min(256, len(fused)//2))
            psd_norm = psd / (psd.sum() + 1e-9)
            spectral_entropies.append(float(scipy_entropy(psd_norm + 1e-9)))
            total_pow = psd.sum() + 1e-9
            high_pow  = psd[(f >= 1.0) & (f <= 5.0)].sum()
            high_freq_ratios.append(float(high_pow / total_pow))
        feats["c1_spectral_entropy"]      = float(np.mean(spectral_entropies))
        feats["c1_high_freq_power_ratio"] = float(np.mean(high_freq_ratios))
    else:
        feats["c1_spectral_entropy"]      = 0.0
        feats["c1_high_freq_power_ratio"] = 0.0

    # Emotion kurtosis (5-second windows)
    win = 5 * FPS
    kurt_vals = []
    for start in range(0, len(fused) - win, win):
        w = emo[start:start+win]
        kurt_vals.append(np.mean([kurtosis(w[:, i]) for i in range(7)]))
    feats["c1_emotion_kurtosis"] = float(np.mean(kurt_vals)) if kurt_vals else 0.0

    # Skewness balance
    skew_vals = [skew(emo[:, i]) for i in range(7)]
    feats["c1_skewness_balance"] = float(np.std(skew_vals))

    # ── AU co-activation network ──────────────────────────────────────────
    if au.shape[1] > 1:
        # AU activation breadth
        active_au = (au > 0.5).astype(float)
        feats["c1_AU_activation_breadth"] = float(active_au.sum(axis=1).mean())

        # AU network density
        n_au = au.shape[1]
        cooc = (active_au.T @ active_au) / (len(fused) + 1e-9)
        np.fill_diagonal(cooc, 0)
        max_edges = n_au * (n_au - 1) / 2
        n_edges = (cooc > 0.1).sum() / 2
        feats["c1_AU_network_density"] = float(n_edges / (max_edges + 1e-9))

        # AU network clustering coefficient (simplified)
        adj = (cooc > 0.1).astype(float)
        np.fill_diagonal(adj, 0)
        degrees = adj.sum(axis=1)
        cc_vals = []
        for i in range(n_au):
            nbrs = np.where(adj[i] > 0)[0]
            if len(nbrs) < 2: continue
            sub = adj[np.ix_(nbrs, nbrs)]
            cc_vals.append(sub.sum() / (len(nbrs)*(len(nbrs)-1) + 1e-9))
        feats["c1_AU_network_clustering"] = float(np.mean(cc_vals)) if cc_vals else 0.0

        # Expression distinctiveness (Mutual information with stage)
        stage_codes = fused["stage"].astype("category").cat.codes.values
        mi_vals = []
        for i in range(au.shape[1]):
            # Discretise AU into 3 bins
            bins = np.digitize(au[:, i], [0.5, 2.0])
            joint = np.zeros((3, len(np.unique(stage_codes))), dtype=float)
            for j, s in zip(bins, stage_codes):
                if s >= 0: joint[j, s] += 1
            joint /= (joint.sum() + 1e-9)
            px = joint.sum(axis=1)
            py = joint.sum(axis=0)
            mi = 0.0
            for r in range(joint.shape[0]):
                for c in range(joint.shape[1]):
                    if joint[r,c] > 0:
                        mi += joint[r,c] * np.log(joint[r,c] / (px[r]*py[c]+1e-9)+1e-9)
            mi_vals.append(mi)
        feats["c1_expression_distinctiveness"] = float(np.mean(mi_vals))

        # Expression ambiguity H(AU|Stage)
        cond_h_vals = []
        for i in range(au.shape[1]):
            bins = np.digitize(au[:, i], [0.5, 2.0])
            cond_h = 0.0
            for stage_name in ALL_STAGES_ORDERED:
                mask = (fused["stage"] == stage_name).values
                if mask.sum() < 5: continue
                stage_bins = bins[mask]
                bc = np.bincount(stage_bins, minlength=3).astype(float)
                bc /= (bc.sum() + 1e-9)
                cond_h += float(scipy_entropy(bc + 1e-9)) * (mask.sum() / len(fused))
            cond_h_vals.append(cond_h)
        feats["c1_expression_ambiguity"] = float(np.mean(cond_h_vals))
    else:
        for k in ["c1_AU_activation_breadth","c1_AU_network_density",
                  "c1_AU_network_clustering","c1_expression_distinctiveness",
                  "c1_expression_ambiguity"]:
            feats[k] = 0.0

    return feats


# ============================================================================
# D2: EMOTION REGULATION  — 22 features
# ============================================================================

def compute_d2(fused: pd.DataFrame, sam_valence: list, sam_arousal: list) -> dict:
    feats  = {}
    stress = sg(fused["Stress"].values).astype(float)
    stress_1hz = downsample_1hz(stress)
    emo    = fused[ALL_EMOTIONS].values
    p_mean = stress.mean()
    p_sd   = stress.std() + 1e-9

    # ── Recovery dynamics ────────────────────────────────────────────────
    peaks = argrelmax(stress, order=FPS*2)[0]
    peaks = peaks[stress[peaks] > p_mean + p_sd]

    latencies     = []
    completeness  = []
    recovery_slopes = []

    for pk in peaks:
        pre_base = stress[max(0, pk - FPS*10):pk].mean() if pk > FPS*10 else p_mean
        target   = pre_base + 0.5 * p_sd
        for j in range(pk, min(pk + FPS*60, len(stress))):
            if stress[j] <= target:
                lat = (j - pk) / FPS
                latencies.append(lat)
                post_val = stress[j]
                comp = (stress[pk] - post_val) / (stress[pk] - pre_base + 1e-9)
                completeness.append(float(np.clip(comp, 0, 2)))
                if j > pk:
                    slope, *_ = linregress(np.arange(j - pk + 1),
                                           stress[pk:j+1])
                    recovery_slopes.append(-slope)  # negate so positive = faster
                break
        else:
            latencies.append(60.0)
            completeness.append(0.0)

    feats["c2_stress_recovery_latency"] = float(np.mean(latencies))    if latencies    else 30.0
    feats["c2_recovery_completeness"]   = float(np.mean(completeness)) if completeness else 0.0
    feats["c2_peak_to_baseline_slope"]  = float(np.mean(recovery_slopes)) if recovery_slopes else 0.0

    # ── Emotional inertia ────────────────────────────────────────────────
    feats["c2_stress_inertia"] = ar1(stress_1hz)

    neg_idx = [ALL_EMOTIONS.index(e) for e in NEGATIVE_EMOS if e in ALL_EMOTIONS]
    neg_ds  = downsample_1hz(emo[:, neg_idx].mean(axis=1))
    feats["c2_negative_emotion_inertia"] = ar1(neg_ds)

    # ── Affective variability/instability ─────────────────────────────────
    feats["c2_stress_instability_MSSD"] = mssd(stress_1hz)
    feats["c2_emotion_variability_SD"]  = float(stress_1hz.std())

    # ── Regulation flexibility ────────────────────────────────────────────
    ent_per_stage = []
    stress_per_stage = []
    for st in ALL_STAGES_ORDERED:
        mask = fused["stage"] == st
        if mask.sum() < 5: continue
        seg_emo = emo[mask.values]
        # Binary active
        active  = (seg_emo >= 1.0).astype(float)
        mean_a  = active.mean(axis=0)
        p_s     = mean_a / (mean_a.sum() + 1e-9)
        ent_per_stage.append(float(scipy_entropy(p_s + 1e-9)))
        stress_per_stage.append(float(fused.loc[mask.values, "Stress"].mean()))

    feats["c2_emotion_repertoire_diversity"] = float(np.mean(ent_per_stage)) if ent_per_stage else 0.0
    feats["c2_cross_stage_regulation_variability"] = float(np.std(stress_per_stage)) if len(stress_per_stage) > 1 else 0.0

    # Regulation strategy switching
    dom = np.argmax(emo, axis=1)
    switches = np.sum(np.diff(dom) != 0)
    feats["c2_regulation_strategy_switching"] = float(switches / (len(dom) / FPS + 1e-9))

    # ── Stress dampening / habituation ───────────────────────────────────
    if len(peaks) >= 2:
        peak_intensities = stress[peaks]
        slope, *_ = linregress(np.arange(len(peaks)), peak_intensities)
        feats["c2_stress_peak_attenuation"] = float(-slope)   # negative slope = good
    else:
        feats["c2_stress_peak_attenuation"] = 0.0

    n = len(stress_1hz)
    if n >= 4:
        first_half = stress_1hz[:n//2].mean()
        second_half = stress_1hz[n//2:].mean()
        feats["c2_stress_habituation_ratio"] = float(second_half / (first_half + 1e-9))
    else:
        feats["c2_stress_habituation_ratio"] = 1.0

    # ── AU-level suppression ──────────────────────────────────────────────
    if "AU23_r" in fused.columns:
        au23    = fused["AU23_r"].values
        hi_mask = stress > (p_mean + p_sd)
        lo_mask = stress < p_mean
        feats["c2_lip_press_index"] = float(
            au23[hi_mask].mean() - au23[lo_mask].mean()
            if hi_mask.sum() > 5 and lo_mask.sum() > 5 else 0.0)
    else:
        feats["c2_lip_press_index"] = 0.0

    neg_au_cols = ["AU04_r","AU05_r","AU01_r","AU15_r","AU17_r"]
    avail_neg = [c for c in neg_au_cols if c in fused.columns]
    if avail_neg and len(peaks) >= 2:
        neg_au = fused[avail_neg].values.sum(axis=1)
        early_idx = peaks[:len(peaks)//2]
        late_idx  = peaks[len(peaks)//2:]
        early_ratio = neg_au[early_idx].mean() / (stress[early_idx].mean() + 1e-9)
        late_ratio  = neg_au[late_idx].mean()  / (stress[late_idx].mean() + 1e-9)
        feats["c2_negative_expression_dampening"] = float(late_ratio / (early_ratio + 1e-9))
    else:
        feats["c2_negative_expression_dampening"] = 1.0

    # ── Temporal regulation patterns ──────────────────────────────────────
    challenging = {"Block", "Evaluation"}
    pre_stress_vals = []
    for _, row in fused.groupby((fused["stage"] != fused["stage"].shift()).cumsum()):
        if row["stage"].iloc[0] in challenging:
            t_enter = row["time_s"].iloc[0]
            pre = fused[(fused["time_s"] >= t_enter - 30) &
                        (fused["time_s"] <  t_enter)]["Stress"]
            if len(pre) > 5: pre_stress_vals.append(float(pre.mean()))
    feats["c2_anticipatory_regulation"] = float(
        p_mean - np.mean(pre_stress_vals) if pre_stress_vals else 0.0)

    n3 = n // 3
    feats["c2_early_vs_late_regulation_ratio"] = float(
        stress_1hz[n - n3:].mean() / (stress_1hz[:n3].mean() + 1e-9)) if n3 > 0 else 1.0

    # ── Cross-stage regulation ────────────────────────────────────────────
    block_slopes = []
    for ep in stage_windows(fused, BLOCK_STAGES):
        if len(ep) < FPS * 3: continue
        s_ep = sg(ep["Stress"].values)
        slope, *_ = linregress(np.arange(len(s_ep)), s_ep)
        block_slopes.append(-slope)
    feats["c2_block_stage_regulation_efficiency"] = float(np.mean(block_slopes)) if block_slopes else 0.0

    prod_mask = fused["stage"].isin(PRODUCTIVE_STAGES)
    feats["c2_production_stress_floor"] = float(
        fused.loc[prod_mask, "Stress"].median() if prod_mask.sum() > 5 else p_mean)

    # ── Cognitive reappraisal proxies ──────────────────────────────────────
    if "AU04_r" in fused.columns and "AU12_r" in fused.columns:
        hi_stress = stress > (p_mean + p_sd)
        if hi_stress.sum() > 0:
            au04 = fused["AU04_r"].values
            au12 = fused["AU12_r"].values
            coact = ((au04[hi_stress] > 1.0) & (au12[hi_stress] > 1.0)).mean()
            feats["c2_reappraisal_proxy_coactivation"] = float(coact)
        else:
            feats["c2_reappraisal_proxy_coactivation"] = 0.0

        # SAM-facial discrepancy reduction
        v_arr   = _fill_sam(sam_valence)
        chk_times = np.linspace(fused["time_s"].min(), fused["time_s"].max(), len(v_arr))
        disc = []
        for i, t in enumerate(chk_times):
            win_mask = (fused["time_s"] >= t - 90) & (fused["time_s"] < t)
            if win_mask.sum() < 5: continue
            fv = fused.loc[win_mask, "AU12_r"].mean() - fused.loc[win_mask, "AU04_r"].mean()
            disc.append(abs(v_arr[i] - fv))
        if len(disc) >= 3:
            slope, *_ = linregress(np.arange(len(disc)), disc)
            feats["c2_SAM_facial_discrepancy_reduction"] = float(-slope)
        else:
            feats["c2_SAM_facial_discrepancy_reduction"] = 0.0
    else:
        feats["c2_reappraisal_proxy_coactivation"]    = 0.0
        feats["c2_SAM_facial_discrepancy_reduction"] = 0.0

    # neg-to-pos transition speed
    dom_smooth = np.argmax(sg(emo, window=51), axis=1) if len(fused) > 51 else np.argmax(emo, axis=1)
    neg_emo_idx = [ALL_EMOTIONS.index(e) for e in NEGATIVE_EMOS if e in ALL_EMOTIONS]
    pos_emo_idx = [ALL_EMOTIONS.index(e) for e in POSITIVE_EMOS if e in ALL_EMOTIONS]
    in_neg = np.isin(dom_smooth, neg_emo_idx)
    in_pos = np.isin(dom_smooth, pos_emo_idx)
    trans_times = []
    start_t = None
    for i in range(len(in_neg)):
        if in_neg[i] and start_t is None: start_t = i
        elif in_pos[i] and start_t is not None:
            trans_times.append((i - start_t) / FPS)
            start_t = None
    feats["c2_neg_to_pos_transition_speed"] = float(np.mean(trans_times)) if trans_times else 30.0

    # valence-arousal regulation coupling (SAM)
    v_arr = _fill_sam(sam_valence)
    a_arr = _fill_sam(sam_arousal)
    if len(v_arr) >= 3:
        dv = np.diff(v_arr); da = np.diff(a_arr)
        stress_ds = downsample_1hz(stress)
        # Align SAM to 1Hz stress (checkpoint at regular intervals)
        chk_idx = np.linspace(0, len(stress_ds)-1, len(v_arr), dtype=int)
        ds_stress = np.diff(stress_ds[chk_idx])
        if len(dv) == len(ds_stress):
            r, _ = pearsonr(dv, -ds_stress)
            feats["c2_valence_arousal_regulation_coupling"] = float(r)
        else:
            feats["c2_valence_arousal_regulation_coupling"] = 0.0
    else:
        feats["c2_valence_arousal_regulation_coupling"] = 0.0

    return feats


# ============================================================================
# D3: ADAPTIVE PERSISTENCE  — 25 features
# ============================================================================

def compute_d3(fused: pd.DataFrame, stage_df: pd.DataFrame) -> dict:
    feats  = {}
    stress = fused["Stress"].values.astype(float)
    happy  = fused["Happiness"].values.astype(float)
    anger  = fused["Anger"].values.astype(float)
    stages = fused["stage"].values

    # Build stage sequence (one label per second)
    stage_seq  = []
    stage_times = []
    for _, row in stage_df.sort_values("start").iterrows():
        t_start = row["start"]
        t_end   = row["end"]
        n_secs  = max(1, int(t_end - t_start))
        stage_seq.extend([row["stage"]] * n_secs)
        stage_times.extend(np.linspace(t_start, t_end, n_secs).tolist())

    blocks_df = stage_df[stage_df["stage"].isin(BLOCK_STAGES)].reset_index(drop=True)

    # ── Block-to-productive transition dynamics ───────────────────────────
    post_block_stages = []
    block_durations   = []
    productive_post_durations = []

    sorted_stage = stage_df.sort_values("start").reset_index(drop=True)
    for i, row in sorted_stage.iterrows():
        if row["stage"] not in BLOCK_STAGES: continue
        block_durations.append(row["end"] - row["start"])
        if i + 1 < len(sorted_stage):
            next_st = sorted_stage.iloc[i+1]["stage"]
            post_block_stages.append(next_st)
            if next_st in PRODUCTIVE_STAGES:
                prod_dur = sorted_stage.iloc[i+1]["end"] - sorted_stage.iloc[i+1]["start"]
                productive_post_durations.append(prod_dur)

    if post_block_stages:
        post_counts = {}
        for st in post_block_stages:
            post_counts[st] = post_counts.get(st, 0) + 1
        total_post = sum(post_counts.values())
        p_post = np.array(list(post_counts.values())) / total_post
        feats["c3_block_exit_diversity"] = float(scipy_entropy(p_post + 1e-9))
        feats["c3_block_to_production_rate"] = float(
            post_counts.get("Production", 0) / (total_post + 1e-9))
    else:
        feats["c3_block_exit_diversity"]     = 0.0
        feats["c3_block_to_production_rate"] = 0.0

    if len(block_durations) >= 2:
        slope, *_ = linregress(np.arange(len(block_durations)), block_durations)
        feats["c3_recovery_acceleration"] = float(-slope)
    else:
        feats["c3_recovery_acceleration"] = 0.0

    # ── Productive struggle ───────────────────────────────────────────────
    productive_block_stresses = []
    unproductive_block_stresses = []
    for i, row in sorted_stage.iterrows():
        if row["stage"] not in BLOCK_STAGES: continue
        block_stress_mask = (fused["time_s"] >= row["start"]) & (fused["time_s"] < row["end"])
        block_stress_val = float(stress[block_stress_mask].mean()) if block_stress_mask.sum() > 0 else 0.0
        if i + 1 < len(sorted_stage) and sorted_stage.iloc[i+1]["stage"] in PRODUCTIVE_STAGES:
            productive_block_stresses.append(block_stress_val)
        else:
            unproductive_block_stresses.append(block_stress_val)

    if productive_block_stresses and unproductive_block_stresses:
        feats["c3_productive_confusion_index"] = float(
            np.mean(productive_block_stresses) / (np.mean(unproductive_block_stresses) + 1e-9))
    else:
        feats["c3_productive_confusion_index"] = 1.0

    if len(block_durations) > 0:
        q25 = np.percentile(block_durations, 25)
        stress_med = np.median(stress)
        resolved = 0
        total_short_high = 0
        for i, (dur, row) in enumerate(zip(block_durations, sorted_stage[sorted_stage["stage"].isin(BLOCK_STAGES)].itertuples())):
            bm = (fused["time_s"] >= row.start) & (fused["time_s"] < row.end)
            bs = float(stress[bm.values].mean()) if bm.sum() > 0 else 0.0
            if dur <= q25 and bs > stress_med:
                total_short_high += 1
                if i + 1 < len(sorted_stage) and sorted_stage.iloc[i+1]["stage"] in PRODUCTIVE_STAGES:
                    resolved += 1
        feats["c3_confusion_resolution_rate"] = float(resolved / (total_short_high + 1e-9))
    else:
        feats["c3_confusion_resolution_rate"] = 0.0

    # ── Markov features ───────────────────────────────────────────────────
    # Encode stages for Markov (6 states)
    state_enc = {s: i for i, s in enumerate(ALL_STAGES_ORDERED)}
    enc_seq = [state_enc.get(s, -1) for s in stage_seq if s in state_enc]
    enc_seq = [e for e in enc_seq if e >= 0]

    if len(enc_seq) >= 10:
        T = np.zeros((6, 6), dtype=float)
        for t in range(len(enc_seq)-1):
            T[enc_seq[t], enc_seq[t+1]] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        T_prob = np.divide(T, row_sums, where=row_sums > 0)

        # Stationary distribution (productive absorbing probability)
        try:
            evals, evecs = np.linalg.eig(T_prob.T)
            stat = np.abs(evecs[:, np.argmin(np.abs(evals - 1))])
            stat = stat / stat.sum()
            prod_states = [state_enc[s] for s in PRODUCTIVE_STAGES if s in state_enc]
            feats["c3_productive_absorbing_probability"] = float(stat[prod_states].sum())
        except:
            feats["c3_productive_absorbing_probability"] = 0.0

        # Second-order block escape
        block_state = state_enc["Block"]
        if len(enc_seq) >= 3:
            count_bb = 0; count_bb_escape = 0
            for t in range(len(enc_seq)-2):
                if enc_seq[t] == block_state and enc_seq[t+1] == block_state:
                    count_bb += 1
                    if enc_seq[t+2] != block_state: count_bb_escape += 1
            feats["c3_second_order_block_escape"] = float(count_bb_escape / (count_bb + 1e-9))
        else:
            feats["c3_second_order_block_escape"] = 0.0

        # Forward stage momentum
        ordinals = np.array([STAGE_ORDINAL.get(ALL_STAGES_ORDERED[e], 0) for e in enc_seq])
        non_block = ordinals[ordinals > 0]
        if len(non_block) > 1:
            feats["c3_forward_stage_momentum"] = float(np.diff(non_block).mean())
        else:
            feats["c3_forward_stage_momentum"] = 0.0
    else:
        feats["c3_productive_absorbing_probability"] = 0.0
        feats["c3_second_order_block_escape"]        = 0.0
        feats["c3_forward_stage_momentum"]           = 0.0

    # ── Engagement sustainability ─────────────────────────────────────────
    prod_streaks = []
    current = 0
    for st in stages:
        if st in PRODUCTIVE_STAGES:
            current += 1
        elif st in BLOCK_STAGES and current > 0:
            prod_streaks.append(current / FPS)
            current = 0
        elif current > 0 and st not in PRODUCTIVE_STAGES:
            prod_streaks.append(current / FPS)
            current = 0
    if current > 0: prod_streaks.append(current / FPS)

    feats["c3_longest_productive_streak"]     = float(max(prod_streaks)) if prod_streaks else 0.0
    feats["c3_productive_streak_variability"] = float(
        np.std(prod_streaks) / (np.mean(prod_streaks) + 1e-9)) if len(prod_streaks) > 1 else 0.0

    total_frames = len(fused)
    third = total_frames // 3
    early_prod = (fused.iloc[:third]["stage"].isin(PRODUCTIVE_STAGES)).sum()
    late_prod  = (fused.iloc[2*third:]["stage"].isin(PRODUCTIVE_STAGES)).sum()
    feats["c3_late_task_engagement_ratio"] = float(late_prod / (early_prod + 1e-9))

    # ── Frustration tolerance ─────────────────────────────────────────────
    block_slopes = []
    for ep in stage_windows(fused, BLOCK_STAGES):
        if len(ep) < FPS * 2: continue
        s_ep = sg(ep["Stress"].values)
        slope, *_ = linregress(np.arange(len(s_ep)), s_ep)
        block_slopes.append(-slope)
    feats["c3_block_negative_affect_damping"] = float(np.mean(block_slopes)) if block_slopes else 0.0

    block_mask = fused["stage"].isin(BLOCK_STAGES)
    non_block_mask = ~block_mask
    feats["c3_block_anger_suppression"] = float(
        anger[block_mask.values].mean() /
        (anger[non_block_mask.values].mean() + 1e-9)
        if block_mask.sum() > 5 and non_block_mask.sum() > 5 else 1.0)

    if block_mask.sum() > 5:
        stress_block_1hz = downsample_1hz(stress[block_mask.values])
        feats["c3_stress_inertia_during_blocks"] = ar1(stress_block_1hz)
    else:
        feats["c3_stress_inertia_during_blocks"] = 0.0

    # ── Academic buoyancy ─────────────────────────────────────────────────
    bounce_ratios = []
    composure_vars = []
    for _, row in blocks_df.iterrows():
        t0, t1 = row["start"], row["end"]
        pre_mask  = (fused["time_s"] >= t0 - 60) & (fused["time_s"] <  t0)
        post_mask = (fused["time_s"] >= t1)        & (fused["time_s"] <  t1 + 60)
        if pre_mask.sum() > 5 and post_mask.sum() > 5:
            pre_h  = float(happy[pre_mask.values].mean())
            post_h = float(happy[post_mask.values].mean())
            bounce_ratios.append(post_h / (pre_h + 1e-9))
        block_stress_ep = stress[(fused["time_s"] >= t0).values & (fused["time_s"] < t1).values]
        if len(block_stress_ep) > 5:
            composure_vars.append(float(block_stress_ep.var()))

    feats["c3_bounce_back_index"] = float(np.mean(bounce_ratios)) if bounce_ratios else 1.0
    overall_var = stress.var() + 1e-9
    feats["c3_adversity_composure_index"] = float(
        1 - np.mean(composure_vars) / overall_var) if composure_vars else 0.0

    # ── Grit and time-on-task ─────────────────────────────────────────────
    total_time = (stage_df["end"].max() - stage_df["start"].min()) + 1e-9
    prod_time  = stage_df[stage_df["stage"].isin(PRODUCTIVE_STAGES)]["end"].sub(
        stage_df[stage_df["stage"].isin(PRODUCTIVE_STAGES)]["start"]).sum()
    block_time = stage_df[stage_df["stage"].isin(BLOCK_STAGES)]["end"].sub(
        stage_df[stage_df["stage"].isin(BLOCK_STAGES)]["start"]).sum()

    feats["c3_total_productive_time_ratio"] = float(prod_time / total_time)

    last3_mask = fused["time_s"] >= fused["time_s"].max() - 180
    feats["c3_task_completion_persistence"] = float(
        fused.loc[last3_mask, "stage"].isin(PRODUCTIVE_STAGES).mean()
        if last3_mask.sum() > 0 else 0.0)

    feats["c3_effective_time_efficiency"] = float(
        prod_time / (total_time - block_time + 1e-9))

    # ── Stage cycling quality ─────────────────────────────────────────────
    stage_list = sorted_stage["stage"].tolist()
    pet_cycles = 0
    for i in range(len(stage_list)-2):
        if (stage_list[i]   == "Production" and
            stage_list[i+1] == "Evaluation" and
            stage_list[i+2] == "Refinement"):
            pet_cycles += 1
    feats["c3_productive_iteration_index"] = float(pet_cycles / (len(stage_list) + 1e-9))

    regressive = ["Production→Preparation","Evaluation→Ideation",
                  "Refinement→Ideation","Refinement→Production",
                  "Evaluation→Production"]
    fwd_back = {"→".join([stage_list[i], stage_list[i+1]]) for i in range(len(stage_list)-1)}
    reg_count = sum(1 for t in [f"{stage_list[i]}→{stage_list[i+1]}"
                                  for i in range(len(stage_list)-1)] if t in regressive)
    feats["c3_regressive_transition_ratio"] = float(reg_count / (len(stage_list) + 1e-9))

    # Block emotional trajectory slope
    block_traj_slopes = []
    for _, row in blocks_df.iterrows():
        bm = (fused["time_s"] >= row["start"]) & (fused["time_s"] < row["end"])
        if bm.sum() < FPS * 3: continue
        trajectory = happy[bm.values] - stress[bm.values]
        slope, *_ = linregress(np.arange(len(trajectory)), sg(trajectory))
        block_traj_slopes.append(slope)
    feats["c3_block_emotional_trajectory_slope"] = float(np.mean(block_traj_slopes)) if block_traj_slopes else 0.0

    # Post-block engagement quality
    post_happy = []
    session_mean_happy = float(happy.mean())
    for _, row in blocks_df.iterrows():
        post_mask = (fused["time_s"] >= row["end"]) & (fused["time_s"] < row["end"] + 120)
        if post_mask.sum() > 5:
            post_happy.append(float(happy[post_mask.values].mean()))
    feats["c3_post_block_engagement_quality"] = float(
        np.mean(post_happy) / (session_mean_happy + 1e-9)) if post_happy else 1.0

    # Struggle benefit index
    if len(productive_post_durations) >= 3 and len(block_durations) >= 3:
        n_common = min(len(block_durations), len(productive_post_durations))
        r, _ = pearsonr(block_durations[:n_common], productive_post_durations[:n_common])
        feats["c3_struggle_benefit_index"] = float(r)
    else:
        feats["c3_struggle_benefit_index"] = 0.0

    return feats


# ============================================================================
# D4: AFFECTIVE SELF-AWARENESS  — 25 features
# ============================================================================

def compute_d4(fused: pd.DataFrame, sam_valence: list, sam_arousal: list) -> dict:
    feats   = {}
    v_arr   = _fill_sam(sam_valence)
    a_arr   = _fill_sam(sam_arousal)
    n_chk   = len(v_arr)
    t_max   = fused["time_s"].max()
    t_min   = fused["time_s"].min()
    chk_t   = np.linspace(t_min, t_max, n_chk)

    def window_mean(col, t_end, window=180):
        m = (fused["time_s"] >= t_end - window) & (fused["time_s"] < t_end)
        return float(fused.loc[m, col].mean()) if m.sum() > 5 else np.nan

    # Per-checkpoint facial valence index: Happiness - mean(neg)
    neg_cols_emo = [e for e in NEGATIVE_EMOS if e in fused.columns]
    fv_list = []
    for t in chk_t:
        h  = window_mean("Happiness", t)
        ns = [window_mean(e, t) for e in neg_cols_emo]
        ns = [x for x in ns if not np.isnan(x)]
        fv = h - np.mean(ns) if (not np.isnan(h) and ns) else np.nan
        fv_list.append(fv)
    fv_arr = np.array(fv_list, dtype=float)
    valid_both = ~(np.isnan(v_arr) | np.isnan(fv_arr))

    # ── SAM-AU concordance ────────────────────────────────────────────────
    happy_per_chk = np.array([window_mean("Happiness", t) for t in chk_t])
    if valid_both.sum() >= 3:
        r, _ = pearsonr(v_arr[valid_both], happy_per_chk[valid_both & ~np.isnan(happy_per_chk)])
        feats["c4_valence_happiness_concordance"] = float(r)
    else:
        feats["c4_valence_happiness_concordance"] = 0.0

    neg_comp_mean = np.array([
        np.nanmean([window_mean(e, t) for e in neg_cols_emo[:4]])
        for t in chk_t])
    inv_v = 6 - v_arr
    valid_n = ~(np.isnan(inv_v) | np.isnan(neg_comp_mean))
    if valid_n.sum() >= 3:
        r, _ = pearsonr(inv_v[valid_n], neg_comp_mean[valid_n])
        feats["c4_valence_negemo_concordance"] = float(r)
    else:
        feats["c4_valence_negemo_concordance"] = 0.0

    au_act_per_chk = np.array([window_mean_au(fused, t) for t in chk_t])
    valid_a = ~(np.isnan(a_arr) | np.isnan(au_act_per_chk))
    if valid_a.sum() >= 3:
        r, _ = pearsonr(a_arr[valid_a], au_act_per_chk[valid_a])
        feats["c4_arousal_AU_concordance"] = float(r)
    else:
        feats["c4_arousal_AU_concordance"] = 0.0

    stress_per_chk = np.array([window_mean("Stress", t) for t in chk_t])
    valid_sv = ~(np.isnan(v_arr) | np.isnan(stress_per_chk))
    if valid_sv.sum() >= 3:
        r, _ = pearsonr(v_arr[valid_sv], -stress_per_chk[valid_sv])
        feats["c4_valence_stress_inverse_concordance"] = float(r)
    else:
        feats["c4_valence_stress_inverse_concordance"] = 0.0

    # ── SAM trajectory features ───────────────────────────────────────────
    if n_chk >= 4:
        t_idx = np.arange(n_chk, dtype=float)
        coeffs = np.polyfit(t_idx, v_arr, 2)
        feats["c4_valence_curvature"] = float(coeffs[0])  # positive = U-shape

        dv = np.diff(v_arr)
        turns = int(np.sum(np.diff(np.sign(dv)) != 0))
        feats["c4_valence_turning_points"] = float(turns)

        slope, *_ = linregress(t_idx, a_arr)
        feats["c4_arousal_trend"]    = float(slope)
        feats["c4_arousal_stability"] = float(-a_arr.std())
    else:
        feats["c4_valence_curvature"]   = 0.0
        feats["c4_valence_turning_points"] = 0.0
        feats["c4_arousal_trend"]       = 0.0
        feats["c4_arousal_stability"]   = 0.0

    # ── Interoceptive accuracy ────────────────────────────────────────────
    z_sam = (v_arr - v_arr.mean()) / (v_arr.std() + 1e-9)
    z_fv  = (fv_arr - np.nanmean(fv_arr)) / (np.nanstd(fv_arr) + 1e-9)
    valid_both_z = ~(np.isnan(z_sam) | np.isnan(z_fv))
    if valid_both_z.sum() >= 3:
        r, _ = pearsonr(z_sam[valid_both_z], z_fv[valid_both_z])
        feats["c4_facial_valence_accuracy"] = float(r)
    else:
        feats["c4_facial_valence_accuracy"] = 0.0

    # Change detection alignment
    if valid_both_z.sum() >= 4:
        d_sam = np.diff(z_sam); d_fv = np.diff(z_fv)
        valid_d = ~(np.isnan(d_sam) | np.isnan(d_fv))
        if valid_d.sum() >= 3:
            r, _ = pearsonr(d_sam[valid_d], d_fv[valid_d])
            feats["c4_change_detection_alignment"] = float(r)
        else:
            feats["c4_change_detection_alignment"] = 0.0
    else:
        feats["c4_change_detection_alignment"] = 0.0

    # Block stage awareness
    block_chk_idx = []
    for i, t in enumerate(chk_t):
        window = fused[(fused["time_s"] >= t - 90) & (fused["time_s"] < t)]
        if window["stage"].isin(BLOCK_STAGES).mean() > 0.3:
            block_chk_idx.append(i)
    if len(block_chk_idx) >= 2:
        v_block = v_arr[block_chk_idx]
        neg_block = neg_comp_mean[block_chk_idx]
        valid_bl = ~(np.isnan(v_block) | np.isnan(neg_block))
        if valid_bl.sum() >= 2:
            r, _ = pearsonr(-v_block[valid_bl], neg_block[valid_bl])
            feats["c4_block_stage_awareness"] = float(r)
        else:
            feats["c4_block_stage_awareness"] = 0.0
    else:
        feats["c4_block_stage_awareness"] = 0.0

    # ── Self-report calibration ───────────────────────────────────────────
    feats["c4_positive_reporting_bias"] = float(
        np.nanmean(z_sam - z_fv)) if valid_both_z.sum() >= 3 else 0.0
    feats["c4_absolute_calibration_error"] = float(
        -np.nanmean(np.abs(z_sam[valid_both_z] - z_fv[valid_both_z]))
        ) if valid_both_z.sum() >= 3 else 0.0

    # ── Arousal-performance alignment ─────────────────────────────────────
    feats["c4_arousal_optimality"]       = float(-abs(a_arr.mean() - 3.0))
    feats["c4_arousal_midrange_proportion"] = float(((a_arr >= 2) & (a_arr <= 4)).mean())

    # ── Emotional self-efficacy ───────────────────────────────────────────
    dips = np.where(np.diff(v_arr) < 0)[0]
    recovery_mags = []
    for d in dips:
        if d + 1 < n_chk:
            recovery_mags.append(max(0.0, v_arr[d+1] - v_arr[d]))
    feats["c4_valence_recovery_magnitude"] = float(np.mean(recovery_mags)) if recovery_mags else 0.0

    if "Pride" in fused.columns:
        prod_mask = fused["stage"].isin(PRODUCTIVE_STAGES)
        global_mean_pride = float(fused["Pride"].mean()) + 1e-9
        feats["c4_pride_during_production"] = float(
            fused.loc[prod_mask, "Pride"].mean() / global_mean_pride
            if prod_mask.sum() > 5 else 1.0)
    else:
        feats["c4_pride_during_production"] = 1.0

    # Stage emotion appropriateness
    appropriate_scores = []
    mappings = [("Ideation", "Happiness"), ("Evaluation", "Stress"),
                ("Refinement", "Happiness"), ("Block", "Anger")]
    for st, emo_target in mappings:
        mask = fused["stage"] == st
        if mask.sum() > 5 and emo_target in fused.columns:
            z = (fused.loc[mask, emo_target].mean() - fused[emo_target].mean()) / \
                (fused[emo_target].std() + 1e-9)
            appropriate_scores.append(float(z))
    feats["c4_stage_emotion_appropriateness"] = float(np.mean(appropriate_scores)) if appropriate_scores else 0.0

    eval_mask = fused["stage"] == "Evaluation"
    if eval_mask.sum() > 5:
        eval_stress = fused.loc[eval_mask, "Stress"].values
        # Get nearest SAM arousal checkpoints
        eval_t = fused.loc[eval_mask, "time_s"].mean()
        nearest_chk = np.argmin(np.abs(chk_t - eval_t))
        if 0 <= nearest_chk < n_chk:
            r, _ = pearsonr(
                sg(eval_stress)[:min(len(eval_stress), 10)],
                np.full(min(len(eval_stress), 10), a_arr[nearest_chk])
            ) if len(eval_stress) >= 5 else (0.0, 1.0)
            feats["c4_evaluation_anxiety_alignment"] = float(
                fused.loc[eval_mask, "Stress"].mean() * a_arr[nearest_chk] / 25.0)
        else:
            feats["c4_evaluation_anxiety_alignment"] = 0.0
    else:
        feats["c4_evaluation_anxiety_alignment"] = 0.0

    # ── Alexithymia-reversed features ────────────────────────────────────
    fv_range = np.nanmax(fv_arr) - np.nanmin(fv_arr) + 1e-9
    sam_range = v_arr.max() - v_arr.min() + 1e-9
    feats["c4_emotional_range_concordance"] = float(1 - abs(1 - sam_range / fv_range))

    # Negative emotion differentiation when SAM valence < median
    v_med = np.nanmedian(v_arr)
    low_v_times = chk_t[v_arr < v_med]
    if len(low_v_times) >= 3:
        low_v_emo = []
        for t in low_v_times:
            win = (fused["time_s"] >= t - 90) & (fused["time_s"] < t)
            if win.sum() > 5:
                row_vals = [fused.loc[win, e].mean() for e in NEGATIVE_EMOS[:4] if e in fused.columns]
                low_v_emo.append(row_vals)
        if len(low_v_emo) >= 3:
            mat = np.array(low_v_emo)
            feats["c4_negative_emotion_differentiation"] = float(1 - icc(mat))
        else:
            feats["c4_negative_emotion_differentiation"] = 0.0
    else:
        feats["c4_negative_emotion_differentiation"] = 0.0

    feats["c4_sam_response_range"]  = float(sam_range)
    feats["c4_anchoring_resistance"] = float(1 - ar1(v_arr))
    r_va = abs(pearsonr(v_arr, a_arr)[0]) if n_chk >= 3 else 0.0
    feats["c4_valence_arousal_discrimination"] = float(1 - r_va)

    # Temporal coherence (max cross-correlation at lags 0, ±1)
    if valid_both_z.sum() >= 4:
        d_sam = np.diff(z_sam); d_fv = np.diff(z_fv)
        valid_d = ~(np.isnan(d_sam) | np.isnan(d_fv))
        if valid_d.sum() >= 3:
            xcorrs = [pearsonr(d_sam[valid_d], d_fv[valid_d])[0]]
            if len(d_sam) > 2:
                xcorrs.append(pearsonr(d_sam[valid_d][:-1], d_fv[valid_d][1:])[0])
                xcorrs.append(pearsonr(d_sam[valid_d][1:],  d_fv[valid_d][:-1])[0])
            feats["c4_temporal_coherence"] = float(np.nanmax(xcorrs))
        else:
            feats["c4_temporal_coherence"] = 0.0
    else:
        feats["c4_temporal_coherence"] = 0.0

    return feats


def window_mean_au(fused, t_end, window=180):
    """Mean of all AU columns in a window (for arousal AU concordance)."""
    m = (fused["time_s"] >= t_end - window) & (fused["time_s"] < t_end)
    au_cols = [c for c in ALL_AU_COLS if c in fused.columns]
    if m.sum() < 5 or not au_cols: return np.nan
    return float(fused.loc[m, au_cols].values.mean())


# ============================================================================
# D5: EMOTION-COGNITION ALIGNMENT  — 22 features
# ============================================================================

def compute_d5(fused: pd.DataFrame, sam_arousal: list) -> dict:
    feats   = {}
    a_arr   = _fill_sam(sam_arousal)
    happy   = fused["Happiness"].values.astype(float)
    stress  = fused["Stress"].values.astype(float)
    anger   = fused["Anger"].values.astype(float) if "Anger" in fused.columns else np.zeros(len(fused))
    emo     = fused[ALL_EMOTIONS].values

    au04 = fused["AU04_r"].values if "AU04_r" in fused.columns else np.zeros(len(fused))
    au07 = fused["AU07_r"].values if "AU07_r" in fused.columns else np.zeros(len(fused))
    au01 = fused["AU01_r"].values if "AU01_r" in fused.columns else np.zeros(len(fused))
    au02 = fused["AU02_r"].values if "AU02_r" in fused.columns else np.zeros(len(fused))
    au12 = fused["AU12_r"].values if "AU12_r" in fused.columns else np.zeros(len(fused))
    au14 = fused["AU14_r"].values if "AU14_r" in fused.columns else np.zeros(len(fused))

    def stage_mean(arr, stage_name_set):
        m = fused["stage"].isin(stage_name_set).values
        return float(arr[m].mean()) if m.sum() > 5 else 0.0

    # ── Stage-appropriate emotion features ───────────────────────────────
    id_mask  = (fused["stage"] == "Ideation").values
    pr_mask  = fused["stage"].isin(PRODUCTIVE_STAGES).values
    ev_mask  = (fused["stage"] == "Evaluation").values
    bl_mask  = fused["stage"].isin(BLOCK_STAGES).values
    ref_mask = (fused["stage"] == "Refinement").values

    happy_mean = happy.mean() + 1e-9
    feats["c5_ideation_positive_affect"] = float(
        (happy[id_mask].mean() - happy_mean) / happy_mean if id_mask.sum() > 5 else 0.0)

    feats["c5_production_concentration"] = float(
        (au04[pr_mask] + au07[pr_mask]).mean() if pr_mask.sum() > 5 else 0.0)

    feats["c5_evaluation_critical_engagement"] = float(
        (au04[ev_mask] + au14[ev_mask]).mean() if ev_mask.sum() > 5 else 0.0)

    # Stage emotion profile fit (cosine similarity to ideal profiles)
    ideal = {
        "Ideation":    np.array([0.3, 0.2, 0.1, 0.1, 0.8, 0.5, 0.4]),  # high happy/pride
        "Production":  np.array([0.5, 0.3, 0.2, 0.1, 0.3, 0.5, 0.8]),  # high stress/focus
        "Evaluation":  np.array([0.5, 0.4, 0.2, 0.3, 0.4, 0.4, 1.0]),  # high stress
        "Refinement":  np.array([0.2, 0.2, 0.1, 0.1, 0.7, 0.8, 0.4]),  # high happy/pride
        "Block":       np.array([0.8, 0.7, 0.5, 0.6, 0.1, 0.1, 1.5]),  # high neg/stress
    }
    sims = []
    for st, ideal_v in ideal.items():
        mask = (fused["stage"] == st).values
        if mask.sum() < 5: continue
        obs = emo[mask].mean(axis=0)
        obs_n = obs / (np.linalg.norm(obs) + 1e-9)
        id_n  = ideal_v / (np.linalg.norm(ideal_v) + 1e-9)
        sims.append(float(np.dot(obs_n, id_n)))
    feats["c5_stage_emotion_profile_fit"] = float(np.mean(sims)) if sims else 0.0

    # ── Emotion-stage coupling (η²) ───────────────────────────────────────
    eta2_vals = []
    for i, em_name in enumerate(ALL_EMOTIONS):
        groups = [emo[(fused["stage"] == st).values, i]
                  for st in ALL_STAGES_ORDERED
                  if (fused["stage"] == st).sum() > 5]
        if len(groups) < 2: continue
        grand = np.concatenate(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand)**2 for g in groups)
        ss_total   = sum(((g - grand)**2).sum() for g in groups)
        eta2_vals.append(ss_between / (ss_total + 1e-9))
    feats["c5_emotion_stage_differentiation"] = float(np.mean(eta2_vals)) if eta2_vals else 0.0

    high_demand = ["Production","Evaluation"]
    low_demand  = ["Preparation","Refinement"]
    au04_hi = stage_mean(au04, high_demand)
    au04_lo = stage_mean(au04, low_demand)
    feats["c5_cognitive_effort_deployment"] = float(au04_hi - au04_lo)

    # ── Interest and curiosity ────────────────────────────────────────────
    interest_ideation    = (au01[id_mask] + au02[id_mask] + au07[id_mask]).mean() if id_mask.sum() > 5 else 0.0
    interest_production  = (au01[pr_mask] + au02[pr_mask] + au07[pr_mask]).mean() if pr_mask.sum() > 5 else 0.0
    feats["c5_interest_during_ideation"] = float(interest_ideation - interest_production)

    # ── Flow state indicators ──────────────────────────────────────────────
    if pr_mask.sum() > 5 * FPS:
        au07_prod = au07[pr_mask]; happy_prod = happy[pr_mask]; stress_prod = stress[pr_mask]
        def z_sc(a): return (a - a.mean()) / (a.std() + 1e-9)
        flow = (z_sc(au07_prod) + z_sc(happy_prod) - z_sc(stress_prod) - z_sc(au07_prod.std() * np.ones_like(au07_prod))) / 4.0
        feats["c5_flow_proxy_production"] = float(flow.mean())

        win = 5 * FPS
        flow_wins = [flow[i:i+win].mean() for i in range(0, len(flow)-win, win)]
        feats["c5_flow_stability"] = float(
            -(np.std(flow_wins) / (np.mean(np.abs(flow_wins)) + 1e-9))
            if len(flow_wins) > 1 else 0.0)
    else:
        feats["c5_flow_proxy_production"] = 0.0
        feats["c5_flow_stability"]        = 0.0

    # ── Boekaerts dual-processing ─────────────────────────────────────────
    p_mean_h = happy.mean(); p_mean_s = stress.mean()
    win = 5 * FPS
    mastery_count = 0; coping_count = 0; total_wins = 0
    for i in range(0, len(fused) - win, win):
        seg_h = happy[i:i+win].mean()
        seg_s = stress[i:i+win].mean()
        seg_a = anger[i:i+win].mean() if len(anger) > 0 else 0.0
        total_wins += 1
        if seg_h > p_mean_h and seg_s < p_mean_s and au12[i:i+win].mean() > 0.5:
            mastery_count += 1
        elif seg_s > p_mean_s and seg_h < p_mean_h and seg_a > p_mean_s * 0.5:
            coping_count += 1
    feats["c5_mastery_mode_ratio"] = float(
        mastery_count / (mastery_count + coping_count + 1e-9))

    mastery_challenge = 0; coping_challenge = 0; total_challenge = 0
    challenge_mask = fused["stage"].isin({"Production","Evaluation"}).values
    for i in range(0, len(fused) - win, win):
        if not challenge_mask[i:i+win].mean() > 0.5: continue
        seg_h = happy[i:i+win].mean()
        seg_s = stress[i:i+win].mean()
        total_challenge += 1
        if seg_h > p_mean_h and seg_s < p_mean_s:
            mastery_challenge += 1
        elif seg_s > p_mean_s and seg_h < p_mean_h:
            coping_challenge += 1
    feats["c5_mastery_during_challenge"] = float(
        mastery_challenge / (mastery_challenge + coping_challenge + 1e-9))

    # ── Confusion-resolution dynamics ─────────────────────────────────────
    au04_75  = np.percentile(au04, 75)
    au01_med = np.median(au01)
    confusion_episodes_resolved = 0; total_confusion = 0
    for i in range(len(fused) - FPS * 18):
        win3 = slice(i, i + FPS * 3)
        if au04[win3].mean() > au04_75 and au01[win3].mean() > au01_med:
            total_confusion += 1
            future = slice(i + FPS * 3, i + FPS * 18)
            if (au04[future].mean() < au04_75 * 0.75 and
                    au07[future].mean() > np.median(au07)):
                confusion_episodes_resolved += 1
    feats["c5_confusion_resolution_rate"] = float(
        confusion_episodes_resolved / (total_confusion + 1e-9))

    feats["c5_block_constructive_confusion"] = float(
        (au04[bl_mask] + au01[bl_mask]).mean() - anger[bl_mask].mean()
        if bl_mask.sum() > 5 else 0.0)

    # ── Stage transition quality ──────────────────────────────────────────
    trans_dists = []
    stage_df_sorted = fused.copy()
    stage_changes   = stage_df_sorted["stage"] != stage_df_sorted["stage"].shift()
    change_indices  = stage_df_sorted.index[stage_changes].tolist()
    for ci in change_indices[1:]:
        loc = stage_df_sorted.index.get_loc(ci)
        if loc < 15 or loc > len(fused) - 15: continue
        pre  = emo[max(0, loc-15*FPS//FPS):loc].mean(axis=0)
        post = emo[loc:min(len(fused), loc+15*FPS//FPS)].mean(axis=0)
        if pre.sum() > 0 and post.sum() > 0:
            trans_dists.append(float(np.linalg.norm(pre - post)))
    feats["c5_transition_emotional_preparation"] = float(-np.mean(trans_dists)) if trans_dists else 0.0

    if pr_mask.sum() > 5 and ev_mask.sum() > 5:
        pride = fused["Pride"].values if "Pride" in fused.columns else (au12 + au14) / 2
        # Last 30s of Production, first 30s of Evaluation
        prod_times = fused.loc[pr_mask, "time_s"]
        eval_times = fused.loc[ev_mask, "time_s"]
        if len(prod_times) > 0 and len(eval_times) > 0:
            prod_last = (fused["time_s"] >= prod_times.max() - 30) & fused["stage"].isin(PRODUCTIVE_STAGES)
            eval_first = (fused["time_s"] <= eval_times.min() + 30) & (fused["stage"] == "Evaluation")
            p_pride_last  = float(pride[prod_last.values].mean()) if prod_last.sum() > 0 else 0.0
            e_pride_first = float(pride[eval_first.values].mean()) if eval_first.sum() > 0 else 0.0
            feats["c5_pride_after_production"] = float(e_pride_first - p_pride_last)
        else:
            feats["c5_pride_after_production"] = 0.0
    else:
        feats["c5_pride_after_production"] = 0.0

    # ── Task-appropriate arousal ───────────────────────────────────────────
    opt_dev = []
    for st, opt in STAGE_OPTIMAL_STRESS.items():
        mask = (fused["stage"] == st).values
        if mask.sum() < 5: continue
        obs_z = (stress[mask].mean() - stress.mean()) / (stress.std() + 1e-9)
        opt_z = (opt - stress.mean()) / (stress.std() + 1e-9)
        opt_dev.append(abs(obs_z - opt_z))
    feats["c5_stress_appropriate_modulation"] = float(-np.mean(opt_dev)) if opt_dev else 0.0

    # SAM arousal vs stage complexity
    t_max = fused["time_s"].max(); t_min = fused["time_s"].min()
    chk_t = np.linspace(t_min, t_max, len(a_arr))
    complexity_at_chk = []
    for t in chk_t:
        win = (fused["time_s"] >= t - 90) & (fused["time_s"] < t)
        if win.sum() > 5:
            dominant_stage = fused.loc[win, "stage"].mode()
            if len(dominant_stage) > 0:
                complexity_at_chk.append(STAGE_COMPLEXITY.get(dominant_stage.iloc[0], 3))
            else:
                complexity_at_chk.append(np.nan)
        else:
            complexity_at_chk.append(np.nan)
    complexity_arr = np.array(complexity_at_chk, dtype=float)
    valid = ~(np.isnan(complexity_arr) | np.isnan(a_arr))
    if valid.sum() >= 3:
        rho, _ = spearmanr(a_arr[valid], complexity_arr[valid])
        feats["c5_arousal_SAM_task_match"] = float(rho)
    else:
        feats["c5_arousal_SAM_task_match"] = 0.0

    # ── Cross-stage differentiation ───────────────────────────────────────
    h_ideation   = happy[id_mask].mean()  if id_mask.sum() > 5 else 0.0
    h_production = happy[pr_mask].mean()  if pr_mask.sum() > 5 else 0.0
    au_conc_prod = (au04[pr_mask] + au07[pr_mask]).mean() if pr_mask.sum() > 5 else 0.0
    au_conc_idea = (au04[id_mask] + au07[id_mask]).mean() if id_mask.sum() > 5 else 0.0
    feats["c5_positive_creative_vs_focused_production"] = float(
        (h_ideation - h_production) + (au_conc_prod - au_conc_idea)) / 2.0

    neg_emo = fused[[e for e in NEGATIVE_EMOS if e in fused.columns]].values.mean(axis=1)
    feats["c5_negative_affect_block_specificity"] = float(
        neg_emo[bl_mask].mean() - neg_emo[~bl_mask].mean()
        if bl_mask.sum() > 5 else 0.0)

    win10 = 10 * FPS
    complexity_per_win = []
    for i in range(0, len(fused) - win10, win10):
        active = (emo[i:i+win10] > 1.0).sum(axis=1).mean()
        complexity_per_win.append(float(active))
    feats["c5_emotion_complexity_index"] = float(np.mean(complexity_per_win)) if complexity_per_win else 0.0

    # SAM-AU coherence
    neg_cols_emo = [e for e in NEGATIVE_EMOS if e in fused.columns]
    fv_sad = [float(fused.loc[(fused["time_s"] >= t-90) & (fused["time_s"]<t), "Happiness"].mean()
               - fused.loc[(fused["time_s"] >= t-90) & (fused["time_s"]<t),
                            ["Sadness"] if "Sadness" in fused.columns else neg_cols_emo[:1]].values.mean())
              if ((fused["time_s"] >= t-90) & (fused["time_s"]<t)).sum() > 5 else np.nan
              for t in chk_t]
    fv_str = [float(fused.loc[(fused["time_s"] >= t-90) & (fused["time_s"]<t),
                               ["Stress"] + (["Fear"] if "Fear" in fused.columns else [])].values.mean())
              if ((fused["time_s"] >= t-90) & (fused["time_s"]<t)).sum() > 5 else np.nan
              for t in chk_t]
    v_arr = _fill_sam(list(a_arr))  # reuse for coherence check
    sam_v = _fill_sam(list(a_arr))
    coh_vals = []
    fv_sad_arr = np.array(fv_sad); fv_str_arr = np.array(fv_str)
    valid1 = ~np.isnan(fv_sad_arr); valid2 = ~np.isnan(fv_str_arr)
    if valid1.sum() >= 3:
        r1, _ = pearsonr(sam_v[valid1], fv_sad_arr[valid1])
        coh_vals.append(r1)
    if valid2.sum() >= 3:
        r2, _ = pearsonr(a_arr[valid2], fv_str_arr[valid2])
        coh_vals.append(r2)
    feats["c5_SAM_AU_coherence"] = float(np.mean(coh_vals)) if coh_vals else 0.0

    # Engagement recovery speed
    stress_75 = np.percentile(stress, 75)
    au07_med  = np.median(au07)
    stress_med = np.median(stress)
    recovery_speeds = []
    in_episode = False; ep_start = 0
    for i in range(len(fused) - FPS * 3):
        if not in_episode:
            if stress[i:i+FPS*3].mean() > stress_75 or anger[i:i+FPS*3].mean() > stress_75:
                in_episode = True; ep_start = i
        else:
            if au07[i] > au07_med and stress[i] < stress_med:
                recovery_speeds.append((i - ep_start) / FPS)
                in_episode = False
    feats["c5_engagement_recovery_speed"] = float(
        -np.mean(recovery_speeds) if recovery_speeds else -60.0)

    return feats


# ============================================================================
# PROCESS ALL PARTICIPANTS
# ============================================================================

def process_all(data_dir: Path) -> tuple:
    for zname, dname in [(STAGE_ZIP, STAGE_DIR), (FACIAL_ZIP, FACIAL_DIR)]:
        zp = data_dir / zname; od = data_dir / dname
        if zp.exists(): extract_zip(zp, od)

    stage_files  = find_files(data_dir / STAGE_DIR)
    facial_files = find_files(data_dir / FACIAL_DIR)
    exam_data    = load_exam(data_dir / EXAM_FILE)
    sam_data     = load_sam(data_dir / SAM_FILE)

    common = sorted(
        set(stage_files) & set(facial_files) & set(exam_data) & set(sam_data),
        key=lambda p: int(re.search(r"\d+", p).group())
    )
    log.info("Participants found: %d", len(common))

    all_features = {}
    for pid in common:
        try:
            facial = load_facial(facial_files[pid])
            stage  = load_stage(stage_files[pid])
            if facial.empty or stage.empty: continue
            facial = compute_emotions(facial)
            fused  = align_stages(facial, stage)
            fused  = fused[np.isfinite(fused["time_s"].values)].copy()
            if len(fused) < 100: continue

            sam = sam_data[pid]
            d1  = compute_d1(fused)
            d2  = compute_d2(fused, sam["valence"], sam["arousal"])
            d3  = compute_d3(fused, stage)
            d4  = compute_d4(fused, sam["valence"], sam["arousal"])
            d5  = compute_d5(fused, sam["arousal"])

            all_features[pid] = {**d1, **d2, **d3, **d4, **d5}
            log.info("[%s] OK  —  %d features extracted", pid, len(all_features[pid]))

        except Exception as e:
            log.error("[%s] failed: %s", pid, e)
            import traceback; traceback.print_exc()

    return all_features, exam_data


# ============================================================================
# PHASE 1: MEASUREMENT MODEL — VIF (formative) / McDonald's ω (reflective)
# ============================================================================

def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor for each column. VIF < 5 = acceptable."""
    from numpy.linalg import inv
    Xs = StandardScaler().fit_transform(X.fillna(0))
    corr = np.corrcoef(Xs.T)
    try:
        vif = np.diag(inv(corr))
    except np.linalg.LinAlgError:
        vif = np.full(X.shape[1], np.nan)
    return pd.Series(vif, index=X.columns)

def mcdonalds_omega(X: pd.DataFrame) -> float:
    """McDonald's ω (total) — does not require tau-equivalence."""
    try:
        from numpy.linalg import eigh
        Xs = StandardScaler().fit_transform(X.dropna())
        cov = np.cov(Xs.T)
        evals, evecs = eigh(cov)
        # First factor loadings
        lam = evecs[:, -1] * np.sqrt(evals[-1])
        omega = lam.sum()**2 / (lam.sum()**2 + (np.diag(cov) - lam**2).sum())
        return float(np.clip(omega, -1, 1))
    except:
        return np.nan

def classify_dimension(dim_features: pd.DataFrame, dim_name: str) -> dict:
    """
    Classify dimension as formative or reflective based on inter-feature
    correlation pattern and compute appropriate reliability metric.
    """
    if dim_features.shape[1] < 2:
        return {"dimension": dim_name, "type": "insufficient_features",
                "metric": np.nan, "action": "need more features"}

    corr_mat = dim_features.corr().values
    np.fill_diagonal(corr_mat, np.nan)
    mean_inter_r = float(np.nanmean(corr_mat))

    # Negative or near-zero mean inter-feature r → formative
    is_formative = mean_inter_r < 0.15

    result = {"dimension": dim_name, "mean_inter_r": round(mean_inter_r, 3)}
    if is_formative:
        result["type"]   = "FORMATIVE"
        result["metric"] = "VIF (use < 5.0 threshold)"
        vif = compute_vif(dim_features)
        result["vif_max"]  = float(vif.max())
        result["vif_mean"] = float(vif.mean())
        result["pass_vif"] = bool(vif.max() < 5.0)
        result["action"]   = ("RETAIN all — formative construct, negative inter-r expected."
                               if result["pass_vif"] else
                               "DROP collinear indicators (VIF > 5)")
    else:
        result["type"]   = "REFLECTIVE"
        result["metric"] = "McDonald's ω"
        result["omega"]  = mcdonalds_omega(dim_features)
        result["action"] = ("RETAIN — ω acceptable"
                             if result["omega"] > 0.6 else
                             "REVIEW — ω below threshold, revise indicator selection")
    return result


# ============================================================================
# PHASE 2: CRITERION VALIDITY — bootstrap BCa, permutation p, BH-FDR
# ============================================================================

def bca_bootstrap_ci(x, y, n_boot=2000, alpha=0.05):
    """Bias-corrected and accelerated (BCa) bootstrap CI for Pearson r."""
    rng = np.random.default_rng(42)
    n   = len(x)
    obs_r = pearsonr(x, y)[0]
    boot_r = np.array([
        pearsonr(x[idx := rng.integers(0, n, n)], y[idx])[0]
        for _ in range(n_boot)
    ])
    # Bias correction
    z0 = float(np.sum(boot_r < obs_r) / n_boot)
    z0 = np.clip(z0, 1e-6, 1-1e-6)
    z0_val = float(np.log(z0 / (1 - z0)))  # probit-like

    # Acceleration (jackknife)
    jk = np.array([pearsonr(np.delete(x, i), np.delete(y, i))[0] for i in range(n)])
    jk_mean = jk.mean()
    num  = np.sum((jk_mean - jk)**3)
    den  = 6 * (np.sum((jk_mean - jk)**2)**1.5 + 1e-9)
    a    = float(num / den)

    from scipy.stats import norm
    z_alpha_lo = norm.ppf(alpha / 2)
    z_alpha_hi = norm.ppf(1 - alpha / 2)

    def adj_pct(z_alpha):
        return norm.cdf(z0_val + (z0_val + z_alpha) / (1 - a*(z0_val + z_alpha) + 1e-9))

    lo = float(np.percentile(boot_r, 100 * adj_pct(z_alpha_lo)))
    hi = float(np.percentile(boot_r, 100 * adj_pct(z_alpha_hi)))
    return obs_r, lo, hi

def permutation_p(x, y, n_perm=5000):
    """Two-tailed permutation p-value for Pearson r."""
    rng = np.random.default_rng(0)
    obs_r = abs(pearsonr(x, y)[0])
    perm_r = np.array([abs(pearsonr(x, rng.permutation(y))[0]) for _ in range(n_perm)])
    return float((perm_r >= obs_r).mean())

def bh_fdr(p_values: list) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    idx = np.argsort(p_values)
    p_sorted = np.array(p_values)[idx]
    adj = np.minimum(1.0, p_sorted * n / (np.arange(1, n+1)))
    # Ensure monotonicity (cumulative min from right)
    for i in range(n-2, -1, -1):
        adj[i] = min(adj[i], adj[i+1])
    result = np.empty(n)
    result[idx] = adj
    return result

def criterion_validity(feature_matrix: pd.DataFrame, exam_scores: dict,
                       n_boot=2000, n_perm=5000) -> pd.DataFrame:
    pids = sorted(feature_matrix.index,
                  key=lambda p: int(re.search(r"\d+", p).group()))
    y    = np.array([exam_scores[p] for p in pids])
    rows = []

    for col in feature_matrix.columns:
        x = feature_matrix.loc[pids, col].fillna(0).values
        if x.std() < 1e-9: continue
        try:
            r, lo, hi = bca_bootstrap_ci(x, y, n_boot=n_boot)
            perm_pval  = permutation_p(x, y, n_perm=n_perm)
            rho, _     = spearmanr(x, y)
            rows.append({
                "Feature":    col,
                "Pearson_r":  round(r, 4),
                "BCa_lo":     round(lo, 4),
                "BCa_hi":     round(hi, 4),
                "Perm_p":     round(perm_pval, 4),
                "Spearman_r": round(rho, 4),
                "CI_excl_0":  bool(lo > 0 or hi < 0),
            })
        except Exception as e:
            log.warning("Criterion validity failed for %s: %s", col, e)

    df = pd.DataFrame(rows)
    if df.empty: return df
    df["FDR_p"]       = bh_fdr(df["Perm_p"].tolist())
    df["Significant"] = (df["FDR_p"] < 0.05) | df["CI_excl_0"]
    df["Strength"]    = df["Pearson_r"].abs().apply(
        lambda r: "large" if r >= 0.50 else ("medium" if r >= 0.30 else "small"))
    df = df.sort_values("Pearson_r", key=abs, ascending=False)
    return df


# ============================================================================
# PHASE 3: FEATURE SELECTION — ElasticNet + bootstrap stability (≥75%)
# ============================================================================

def bootstrap_stability_selection(X: np.ndarray, y: np.ndarray,
                                   feature_names: list,
                                   n_boot=1000, threshold=0.75) -> pd.DataFrame:
    """
    Meinshausen & Bühlmann (2010) bootstrap stability selection.
    Returns features selected in ≥ threshold fraction of resamples.
    """
    rng      = np.random.default_rng(42)
    n, p     = X.shape
    sel_counts = np.zeros(p)

    for _ in range(n_boot):
        idx   = rng.integers(0, n, n)
        X_b   = X[idx]; y_b = y[idx]
        scaler = StandardScaler()
        X_bs  = scaler.fit_transform(X_b)
        try:
            model = ElasticNetCV(cv=min(5, n), l1_ratio=[0.5, 0.7, 0.9],
                                  max_iter=2000, random_state=int(_ % 1000))
            model.fit(X_bs, y_b)
            selected = (np.abs(model.coef_) > 1e-9).astype(float)
            sel_counts += selected
        except:
            pass

    stability = sel_counts / n_boot
    df = pd.DataFrame({"Feature": feature_names, "Stability": stability})
    df = df.sort_values("Stability", ascending=False)
    df["Selected"] = df["Stability"] >= threshold
    return df

def run_feature_selection(feature_matrix: pd.DataFrame, exam_scores: dict,
                           out_dir: Path) -> pd.DataFrame:
    pids = sorted(feature_matrix.index,
                  key=lambda p: int(re.search(r"\d+", p).group()))
    y    = np.array([exam_scores[p] for p in pids])
    X    = feature_matrix.loc[pids].fillna(0).values
    names = list(feature_matrix.columns)

    log.info("Running bootstrap stability selection (%d features, n=%d) ...",
             len(names), len(pids))
    stability_df = bootstrap_stability_selection(X, y, names, n_boot=500)
    stability_df.to_csv(out_dir / "stability_selection.csv", index=False)

    # LOOCV R² with stable features
    stable_feats = stability_df[stability_df["Selected"]]["Feature"].tolist()
    if len(stable_feats) == 0:
        log.warning("No features cleared stability threshold — using top 4 by stability.")
        stable_feats = stability_df.head(4)["Feature"].tolist()

    # Cap at 4 predictors (10:1 rule for n=42)
    stable_feats = stable_feats[:4]
    log.info("Stable features selected (%d): %s", len(stable_feats), stable_feats)

    Xs   = StandardScaler().fit_transform(X[:, [names.index(f) for f in stable_feats]])
    loocv_preds = []
    for i in range(len(pids)):
        X_tr = np.delete(Xs, i, axis=0); y_tr = np.delete(y, i)
        X_te = Xs[i:i+1]
        model = RidgeCV(alphas=[0.01, 0.1, 1, 10])
        model.fit(X_tr, y_tr)
        loocv_preds.append(float(model.predict(X_te)[0]))

    loocv_preds = np.array(loocv_preds)
    loocv_r2    = r2_score(y, loocv_preds)
    baseline_r2 = r2_score(y, np.full_like(y, y.mean()))
    log.info("LOOCV R² (stable features): %.4f  |  baseline: %.4f", loocv_r2, baseline_r2)

    return stability_df, stable_feats, loocv_r2


# ============================================================================
# MAIN
# ============================================================================

def main(data_dir: str):
    data_path = Path(data_dir)
    out_dir   = data_path / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  EI CANDIDATE FEATURE ENGINEERING — 116 features")
    log.info("=" * 60)

    # ── Load and extract ──────────────────────────────────────────────────
    all_features, exam_data = process_all(data_path)
    if len(all_features) < 5:
        log.error("Too few participants (%d). Stopping.", len(all_features)); return

    feature_matrix = pd.DataFrame(all_features).T
    feature_matrix.to_csv(out_dir / "all_features_116.csv")
    log.info("Feature matrix: %d participants × %d features",
             *feature_matrix.shape)

    # ── Phase 1: Measurement model ────────────────────────────────────────
    log.info("\n%s\nPHASE 1: Measurement Model Classification\n%s", "─"*60, "─"*60)
    dim_prefixes = {"D1": "c1_", "D2": "c2_", "D3": "c3_", "D4": "c4_", "D5": "c5_"}
    mm_results = []
    for dim, prefix in dim_prefixes.items():
        cols = [c for c in feature_matrix.columns if c.startswith(prefix)]
        if not cols: continue
        result = classify_dimension(feature_matrix[cols].dropna(), dim)
        mm_results.append(result)
        log.info("[%s] type=%-10s  metric=%s  action=%s",
                 dim, result["type"], result.get("metric","—"), result["action"])

    mm_df = pd.DataFrame(mm_results)
    mm_df.to_csv(out_dir / "measurement_model.csv", index=False)

    # ── Phase 2: Criterion validity ───────────────────────────────────────
    log.info("\n%s\nPHASE 2: Criterion Validity (bootstrap BCa + permutation FDR)\n%s",
             "─"*60, "─"*60)
    cv_df = criterion_validity(feature_matrix, exam_data, n_boot=2000, n_perm=5000)
    cv_df.to_csv(out_dir / "criterion_validity.csv", index=False)

    sig_feats = cv_df[cv_df["Significant"]]
    log.info("Features significant after FDR correction or CI excludes 0: %d / %d",
             len(sig_feats), len(cv_df))
    for _, row in sig_feats.head(15).iterrows():
        log.info("  %-42s r=%+.3f [%.3f, %.3f]  FDR_p=%.4f  (%s)",
                 row["Feature"], row["Pearson_r"],
                 row["BCa_lo"], row["BCa_hi"],
                 row["FDR_p"], row["Strength"])

    # ── Phase 3: Feature selection ────────────────────────────────────────
    log.info("\n%s\nPHASE 3: Bootstrap Stability Selection\n%s", "─"*60, "─"*60)
    stability_df, stable_feats, loocv_r2 = run_feature_selection(
        feature_matrix, exam_data, out_dir)

    # ── Phase 4: Report ───────────────────────────────────────────────────
    lines = [
        "=" * 72,
        "  EI CANDIDATE FEATURE ENGINEERING — PILOT STUDY REPORT",
        "  n=42 < 150 minimum for formal validation — EXPLORATORY ONLY",
        "=" * 72,
        f"  Features extracted:          {feature_matrix.shape[1]}",
        f"  Participants:                {feature_matrix.shape[0]}",
        "",
        "─" * 72,
        "  PHASE 1 — MEASUREMENT MODEL",
        "─" * 72,
    ]
    for r in mm_results:
        lines.append(f"  {r['dimension']}  [{r['type']:<10}]  action: {r['action']}")

    lines += [
        "",
        "─" * 72,
        "  PHASE 2 — TOP FEATURES BY CRITERION VALIDITY (FDR-corrected)",
        "─" * 72,
        f"  {'Feature':<42} {'r':>6} {'BCa CI':>18} {'FDR_p':>8} {'Size':>7}",
        "  " + "─" * 68,
    ]
    for _, row in cv_df.head(20).iterrows():
        sig_marker = " *" if row["Significant"] else ""
        lines.append(
            f"  {row['Feature']:<42} {row['Pearson_r']:>+6.3f} "
            f"[{row['BCa_lo']:>+6.3f},{row['BCa_hi']:>+6.3f}] "
            f"{row['FDR_p']:>8.4f} {row['Strength']:>7}{sig_marker}"
        )
    lines += [
        "",
        "─" * 72,
        "  PHASE 3 — BOOTSTRAP STABILITY SELECTION",
        "─" * 72,
        f"  Stable features (≥75% resamples):  {stable_feats}",
        f"  LOOCV R² with ≤4 stable features:  {loocv_r2:.4f}",
        "",
        "─" * 72,
        "  PILOT STUDY CAVEATS",
        "─" * 72,
        "  • n=42 provides 80% power only for large effects (|r| ≈ 0.42)",
        "  • Any significant correlations should be treated as exploratory",
        "  • Winner's curse: effect sizes may be inflated at first-pass significance",
        "  • Replicate with n ≥ 150 for formal psychometric validation",
        "  • Formative dimensions (D1, D2, D5): use VIF not Cronbach's α",
        "",
        "─" * 72,
        "  OUTPUT FILES",
        "─" * 72,
        "  all_features_116.csv       — full feature matrix (participants × features)",
        "  measurement_model.csv      — formative/reflective classification per dimension",
        "  criterion_validity.csv     — BCa CIs, permutation p, FDR-corrected p per feature",
        "  stability_selection.csv    — bootstrap stability scores (500 resamples)",
        "=" * 72,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(out_dir / "candidate_features_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    log.info("All outputs saved → %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EI Candidate Feature Engineering")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    main(args.data_dir)
