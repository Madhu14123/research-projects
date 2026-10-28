"""
EI Advanced Temporal Analysis
==============================
Goes beyond scalar features to analyse the *shape* of each participant's
emotional trajectory through the 30-minute task.

Seven advanced temporal approaches:

1. LOOCV-validated EI score     — clean hold-out validation of the composite
2. Phase portrait (attractor)   — Stress vs Happiness in 2D; shape reveals
                                   regulation style (confined = regulated,
                                   sprawling = chaotic)
3. Temporal segmentation        — Early / Mid / Late task emotional profiles;
                                   who improves across the session?
4. Emotion state machine        — Discretise into states (Calm/Stressed/Happy/
                                   Mixed); compute transition matrix per person
5. Block-triggered response     — Event-locked analysis: what happens to
                                   emotions in the 60s before and after each
                                   Block episode? (like ERP in neuroscience)
6. Cross-lagged correlation     — Does stress at time t predict being in Block
                                   at time t+k? Detects anticipatory stress.
7. Regulation speed distribution— For each participant: distribution of how
                                   many seconds it takes to drop stress by 20%
                                   after a peak; reveals individual differences
                                   in recovery consistency.

References:
  Phase portraits: Takens (1981) Detecting Strange Attractors in Turbulence.
  Recurrence analysis: Marwan et al. (2007) Phys. Rep. 438, 237-329.
  Event-locked ERP paradigm: Luck (2014) Introduction to ERP Technique.
  Cross-lagged correlation: Granger (1969) Econometrica 37, 424-438.

Results → results_temporal/

Usage:
    python ei_temporal_analysis.py
    python ei_temporal_analysis.py --data_dir "D:/A_emotions in exam/data"
"""

import re, zipfile, argparse, warnings, logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, correlate
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, linregress
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DATA_DIR = r"D:\A_emotions in exam\data"
RESULTS_DIR      = "results_temporal"

STAGE_ZIP  = "stage coded data-20260321T120851Z-1-001.zip"
FACIAL_ZIP = "facial expressions- data-20260321T182549Z-1-001.zip"
EXAM_FILE  = "Exam Scores.xlsx"
SAM_FILE   = "SAM Responses every 3 minutes.xlsx"
STAGE_DIR  = "stage_coded_data"
FACIAL_DIR = "facial_expressions"

FPS = 25

EMOTION_AU = {
    "Anger":     ["AU04_r","AU05_r","AU07_r","AU23_r"],
    "Fear":      ["AU01_r","AU04_r","AU05_r"],
    "Sadness":   ["AU01_r","AU04_r","AU15_r"],
    "Disgust":   ["AU04_r","AU07_r","AU09_r","AU10_r","AU17_r","AU20_r"],
    "Pride":     ["AU06_r","AU07_r","AU12_r"],
    "Happiness": ["AU06_r","AU12_r"],
    "Stress":    ["AU01_r","AU06_r","AU12_r","AU15_r"],
}
POSITIVE_EMOTIONS = ["Happiness","Pride"]
NEGATIVE_EMOTIONS = ["Anger","Fear","Sadness","Disgust","Stress"]
STAGE_MAP = {
    "preparation":"Preparation","ideation":"Ideation",
    "production":"Production","prod":"Production",
    "evaluation":"Evaluation","refinement":"Refinement",
    "block":"Block","sam":"SAM",
}
PRODUCTIVE_STAGES = {"Production","Refinement","Evaluation","prod","Prod"}
BLOCK_STAGES      = {"Block","Block ","block"}

# Emotional state bins (discretisation thresholds on AU scale 0-5)
STATE_CALM    = 0   # stress < 0.5 and happiness < 0.5
STATE_STRESSED= 1   # stress >= 1.0
STATE_HAPPY   = 2   # happiness >= 0.5
STATE_MIXED   = 3   # other

STATE_NAMES  = ["Calm","Stressed","Happy","Mixed"]
STATE_COLORS = ["#AED6F1","#E74C3C","#27AE60","#F39C12"]

EI_FEATURES = ["c3_block_score","c4_trend","c3_productive_ratio","c4_mean_valence"]
EI_WEIGHTS  = [0.40, 0.25, 0.20, 0.15]

# ============================================================================
# DATA LOADING
# ============================================================================

def extract_zip(zp, od):
    if od.exists() and any(od.iterdir()): return
    od.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp,"r") as z: z.extractall(od)

def pid_from(fn): m = re.search(r"(P\d+)",fn,re.I); return m.group(1).upper() if m else None

def find_files(d):
    m = {}
    for f in Path(d).rglob("*"):
        if f.suffix.lower() in {".csv",".xlsx",".xls"}:
            p = pid_from(f.name)
            if p: m.setdefault(p, f)
    return m

def clean_stage(raw):
    if pd.isna(raw): return "Unknown"
    return STAGE_MAP.get(str(raw).strip().lower(), str(raw).strip())

def load_facial(path):
    df = pd.read_csv(path) if path.suffix==".csv" else pd.read_excel(path)
    df.columns = df.columns.str.strip()
    if "cleaned time" in df.columns: df = df.rename(columns={"cleaned time":"time_s"})
    elif "timestamp"  in df.columns: df = df.rename(columns={"timestamp":"time_s"})
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
    df = df.rename(columns={v:k for k,v in col_map.items()})
    for col in ["start","end"]:
        if col not in df.columns: continue
        if pd.api.types.is_timedelta64_dtype(df[col]): df[col] = df[col].dt.total_seconds()
        elif df[col].dtype == object:
            def _s(v):
                try: p = str(v).split(":"); return sum(float(x)*60**(len(p)-i-1) for i,x in enumerate(p))
                except: return np.nan
            df[col] = df[col].apply(_s)
    if "stage" not in df.columns: return pd.DataFrame()
    df["stage"] = df["stage"].apply(clean_stage)
    df = df[df["stage"]!="SAM"].copy()
    return df.dropna(subset=["start","end","stage"]).reset_index(drop=True)

def load_exam(path):
    df = pd.read_excel(path); df.columns = df.columns.str.strip()
    scores = {}
    for _, row in df.iterrows():
        m = re.search(r"P(\d+)", str(row.iloc[0]).upper())
        if not m: continue
        try: scores[f"P{m.group(1)}"] = float(row.iloc[1])/(float(row.iloc[2]) if len(row)>2 else 26.)
        except: pass
    return scores

def load_sam(path):
    df = pd.read_excel(path, header=None)
    hrow = 0
    for i,row in df.iterrows():
        if any("participant" in str(v).lower() for v in row.values): hrow=i; break
    df.columns = df.iloc[hrow]; df = df.iloc[hrow+1:].reset_index(drop=True)
    out = {}
    for _, row in df.iterrows():
        m = re.search(r"P(\d+)", str(row.iloc[0]).upper())
        if not m: continue
        pid = f"P{m.group(1)}"; vals = row.iloc[1:].tolist()
        def _f(v):
            try: return float(v) if not pd.isna(v) else np.nan
            except: return np.nan
        out[pid] = {"valence":[_f(v) for v in vals[0::2]],
                    "arousal": [_f(v) for v in vals[1::2]]}
    return out

def compute_emotions(df):
    for em, aus in EMOTION_AU.items():
        avail = [a for a in aus if a in df.columns]
        df[em] = df[avail].sum(axis=1) if avail else 0.0
    return df

def align_stages(facial, stage):
    facial = facial.copy(); facial["stage"] = "Unknown"
    for _, row in stage.iterrows():
        mask = (facial["time_s"]>=row["start"]) & (facial["time_s"]<row["end"])
        facial.loc[mask,"stage"] = row["stage"]
    return facial

def sg(series, window=75, poly=3):
    n = len(series); w = min(window, n//2*2+1)
    if w%2==0: w+=1; w = max(5,w)
    return savgol_filter(series,w,poly) if n>=w else series

def _norm(v,lo,hi): return float(np.clip((v-lo)/(hi-lo+1e-9),0,1))

# ============================================================================
# EI FEATURE EXTRACTION
# ============================================================================

def extract_features(fused, stage_df, sam_valence, sam_arousal):
    total = stage_df["end"].max() - stage_df["start"].min() + 1e-9
    b = stage_df[stage_df["stage"].isin(BLOCK_STAGES)]
    p = stage_df[stage_df["stage"].isin(PRODUCTIVE_STAGES)]
    block_score      = _norm(1-(b["end"]-b["start"]).sum()/total, 0.8, 1.0)
    productive_ratio = _norm((p["end"]-p["start"]).sum()/total, 0.3, 0.7)
    def _fill(lst):
        arr = np.array(lst, dtype=float)
        if np.all(np.isnan(arr)): return np.full(len(arr),3.)
        arr[np.isnan(arr)] = np.nanmean(arr); return arr
    v = _fill(sam_valence)
    slope,*_ = np.polyfit(range(len(v)),v,1) if len(v)>1 else [0]
    return {
        "c3_block_score":      block_score,
        "c3_productive_ratio": productive_ratio,
        "c4_trend":            _norm(slope,-0.5,0.5),
        "c4_mean_valence":     _norm(v.mean(),1,5),
    }

def ei_score(feats): return round(sum(EI_WEIGHTS[i]*feats[f] for i,f in enumerate(EI_FEATURES))*100,1)

# ============================================================================
# 1. LOOCV-VALIDATED EI SCORE
# ============================================================================

def compute_loocv_ei(all_features: dict, exam_scores: dict) -> dict:
    """
    For each participant, compute EI score using weights derived from OTHER 41.
    Weights are the normalised Pearson correlations of each feature with exam
    outcome, computed only on the training fold.
    This is the methodologically clean version of the composite score.
    """
    pids = sorted(all_features.keys(), key=lambda p: int(re.search(r"\d+",p).group()))
    loocv_scores = {}

    for held_out in pids:
        train_pids = [p for p in pids if p != held_out]
        train_fm   = pd.DataFrame([all_features[p] for p in train_pids],
                                   index=train_pids)
        train_y    = np.array([exam_scores[p] for p in train_pids])

        # Compute per-feature Pearson r on training fold
        weights = {}
        for feat in EI_FEATURES:
            r, _ = pearsonr(train_fm[feat].values, train_y)
            weights[feat] = max(r, 0.0)   # clip negative (feature works in +ve direction)

        total_w = sum(weights.values()) + 1e-9
        weights = {f: w/total_w for f, w in weights.items()}

        # Score held-out participant
        feats = all_features[held_out]
        score = sum(weights[f]*feats[f] for f in EI_FEATURES) * 100
        loocv_scores[held_out] = round(float(score), 1)

    return loocv_scores

# ============================================================================
# 2. PHASE PORTRAIT — STRESS vs HAPPINESS 2D ATTRACTOR
# ============================================================================

def compute_phase_portrait_metrics(fused: pd.DataFrame) -> dict:
    """
    Analyse the 2D phase portrait (Stress, Happiness).
    Metrics:
      - attractor_area: convex hull area — how wide the emotional space is
      - centroid_stress: average position in stress dimension
      - centroid_happy:  average position in happiness dimension
      - trajectory_length: total arc length — how much the state moves
      - return_rate: fraction of time spent near the centroid (stability)
    Reference: Takens (1981); Marwan et al. (2007).
    """
    s = sg(fused["Stress"].values).astype(float)
    h = sg(fused["Happiness"].values).astype(float)

    # Convex hull area via shoelace (subsample to 1000 points)
    idx = np.linspace(0, len(s)-1, min(1000, len(s)), dtype=int)
    pts = np.stack([s[idx], h[idx]], axis=1)

    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        area = float(hull.volume)   # in 2D, volume = area
    except Exception:
        area = 0.0

    centroid_s = float(s.mean())
    centroid_h = float(h.mean())

    # Total trajectory length (L2 distance between consecutive frames)
    ds = np.diff(s); dh = np.diff(h)
    traj_len = float(np.sqrt(ds**2 + dh**2).sum())
    # Normalise by session length
    traj_len_norm = traj_len / max(len(s), 1)

    # Return rate: fraction of frames within 0.5 units of centroid
    dist_to_centroid = np.sqrt((s-centroid_s)**2 + (h-centroid_h)**2)
    return_rate = float((dist_to_centroid < 0.5).mean())

    return {
        "pp_attractor_area":   _norm(area, 0, 20),
        "pp_centroid_stress":  centroid_s,
        "pp_centroid_happy":   centroid_h,
        "pp_traj_length":      _norm(traj_len_norm, 0, 0.5),
        "pp_return_rate":      return_rate,   # already 0-1
    }

# ============================================================================
# 3. TEMPORAL SEGMENTATION — EARLY / MID / LATE
# ============================================================================

def compute_temporal_segments(fused: pd.DataFrame) -> dict:
    """
    Split the session into thirds (0-10, 10-20, 20-30 min).
    For each segment: mean Stress, mean Happiness, stage distribution.
    Key metric: stress_slope (does stress go down over the session?)
    and happiness_slope (does happiness go up?).
    """
    t_max = fused["time_s"].max()
    t1    = t_max / 3
    t2    = 2 * t_max / 3

    segs = {
        "early": fused[fused["time_s"] <  t1],
        "mid":   fused[(fused["time_s"] >= t1) & (fused["time_s"] < t2)],
        "late":  fused[fused["time_s"] >= t2],
    }

    stress_by_seg  = {k: v["Stress"].mean()    if len(v) else 0.0 for k, v in segs.items()}
    happy_by_seg   = {k: v["Happiness"].mean() if len(v) else 0.0 for k, v in segs.items()}

    # Slope of stress across thirds (negative = improving)
    stress_vals = [stress_by_seg["early"], stress_by_seg["mid"], stress_by_seg["late"]]
    happy_vals  = [happy_by_seg["early"],  happy_by_seg["mid"],  happy_by_seg["late"]]

    stress_slope = stress_vals[2] - stress_vals[0]   # negative = stress decreasing = better
    happy_slope  = happy_vals[2]  - happy_vals[0]    # positive = happiness increasing = better

    # Block proportion in each segment
    block_pct = {}
    for seg_name, seg_df in segs.items():
        block_pct[seg_name] = float(
            seg_df["stage"].isin(BLOCK_STAGES).mean()
        ) if len(seg_df) else 0.0

    return {
        "seg_stress_early":  float(stress_by_seg["early"]),
        "seg_stress_mid":    float(stress_by_seg["mid"]),
        "seg_stress_late":   float(stress_by_seg["late"]),
        "seg_happy_early":   float(happy_by_seg["early"]),
        "seg_happy_mid":     float(happy_by_seg["mid"]),
        "seg_happy_late":    float(happy_by_seg["late"]),
        "seg_stress_slope":  float(stress_slope),
        "seg_happy_slope":   float(happy_slope),
        "seg_block_early":   block_pct["early"],
        "seg_block_late":    block_pct["late"],
    }

# ============================================================================
# 4. EMOTION STATE MACHINE — TRANSITION MATRIX
# ============================================================================

def discretise_state(stress_val: float, happy_val: float) -> int:
    if   stress_val >= 1.0:              return STATE_STRESSED
    elif happy_val  >= 0.5:              return STATE_HAPPY
    elif stress_val < 0.5 and happy_val < 0.5: return STATE_CALM
    else:                                return STATE_MIXED

def compute_state_machine(fused: pd.DataFrame) -> dict:
    """
    Discretise each frame into one of 4 emotional states.
    Compute 4×4 transition matrix.
    Key metrics: P(Stressed→Calm), P(Calm→Stressed), stability (P(s→s)).
    """
    s = fused["Stress"].values
    h = fused["Happiness"].values
    # Smooth before discretising to reduce noise-driven transitions
    s = sg(s); h = sg(h)

    states = np.array([discretise_state(si, hi) for si, hi in zip(s, h)])

    # 4×4 transition count matrix
    n_states = 4
    trans    = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(states)-1):
        trans[states[i], states[i+1]] += 1

    # Row-normalise to get probabilities
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_prob = np.divide(trans, row_sums, where=row_sums>0)

    # Key metrics
    p_stressed_to_calm  = float(trans_prob[STATE_STRESSED, STATE_CALM])
    p_calm_to_stressed  = float(trans_prob[STATE_CALM,     STATE_STRESSED])
    stability           = float(np.diag(trans_prob).mean())  # avg self-transition

    # State occupancy
    n = len(states)
    occupancy = {STATE_NAMES[i]: float((states==i).sum()/n) for i in range(n_states)}

    return {
        "sm_p_stressed_to_calm": p_stressed_to_calm,
        "sm_p_calm_to_stressed": p_calm_to_stressed,
        "sm_stability":          stability,
        "sm_calm_pct":           occupancy["Calm"],
        "sm_stressed_pct":       occupancy["Stressed"],
        "sm_happy_pct":          occupancy["Happy"],
        "sm_trans_matrix":       trans_prob,   # kept for plotting only
    }

# ============================================================================
# 5. BLOCK-TRIGGERED RESPONSE (event-locked analysis)
# ============================================================================

def compute_block_response(fused: pd.DataFrame,
                            stage_df: pd.DataFrame) -> dict:
    """
    For each Block episode, extract Stress in a [-30s, +60s] window
    around the block entry and exit.
    Compute: mean stress before entry, peak stress in block,
    stress at exit, and time to 80% recovery.
    Reference: Luck (2014) ERP Technique.
    """
    sorted_s  = stage_df.sort_values("start")
    pre_stress, peak_stress, exit_stress, recovery_times = [], [], [], []

    for _, row in sorted_s[sorted_s["stage"].isin(BLOCK_STAGES)].iterrows():
        t_enter = row["start"]
        t_exit  = row["end"]

        # Pre-block: 30s before entry
        pre = fused[(fused["time_s"] >= t_enter-30) &
                    (fused["time_s"] <  t_enter)]["Stress"]
        # In-block
        inb = fused[(fused["time_s"] >= t_enter) &
                    (fused["time_s"] <  t_exit)]["Stress"]
        # Post-block: 60s after exit
        post = fused[(fused["time_s"] >= t_exit) &
                     (fused["time_s"] <  t_exit+60)]["Stress"]

        if len(pre) < 5 or len(inb) < 5: continue

        pre_m  = float(pre.mean())
        peak_m = float(inb.max())
        exit_m = float(post.iloc[:10].mean()) if len(post) >= 10 \
                 else float(post.mean()) if len(post) else peak_m
        pre_stress.append(pre_m)
        peak_stress.append(peak_m)
        exit_stress.append(exit_m)

        # Recovery time: frames until stress drops below 80% of peak
        threshold = peak_m * 0.8
        recovered = False
        for j, stress_val in enumerate(post.values):
            if stress_val <= threshold:
                recovery_times.append(j / FPS)   # seconds
                recovered = True
                break
        if not recovered:
            recovery_times.append(60.0)   # did not recover within window

    return {
        "btn_mean_pre_stress":    float(np.mean(pre_stress))    if pre_stress    else 0.0,
        "btn_mean_peak_stress":   float(np.mean(peak_stress))   if peak_stress   else 0.0,
        "btn_mean_exit_stress":   float(np.mean(exit_stress))   if exit_stress   else 0.0,
        "btn_mean_recovery_time": float(np.mean(recovery_times)) if recovery_times else 30.0,
        "btn_std_recovery_time":  float(np.std(recovery_times)) if len(recovery_times)>1 else 0.0,
        "btn_n_blocks":           len(pre_stress),
        "btn_pre_stress_list":    pre_stress,    # kept for plotting
        "btn_recovery_list":      recovery_times,
    }

# ============================================================================
# 6. CROSS-LAGGED CORRELATION — does stress PREDICT block?
# ============================================================================

def compute_crosslag(fused: pd.DataFrame, max_lag_s: float = 30.0) -> dict:
    """
    Compute cross-correlation between Stress and a Block indicator
    at lags from -30s to +30s.
    Positive lag = stress precedes block (anticipatory stress).
    Negative lag = stress follows block (reactive stress).
    Reference: Granger (1969).
    """
    stress      = sg(fused["Stress"].values)
    block_ind   = fused["stage"].isin(BLOCK_STAGES).values.astype(float)

    # Normalise
    s_norm = (stress    - stress.mean())    / (stress.std()    + 1e-9)
    b_norm = (block_ind - block_ind.mean()) / (block_ind.std() + 1e-9)

    n        = len(stress)
    max_lag  = int(max_lag_s * FPS)
    lags     = np.arange(-max_lag, max_lag+1)
    xcorr    = np.array([
        np.correlate(s_norm[max(0,-lag):n-max(0,lag)],
                     b_norm[max(0, lag):n-max(0,-lag)])[0] /
        max(1, n - abs(lag))
        for lag in lags
    ])

    lag_s        = lags / FPS
    peak_lag_idx = int(np.argmax(xcorr))
    peak_lag_sec = float(lag_s[peak_lag_idx])
    peak_xcorr   = float(xcorr[peak_lag_idx])

    # Anticipatory index: mean xcorr at positive lags (stress before block)
    pos_lags      = xcorr[lags > 0]
    anticipatory  = float(pos_lags.mean()) if len(pos_lags) else 0.0

    return {
        "xl_peak_lag_sec":    peak_lag_sec,
        "xl_peak_xcorr":      peak_xcorr,
        "xl_anticipatory":    anticipatory,
        "xl_lag_axis":        lag_s,       # for plotting
        "xl_xcorr_values":    xcorr,
    }

# ============================================================================
# 7. REGULATION SPEED DISTRIBUTION
# ============================================================================

def compute_regulation_speed(fused: pd.DataFrame) -> dict:
    """
    For every stress peak (local maximum > 1.0 AU), measure how many
    seconds it takes for stress to drop by 20% from the peak.
    Returns summary statistics of the distribution.
    Fast regulation = low mean, low std.
    """
    stress     = sg(fused["Stress"].values)
    times      = fused["time_s"].values

    # Find local maxima using sliding window comparison
    from scipy.signal import argrelmax
    peak_idxs  = argrelmax(stress, order=FPS*2)[0]   # min 2s between peaks
    peak_idxs  = peak_idxs[stress[peak_idxs] > 1.0]  # only significant peaks

    recovery_s = []
    for pk in peak_idxs:
        threshold = stress[pk] * 0.8
        for j in range(pk, min(pk + FPS*60, len(stress))):
            if stress[j] <= threshold:
                recovery_s.append((times[j] - times[pk]))
                break
        else:
            recovery_s.append(60.0)   # did not recover within 60s

    if not recovery_s:
        return {
            "rs_mean_s": 30.0, "rs_median_s": 30.0,
            "rs_std_s": 0.0,   "rs_n_peaks":  0,
            "rs_fast_pct": 0.0, "rs_values":  [],
        }

    arr = np.array(recovery_s)
    return {
        "rs_mean_s":   float(arr.mean()),
        "rs_median_s": float(np.median(arr)),
        "rs_std_s":    float(arr.std()),
        "rs_n_peaks":  len(arr),
        "rs_fast_pct": float((arr < 5.0).mean()),  # % recovered in < 5s
        "rs_values":   recovery_s,
    }

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
        key=lambda p: int(re.search(r"\d+",p).group())
    )
    log.info("Processing %d participants ...", len(common))

    participant_data  = {}
    base_features     = {}
    temporal_features = {}

    for pid in common:
        try:
            facial = load_facial(facial_files[pid])
            stage  = load_stage(stage_files[pid])
            if facial.empty or stage.empty: continue
            facial = compute_emotions(facial)
            fused  = align_stages(facial, stage)
            fused  = fused[np.isfinite(fused["time_s"].values)].copy()
            if len(fused) < 100: continue

            sam    = sam_data[pid]
            bf     = extract_features(fused, stage, sam["valence"], sam["arousal"])
            base_features[pid] = bf

            # All temporal metrics
            pp  = compute_phase_portrait_metrics(fused)
            ts  = compute_temporal_segments(fused)
            sm  = compute_state_machine(fused)
            btn = compute_block_response(fused, stage)
            xl  = compute_crosslag(fused)
            rs  = compute_regulation_speed(fused)

            # Store scalar metrics only in feature dict
            tf = {**pp, **ts}
            tf.update({k:v for k,v in sm.items() if k != "sm_trans_matrix"})
            tf.update({k:v for k,v in btn.items()
                       if k not in ("btn_pre_stress_list","btn_recovery_list")})
            tf.update({k:v for k,v in xl.items()
                       if k not in ("xl_lag_axis","xl_xcorr_values")})
            tf.update({k:v for k,v in rs.items() if k != "rs_values"})
            temporal_features[pid] = tf

            participant_data[pid] = {
                "fused":    fused,
                "stage":    stage,
                "ei_base":  ei_score(bf),
                "exam":     exam_data[pid],
                "sm_trans": sm["sm_trans_matrix"],
                "xl_lag":   xl["xl_lag_axis"],
                "xl_xcorr": xl["xl_xcorr_values"],
                "rs_vals":  rs["rs_values"],
                "btn_pre":  btn["btn_pre_stress_list"],
                "btn_rec":  btn["btn_recovery_list"],
            }
            log.info("[%s] OK  EI=%.1f  Exam=%d/26", pid,
                     ei_score(bf), round(exam_data[pid]*26))

        except Exception as e:
            log.error("[%s] failed: %s", pid, e)

    return participant_data, base_features, temporal_features, exam_data

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_phase_portraits(participant_data, exam_data, out_dir):
    """
    Grid of phase portraits (Stress vs Happiness) for all participants,
    coloured by exam performance. Shows how high vs low scorers
    occupy different regions of emotional space.
    """
    pids = sorted(participant_data.keys(),
                  key=lambda p: exam_data[p], reverse=True)
    n    = len(pids)
    ncols = 7
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*2.2, nrows*2.2),
                             squeeze=False)
    cmap = plt.cm.RdYlGn
    exam_vals = np.array([exam_data[p] for p in pids])
    norm_exam = (exam_vals - exam_vals.min()) / (exam_vals.ptp() + 1e-9)

    for idx, pid in enumerate(pids):
        ax   = axes[idx//ncols][idx%ncols]
        d    = participant_data[pid]
        fused = d["fused"]
        s = sg(fused["Stress"].values)
        h = sg(fused["Happiness"].values)
        t = np.linspace(0, 1, len(s))
        col = cmap(norm_exam[idx])

        ax.scatter(s, h, c=t, cmap="viridis", s=0.5, alpha=0.4, linewidths=0)
        # Mark centroid
        ax.plot(s.mean(), h.mean(), "o", color=col,
                markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                zorder=5)
        ax.set_xlim(0, 6); ax.set_ylim(0, 3)
        ax.set_title(f"{pid}\n{exam_data[pid]*26:.0f}/26",
                     fontsize=7, fontweight="bold",
                     color="darkgreen" if exam_data[pid]>0.55 else
                           "darkred"   if exam_data[pid]<0.35 else "black")
        ax.tick_params(labelsize=5)
        if idx % ncols == 0: ax.set_ylabel("Happiness", fontsize=6)
        if idx // ncols == nrows-1: ax.set_xlabel("Stress", fontsize=6)

    for idx in range(n, nrows*ncols):
        axes[idx//ncols][idx%ncols].set_visible(False)

    plt.suptitle("Emotional Phase Portraits — Stress vs Happiness\n"
                 "(dot = centroid; viridis colour = time; "
                 "title colour = exam performance)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir/"phase_portraits_grid.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig); plt.clf(); plt.cla()
    log.info("Phase portrait grid saved.")


def plot_temporal_segments_heatmap(temp_features, exam_data, out_dir):
    """
    Heatmap: rows=participants (sorted by exam), cols=temporal segment metrics.
    Makes it visible how stress/happiness evolve across the session
    and how that pattern differs between high and low scorers.
    """
    pids = sorted(temp_features.keys(),
                  key=lambda p: exam_data[p], reverse=True)
    cols = ["seg_stress_early","seg_stress_mid","seg_stress_late",
            "seg_happy_early", "seg_happy_mid", "seg_happy_late",
            "seg_stress_slope","seg_happy_slope",
            "seg_block_early", "seg_block_late"]
    labels = ["Stress\nEarly","Stress\nMid","Stress\nLate",
              "Happy\nEarly","Happy\nMid","Happy\nLate",
              "Stress\nSlope","Happy\nSlope",
              "Block%\nEarly","Block%\nLate"]

    data = np.array([[temp_features[p].get(c,0) for c in cols] for p in pids])
    # Z-score each column for better visual contrast
    z = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, max(8, len(pids)*0.28)))
    im = ax.imshow(z.T, aspect="auto", cmap="RdYlGn_r",
                   vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels(
        [f"{p}\n({exam_data[p]*26:.0f})" for p in pids],
        fontsize=5.5, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Z-score", fraction=0.015)
    ax.set_title("Temporal Segment Heatmap (sorted by exam score, high→low)\n"
                 "Green = low/decreasing, Red = high/increasing",
                 fontweight="bold", fontsize=10)
    # Mark quartile boundaries
    n = len(pids)
    q1_line = int(n*0.25) - 0.5
    q3_line = int(n*0.75) - 0.5
    ax.axvline(q1_line, color="white", linewidth=1.5, linestyle="--")
    ax.axvline(q3_line, color="white", linewidth=1.5, linestyle="--")
    ax.text(q1_line/2, -1.2, "TOP 25%", ha="center", fontsize=7,
            color="white", fontweight="bold")
    ax.text((q1_line+q3_line)/2, -1.2, "MID 50%", ha="center",
            fontsize=7, color="white", fontweight="bold")
    ax.text((q3_line+n)/2, -1.2, "BOTTOM 25%", ha="center",
            fontsize=7, color="white", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir/"temporal_segment_heatmap.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Temporal segment heatmap saved.")


def plot_block_triggered_responses(participant_data, exam_data, out_dir):
    """
    Event-locked analysis: mean Stress ±1std around Block entry,
    separately for high vs low scorers.
    X-axis: seconds relative to block entry (0 = entry, positive = after entry).
    """
    pids     = list(participant_data.keys())
    y_arr    = np.array([exam_data[p] for p in pids])
    q3, q1   = np.percentile(y_arr, 75), np.percentile(y_arr, 25)
    pids_hi  = [p for p,y in zip(pids,y_arr) if y >= q3]
    pids_lo  = [p for p,y in zip(pids,y_arr) if y <= q1]

    window_before = 30   # seconds
    window_after  = 60
    n_before = window_before * FPS
    n_after  = window_after  * FPS
    total_w  = n_before + n_after

    def _collect_traces(pid_list):
        traces = []
        for pid in pid_list:
            d     = participant_data[pid]
            fused = d["fused"]
            stage = d["stage"]
            stress = sg(fused["Stress"].values)
            times  = fused["time_s"].values
            for _, row in stage[stage["stage"].isin(BLOCK_STAGES)].iterrows():
                t0   = row["start"]
                idx0 = np.searchsorted(times, t0)
                i1   = max(0, idx0 - n_before)
                i2   = min(len(stress), idx0 + n_after)
                seg  = stress[i1:i2]
                if len(seg) < total_w * 0.5: continue
                # Pad / trim to fixed length
                if len(seg) < total_w:
                    seg = np.pad(seg, (0, total_w-len(seg)),
                                 constant_values=np.nan)
                seg = seg[:total_w]
                traces.append(seg)
        return np.array(traces) if traces else np.empty((0, total_w))

    hi_mat = _collect_traces(pids_hi)
    lo_mat = _collect_traces(pids_lo)

    t_axis = (np.arange(total_w) - n_before) / FPS   # seconds

    fig, ax = plt.subplots(figsize=(12, 5))
    for mat, color, label, n_p in [
        (hi_mat, "#1A5276", f"High scorers (n_p={len(pids_hi)},  n_blocks={len(hi_mat)})", len(pids_hi)),
        (lo_mat, "#922B21", f"Low scorers  (n_p={len(pids_lo)}, n_blocks={len(lo_mat)})", len(pids_lo)),
    ]:
        if len(mat) == 0: continue
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat,  axis=0)
        ax.plot(t_axis, mean, color=color, linewidth=2.2, label=label)
        ax.fill_between(t_axis, mean-std, mean+std, color=color, alpha=0.15)

    ax.axvline(0, color="black", linewidth=1.5, linestyle="--",
               label="Block entry")
    ax.axvspan(-window_before, 0, facecolor="#FFF3CD", alpha=0.3,
               label="Pre-block period")
    ax.axvspan(0, window_after, facecolor="#FDECEA", alpha=0.3,
               label="In/post-block period")
    ax.set_xlabel("Seconds relative to Block entry (0 = entry)", fontsize=10)
    ax.set_ylabel("Mean Stress intensity (±1 std)", fontsize=10)
    ax.set_title("Block-Triggered Stress Response — High vs Low Scorers\n"
                 "(event-locked analysis; shading = ±1 std)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir/"block_triggered_response.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Block-triggered response plot saved.")


def plot_crosslag_comparison(participant_data, exam_data, out_dir):
    """
    Mean cross-correlation (Stress → Block) for high vs low scorers.
    Positive peak lag > 0 means stress PRECEDES block entry — anticipatory.
    """
    pids    = list(participant_data.keys())
    y_arr   = np.array([exam_data[p] for p in pids])
    q3, q1  = np.percentile(y_arr,75), np.percentile(y_arr,25)

    hi_xcorrs, lo_xcorrs = [], []
    for pid, y in zip(pids, y_arr):
        xl_lag   = participant_data[pid]["xl_lag"]
        xl_xcorr = participant_data[pid]["xl_xcorr"]
        if y >= q3: hi_xcorrs.append(xl_xcorr)
        elif y <= q1: lo_xcorrs.append(xl_xcorr)

    if not hi_xcorrs or not lo_xcorrs:
        return

    lag_ax  = participant_data[pids[0]]["xl_lag"]
    hi_mean = np.mean(hi_xcorrs, axis=0)
    lo_mean = np.mean(lo_xcorrs, axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lag_ax, hi_mean, color="#1A5276", linewidth=2.2,
            label=f"High scorers (n={len(hi_xcorrs)})")
    ax.plot(lag_ax, lo_mean, color="#922B21", linewidth=2.2,
            linestyle="--", label=f"Low scorers (n={len(lo_xcorrs)})")
    ax.axvline(0, color="black", linewidth=1.0, linestyle=":")
    ax.axhline(0, color="gray",  linewidth=0.6)
    ax.fill_between(lag_ax, hi_mean, 0, where=lag_ax>0,
                    color="#1A5276", alpha=0.12,
                    label="Stress→Block (anticipatory zone)")
    ax.set_xlabel("Lag (seconds): positive = Stress leads Block", fontsize=10)
    ax.set_ylabel("Cross-correlation", fontsize=10)
    ax.set_title("Cross-Lagged Correlation: Stress → Block Indicator\n"
                 "Positive lag = stress anticipates getting stuck",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir/"crosslag_stress_block.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Cross-lag plot saved.")


def plot_regulation_speed_distribution(participant_data, exam_data, out_dir):
    """
    Violin + strip plot of stress regulation speed (seconds to recover),
    split by high vs low scorers.
    """
    pids   = list(participant_data.keys())
    y_arr  = np.array([exam_data[p] for p in pids])
    q3, q1 = np.percentile(y_arr,75), np.percentile(y_arr,25)

    hi_vals, lo_vals = [], []
    for pid, y in zip(pids, y_arr):
        vals = participant_data[pid]["rs_vals"]
        if y >= q3: hi_vals.extend(vals)
        elif y<=q1: lo_vals.extend(vals)

    if not hi_vals or not lo_vals:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot([hi_vals, lo_vals], positions=[1,2],
                           showmedians=True, showextrema=True)
    colors_v = ["#1A5276","#922B21"]
    for body, c in zip(parts["bodies"], colors_v):
        body.set_facecolor(c); body.set_alpha(0.6)

    # Strip points
    np.random.seed(42)
    for vals, pos, c in [(hi_vals,1,"#1A5276"),(lo_vals,2,"#922B21")]:
        jitter = np.random.uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), pos)+jitter, vals,
                   color=c, alpha=0.5, s=18, zorder=3)

    ax.axhline(5, color="gray", linewidth=0.8, linestyle="--",
               alpha=0.6, label="5-second threshold")
    ax.set_xticks([1,2])
    ax.set_xticklabels([f"High scorers\n(n_peaks={len(hi_vals)})",
                        f"Low scorers\n(n_peaks={len(lo_vals)})"], fontsize=10)
    ax.set_ylabel("Seconds to recover (stress → 80% of peak)", fontsize=10)
    ax.set_title("Regulation Speed Distribution\n"
                 "Seconds until stress drops to 80% of peak after each spike",
                 fontweight="bold", fontsize=11)

    # Statistical annotation
    u, p = mannwhitneyu(hi_vals, lo_vals, alternative="two-sided")
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "NS"
    y_top = max(max(hi_vals), max(lo_vals)) * 1.05
    ax.plot([1,2],[y_top,y_top], "k-", linewidth=1)
    ax.text(1.5, y_top*1.01, f"Mann-Whitney {sig}\n(p={p:.4f})",
            ha="center", fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir/"regulation_speed_violin.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Regulation speed violin saved.")


def plot_state_machine_comparison(participant_data, exam_data, out_dir):
    """
    Average transition matrices for high vs low scorers.
    Shows whether high scorers transition differently between emotional states.
    """
    pids   = list(participant_data.keys())
    y_arr  = np.array([exam_data[p] for p in pids])
    q3, q1 = np.percentile(y_arr,75), np.percentile(y_arr,25)

    hi_mats, lo_mats = [], []
    for pid, y in zip(pids, y_arr):
        mat = participant_data[pid]["sm_trans"]
        if y >= q3: hi_mats.append(mat)
        elif y<=q1: lo_mats.append(mat)

    if not hi_mats or not lo_mats: return

    hi_mean = np.mean(hi_mats, axis=0)
    lo_mean = np.mean(lo_mats, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mat, title in [
        (axes[0], hi_mean, f"High scorers (n={len(hi_mats)})"),
        (axes[1], lo_mean, f"Low scorers  (n={len(lo_mats)})"),
        (axes[2], hi_mean - lo_mean, "Difference (High − Low)"),
    ]:
        cmap = "Blues" if "Diff" not in title else "RdBu"
        vmax = 1.0 if "Diff" not in title else 0.3
        vmin = 0.0 if "Diff" not in title else -0.3
        im   = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_xticklabels(STATE_NAMES, fontsize=9)
        ax.set_yticks(range(4)); ax.set_yticklabels(STATE_NAMES, fontsize=9)
        ax.set_xlabel("To state", fontsize=9)
        ax.set_ylabel("From state", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if mat[i,j] > 0.6 else "black")
    fig.suptitle("Emotional State Transition Matrices",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir/"state_transition_matrices.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    log.info("State transition matrices saved.")


def plot_loocv_ei_scatter(loocv_scores, exam_data, out_dir):
    """EI score (LOOCV-validated) vs exam performance."""
    pids = sorted(loocv_scores.keys(),
                  key=lambda p: int(re.search(r"\d+",p).group()))
    ei   = np.array([loocv_scores[p] for p in pids])
    ex   = np.array([exam_data[p]*100 for p in pids])
    r, p = pearsonr(ei, ex)
    sr,sp= spearmanr(ei,ex)

    fig, ax = plt.subplots(figsize=(9, 7))
    med_ei = np.median(ei); med_ex = np.median(ex)
    ax.axhline(med_ex, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(med_ei, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    colors = []
    for e, x in zip(ei, ex):
        if   e>=med_ei and x>=med_ex: colors.append("#1A5276")
        elif e< med_ei and x< med_ex: colors.append("#922B21")
        else:                          colors.append("#7D6608")
    ax.scatter(ei, ex, c=colors, s=80, edgecolors="black", linewidths=0.5, zorder=3)
    for pid, e, x, c in zip(pids, ei, ex, colors):
        ax.annotate(pid, (e, x), textcoords="offset points",
                    xytext=(5,3), fontsize=6.5, color=c)
    m, b = np.polyfit(ei, ex, 1)
    xl   = np.linspace(ei.min(), ei.max(), 100)
    ax.plot(xl, m*xl+b, "k--", linewidth=1.5)
    ax.set_xlabel("LOOCV-validated EI Score (0–100)", fontsize=11)
    ax.set_ylabel("Exam Score (%)", fontsize=11)
    ax.set_title(f"LOOCV-Validated EI Score vs Exam Performance\n"
                 f"Pearson r={r:.3f} (p={p:.6f})  |  Spearman r={sr:.3f}",
                 fontweight="bold", fontsize=11)
    hi_p = mpatches.Patch(color="#1A5276", label="High EI + High Exam")
    lo_p = mpatches.Patch(color="#922B21", label="Low EI  + Low Exam")
    mx_p = mpatches.Patch(color="#7D6608", label="Inconsistent")
    ax.legend(handles=[hi_p,lo_p,mx_p], fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir/"loocv_ei_vs_exam.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("LOOCV EI scatter saved.  r=%.4f p=%.6f", r, p)
    return r, p, sr, sp

# ============================================================================
# STATISTICAL ANALYSIS ON TEMPORAL FEATURES
# ============================================================================

def analyse_temporal_features(temp_features, exam_data, out_dir):
    pids  = sorted(temp_features.keys(),
                   key=lambda p: int(re.search(r"\d+",p).group()))
    y     = np.array([exam_data[p] for p in pids])
    fm    = pd.DataFrame([temp_features[p] for p in pids], index=pids)
    q1, q3 = np.percentile(y,25), np.percentile(y,75)
    hi = y>=q3; lo = y<=q1

    rows = []
    for col in fm.columns:
        try:
            r, p   = pearsonr(fm[col].values, y)
            u, pu  = mannwhitneyu(fm.loc[hi,col].values,
                                  fm.loc[lo,col].values,
                                  alternative="two-sided")
            pool = np.sqrt((fm.loc[hi,col].std()**2 +
                            fm.loc[lo,col].std()**2)/2 + 1e-9)
            d = (fm.loc[hi,col].mean() - fm.loc[lo,col].mean()) / pool
            rows.append({"Feature":col,"Pearson r":round(r,4),
                         "Pearson p":round(p,4),
                         "Cohen's d":round(d,3),
                         "Mann-Whitney p":round(pu,4),
                         "Significant":(
                             "YES **" if pu<0.01 else
                             "YES *"  if pu<0.05 else "no")})
        except: pass

    df = pd.DataFrame(rows).sort_values("Cohen's d", key=abs, ascending=False)
    df.to_csv(out_dir/"temporal_feature_stats.csv", index=False)
    log.info("Top temporal features by effect size:")
    for _, row in df[df["Significant"]!="no"].head(10).iterrows():
        log.info("  %-30s d=%+.2f  r=%+.3f  p=%.4f",
                 row["Feature"], row["Cohen's d"],
                 row["Pearson r"], row["Pearson p"])
    return df

# ============================================================================
# MAIN
# ============================================================================

def main(data_dir: str):
    data_path = Path(data_dir)
    out_dir   = data_path / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 55)
    log.info("Processing all participants ...")
    log.info("=" * 55)
    participant_data, base_features, temporal_features, exam_data = \
        process_all(data_path)

    if len(participant_data) < 5:
        log.error("Too few participants. Stopping."); return

    y_arr = np.array([exam_data[p] for p in participant_data])
    log.info("Participants processed: %d", len(participant_data))

    # ── 1. LOOCV-validated EI score ───────────────────────────────────────
    log.info("=" * 55)
    log.info("STEP 1: LOOCV-validated EI score")
    log.info("=" * 55)
    loocv_scores = compute_loocv_ei(base_features, exam_data)
    loocv_df     = pd.DataFrame({"LOOCV_EI": loocv_scores,
                                  "Exam_pct": {p:v*100 for p,v in exam_data.items()
                                               if p in loocv_scores}})
    loocv_df.to_csv(out_dir/"loocv_ei_scores.csv")
    r, p, sr, sp = plot_loocv_ei_scatter(loocv_scores, exam_data, out_dir)

    # ── 2. Phase portraits ────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("STEP 2: Phase portrait grid")
    log.info("=" * 55)
    plot_phase_portraits(participant_data, exam_data, out_dir)

    # ── 3. Temporal segment heatmap ───────────────────────────────────────
    log.info("=" * 55)
    log.info("STEP 3: Temporal segment heatmap")
    log.info("=" * 55)
    plot_temporal_segments_heatmap(temporal_features, exam_data, out_dir)

    # ── 4. State machine ──────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("STEP 4: State transition matrices")
    log.info("=" * 55)
    plot_state_machine_comparison(participant_data, exam_data, out_dir)

    # ── 5. Block-triggered response ───────────────────────────────────────
    log.info("=" * 55)
    log.info("STEP 5: Block-triggered response")
    log.info("=" * 55)
    plot_block_triggered_responses(participant_data, exam_data, out_dir)

    # ── 6. Cross-lag ──────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("STEP 6: Cross-lagged correlation")
    log.info("=" * 55)
    plot_crosslag_comparison(participant_data, exam_data, out_dir)

    # ── 7. Regulation speed ───────────────────────────────────────────────
    log.info("=" * 55)
    log.info("STEP 7: Regulation speed distribution")
    log.info("=" * 55)
    plot_regulation_speed_distribution(participant_data, exam_data, out_dir)

    # ── 8. Statistical analysis on temporal features ──────────────────────
    log.info("=" * 55)
    log.info("STEP 8: Statistical analysis of temporal features")
    log.info("=" * 55)
    temp_stats = analyse_temporal_features(temporal_features, exam_data, out_dir)

    # ── Summary report ────────────────────────────────────────────────────
    sig_temp = temp_stats[temp_stats["Significant"] != "no"]
    lines = [
        "=" * 68,
        "   EI ADVANCED TEMPORAL ANALYSIS — SUMMARY",
        "=" * 68,
        f"   Participants: {len(participant_data)}",
        "",
        "─" * 68,
        "   1. LOOCV-VALIDATED EI SCORE (methodologically clean)",
        "─" * 68,
        f"   Pearson  r = {r:.4f}  (p = {p:.6f})",
        f"   Spearman r = {sr:.4f}  (p = {sp:.6f})",
        "",
        "─" * 68,
        "   2. SIGNIFICANT TEMPORAL FEATURES (Mann-Whitney, p<0.05)",
        "─" * 68,
        f"   {'Feature':<32} {'d':>7} {'r':>7} {'p':>8}",
        "   " + "-"*58,
    ]
    for _, row in sig_temp.iterrows():
        d_val = row["Cohen's d"]
        lines.append(
            f"   {row['Feature']:<32} {d_val:>7.3f} "
            f"{row['Pearson r']:>7.3f} {row['Pearson p']:>8.4f}"
        )
    lines += [
        "",
        "─" * 68,
        "   OUTPUT FILES",
        "─" * 68,
        "   loocv_ei_scores.csv             — validated EI scores",
        "   loocv_ei_vs_exam.png            — LOOCV EI scatter",
        "   phase_portraits_grid.png        — 2D emotional space per participant",
        "   temporal_segment_heatmap.png    — early/mid/late emotion patterns",
        "   state_transition_matrices.png   — emotional state machine",
        "   block_triggered_response.png    — event-locked stress analysis",
        "   crosslag_stress_block.png       — does stress predict being stuck?",
        "   regulation_speed_violin.png     — recovery speed distribution",
        "   temporal_feature_stats.csv      — full statistics table",
        "=" * 68,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(out_dir/"temporal_summary.txt","w",encoding="utf-8") as f:
        f.write(report)
    log.info("All outputs → %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EI Advanced Temporal Analysis"
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    main(args.data_dir)
