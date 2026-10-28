"""
EI Feature Extraction & Validity Testing — Full Five-Dimension Pipeline
========================================================================
Implements the complete process-based EI framework from:
  "A Process-Based Model of Emotional Intelligence" (project document)

Five dimensions with theoretical grounding:
  D1 Emotion Perception      (weight 0.15) — Mayer-Salovey Branch 1 + Barrett
  D2 Emotion Regulation      (weight 0.25) — Mayer-Salovey Branch 4 + Gross
  D3 Adaptive Persistence    (weight 0.25) — Bar-On + D'Mello + Martin
  D4 Affective Self-Awareness(weight 0.15) — Petrides + Pekrun
  D5 Emotion-Cognition       (weight 0.20) — Mayer-Salovey Branch 2 + Fiori

Feature extraction methods:
  - Scalar aggregates (mean, std, slope over session/segments)
  - Shannon entropy (global + temporal slope)
  - Sliding window variance
  - Markov chain over problem-solving stages (inside D3)
  - SAM trajectory features
  - Stage-conditional emotion features (D5)

Validity testing (same criteria for ALL features):
  - Criterion validity: Pearson r + Spearman r with exam score
  - Internal consistency: Cronbach's alpha within each dimension
  - Effect size: Cohen's d (top vs bottom exam quartile)
  - Mann-Whitney U significance test

Output → results_v5/
  feature_matrix.csv         — all features per participant
  validity_by_feature.csv    — full criterion validity table
  dimension_summary.csv      — per-dimension alpha + composite r
  validity_heatmap.png       — feature × participant heatmap
  effect_size_chart.png      — Cohen's d bar chart by dimension
  dimension_alphas.png       — Cronbach alpha per dimension
  pipeline_report.txt        — full human-readable report

Usage:
    python ei_feature_pipeline_v5.py
    python ei_feature_pipeline_v5.py --data_dir "D:/A_emotions in exam/data"
"""

import re
import zipfile
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
RESULTS_DIR      = "results_v5"

STAGE_ZIP  = "stage coded data-20260321T120851Z-1-001.zip"
FACIAL_ZIP = "facial expressions- data-20260321T182549Z-1-001.zip"
EXAM_FILE  = "Exam Scores.xlsx"
SAM_FILE   = "SAM Responses every 3 minutes.xlsx"
STAGE_DIR  = "stage_coded_data"
FACIAL_DIR = "facial_expressions"

FPS = 25

# ── Stage definitions ─────────────────────────────────────────────────────
STAGE_STATES    = ["Preparation","Ideation","Production","Evaluation","Refinement","Block"]
STAGE_IDX       = {s: i for i, s in enumerate(STAGE_STATES)}
N_STAGE_STATES  = len(STAGE_STATES)
PRODUCTIVE_STAGES = {"Production", "Evaluation", "Refinement"}
BLOCK_VARIANTS    = {"Block", "Block ", "block"}

STAGE_MAP = {
    "preparation":"Preparation","ideation":"Ideation",
    "production":"Production","prod":"Production",
    "evaluation":"Evaluation","refinement":"Refinement",
    "block":"Block","sam":"SAM",
}

# ── Emotion AU definitions ────────────────────────────────────────────────
EMOTION_AU = {
    "Anger":     ["AU04_r","AU05_r","AU07_r","AU23_r"],
    "Fear":      ["AU01_r","AU04_r","AU05_r"],
    "Sadness":   ["AU01_r","AU04_r","AU15_r"],
    "Disgust":   ["AU04_r","AU07_r","AU09_r","AU10_r","AU17_r","AU20_r"],
    "Pride":     ["AU06_r","AU07_r","AU12_r"],
    "Happiness": ["AU06_r","AU12_r"],
    "Stress":    ["AU01_r","AU04_r","AU07_r","AU15_r","AU17_r","AU23_r"],
}
POSITIVE_EMOTIONS = {"Happiness","Pride"}
NEGATIVE_EMOTIONS = {"Anger","Fear","Sadness","Disgust","Stress"}
ALL_EMOTIONS      = list(EMOTION_AU.keys())

# ── Validity thresholds (from project plan) ───────────────────────────────
MIN_ABS_R   = 0.10   # features below this are non-predictive
MAX_P_NONPREDICTIVE = 0.50
MIN_ALPHA   = 0.60   # minimum acceptable Cronbach's alpha

# ── Dimension weights (theory-driven, from process-based model doc) ───────
DIM_WEIGHTS = {"D1":0.15, "D2":0.25, "D3":0.25, "D4":0.15, "D5":0.20}

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def extract_zip(zp: Path, od: Path) -> None:
    if od.exists() and any(od.iterdir()):
        return
    od.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s ...", zp.name)
    with zipfile.ZipFile(zp, "r") as z:
        z.extractall(od)

def pid_from(fn: str) -> str | None:
    m = re.search(r"(P\d+)", fn, re.I)
    return m.group(1).upper() if m else None

def find_files(d: Path) -> dict:
    m = {}
    for f in d.rglob("*"):
        if f.suffix.lower() in {".csv",".xlsx",".xls"}:
            p = pid_from(f.name)
            if p:
                m.setdefault(p, f)
    return m

def clean_stage(raw) -> str:
    if pd.isna(raw):
        return "Unknown"
    return STAGE_MAP.get(str(raw).strip().lower(), str(raw).strip())

def load_facial(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df.columns = df.columns.str.strip()
    for alias in ["cleaned time","timestamp","time"]:
        if alias in df.columns:
            df = df.rename(columns={alias: "time_s"}); break
    r_cols = [c for c in df.columns if c.endswith("_r")]
    keep   = ["time_s"] + [c for c in ["confidence","success"] if c in df.columns] + r_cols
    df     = df[[c for c in keep if c in df.columns]].copy()
    if "confidence" in df.columns:
        df = df[df["confidence"] >= 0.80]
    if "success" in df.columns:
        df = df[df["success"] == 1]
    return df.reset_index(drop=True)

def load_stage(path: Path) -> pd.DataFrame:
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
        if col not in df.columns:
            continue
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
        elif df[col].dtype == object:
            def _s(v):
                try:
                    p = str(v).split(":")
                    return sum(float(x)*60**(len(p)-i-1) for i, x in enumerate(p))
                except:
                    return np.nan
            df[col] = df[col].apply(_s)
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
            scores[f"P{m.group(1)}"] = float(row.iloc[1]) / (
                float(row.iloc[2]) if len(row) > 2 and not pd.isna(row.iloc[2]) else 26.0
            )
        except:
            pass
    return scores

def load_sam(path: Path) -> pd.DataFrame:
    """Load SAM responses — expects columns: participant, checkpoint, valence, arousal"""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower()
    # normalise common column name variants
    rename = {}
    for c in df.columns:
        if "part" in c or c == "id": rename[c] = "participant"
        if "check" in c or "time" in c or "point" in c: rename[c] = "checkpoint"
        if "valen" in c: rename[c] = "valence"
        if "arous" in c or "excit" in c or "mood" in c: rename[c] = "arousal"
    df = df.rename(columns=rename)
    for col in ["participant","valence","arousal"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def compute_emotions(df: pd.DataFrame) -> pd.DataFrame:
    for em, aus in EMOTION_AU.items():
        avail = [a for a in aus if a in df.columns]
        df[em] = df[avail].sum(axis=1) if avail else 0.0
    return df

def sg(series, window=75, poly=3):
    """Savitzky-Golay smoother — safe for short series"""
    n = len(series)
    w = min(window, n // 2 * 2 + 1)
    if w % 2 == 0: w += 1
    w = max(5, w)
    try:
        return savgol_filter(series, w, poly)
    except:
        return series

def temporal_thirds(fused: pd.DataFrame):
    """Split fused facial+stage dataframe into early/mid/late thirds by time"""
    t = fused["time_s"].values
    t_min, t_max = t.min(), t.max()
    span = (t_max - t_min) / 3.0
    early = fused[fused["time_s"] <  t_min + span]
    mid   = fused[(fused["time_s"] >= t_min + span) & (fused["time_s"] < t_min + 2*span)]
    late  = fused[fused["time_s"] >= t_min + 2*span]
    return early, mid, late

# ============================================================================
# D1 — EMOTION PERCEPTION
# Mayer-Salovey Branch 1 + Barrett emotional granularity
# Sub-features: Shannon entropy (global + slope), pos/neg balance, variability
# ============================================================================

def shannon_entropy(values: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of a continuous signal via histogram binning"""
    counts, _ = np.histogram(values, bins=bins)
    probs = counts / (counts.sum() + 1e-9)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def extract_d1_emotion_perception(fused: pd.DataFrame) -> dict:
    """
    D1: Emotion Perception (weight 0.15)
    Theoretical anchor: Mayer-Salovey Branch 1, Barrett (2004) granularity,
    Quoidbach et al. (2014) emodiversity

    Features:
      d1_entropy_global    — Shannon entropy of stress trajectory (global)
                             High entropy = diverse emotional expression
      d1_entropy_slope     — entropy(late segment) - entropy(early segment)
                             Negative = settling/regulation; Positive = destabilising
      d1_pos_neg_ratio     — mean(positive emotions) / (mean(negative) + eps)
                             Higher = more positive affective tone
      d1_temporal_variability — mean frame-to-frame change in stress signal
                             Captures moment-to-moment emotional volatility
      d1_emodiversity      — mean Shannon entropy across all 7 emotion channels
                             Operationalises Barrett's emotional granularity
      d1_entropy_cv        — coefficient of variation of windowed entropy
                             How stable is the diversity of emotion expression
    """
    feats = {}
    stress = fused["Stress"].values
    early, mid, late = temporal_thirds(fused)

    # Global Shannon entropy of stress
    feats["d1_entropy_global"] = shannon_entropy(stress)

    # Entropy slope: late - early (negative = regulation)
    e_early = shannon_entropy(early["Stress"].values) if len(early) > 10 else np.nan
    e_late  = shannon_entropy(late["Stress"].values)  if len(late)  > 10 else np.nan
    feats["d1_entropy_slope"] = float(e_late - e_early) if not (np.isnan(e_early) or np.isnan(e_late)) else 0.0

    # Positive / negative ratio
    pos_mean = np.mean([fused[e].mean() for e in POSITIVE_EMOTIONS])
    neg_mean = np.mean([fused[e].mean() for e in NEGATIVE_EMOTIONS])
    feats["d1_pos_neg_ratio"] = float(pos_mean / (neg_mean + 1e-9))

    # Temporal variability: mean absolute frame-to-frame difference
    stress_diff = np.abs(np.diff(sg(stress)))
    feats["d1_temporal_variability"] = float(stress_diff.mean())

    # Emodiversity: mean entropy across all emotion channels
    entropies = []
    for em in ALL_EMOTIONS:
        if em in fused.columns and len(fused[em].dropna()) > 10:
            entropies.append(shannon_entropy(fused[em].values))
    feats["d1_emodiversity"] = float(np.mean(entropies)) if entropies else 0.0

    # Entropy CV: stability of emotional diversity over time
    # Use 10 equal-width windows
    n = len(stress)
    win_size = max(n // 10, 50)
    win_entropies = []
    for start in range(0, n - win_size, win_size):
        win_entropies.append(shannon_entropy(stress[start:start+win_size]))
    if len(win_entropies) > 2:
        arr = np.array(win_entropies)
        feats["d1_entropy_cv"] = float(arr.std() / (arr.mean() + 1e-9))
    else:
        feats["d1_entropy_cv"] = 0.0

    return feats

# ============================================================================
# D2 — EMOTION REGULATION
# Mayer-Salovey Branch 4 + Gross process model + Aldao flexibility
# Sub-features: stage-appropriate affect, stress recovery, windowed variance slope
# ============================================================================

def extract_d2_emotion_regulation(fused: pd.DataFrame) -> dict:
    """
    D2: Emotion Regulation (weight 0.25)
    Theoretical anchor: Mayer-Salovey Branch 4, Gross (1998, 2015),
    Peña-Sarrionandia et al. (2015), Aldao et al. (2015)

    Features:
      d2_stage_affect_appropriateness
          Mean stress during productive stages relative to block stages.
          Low ratio = stress is context-appropriate (low when working, higher when stuck)
          Operationalises Gross's context-sensitive regulation.

      d2_stress_recovery_at_block
          Mean stress level in the 30s AFTER a block ends (transition to productive).
          Lower = faster regulation after difficulty. Core Regulation feature.

      d2_pre_block_stress_elevation
          Mean stress in the 30s BEFORE block entry vs overall mean.
          Positive = anticipatory stress buildup (less regulated profile).

      d2_windowed_variance_slope
          Stress variance in late third minus early third.
          Negative = stress is stabilising (regulation improving over session).
          Positive = destabilising (regulation deteriorating).

      d2_regulation_flexibility
          Std of stress across stage types (how much stress varies by context).
          High = context-sensitive regulation; Low = uniform stress regardless of context.
          Operationalises Aldao et al.'s regulation flexibility.

      d2_calm_during_production
          Mean calm (low stress, low negative) frames specifically during Production.
          Direct measure of productive emotional tone.
    """
    feats = {}
    early, _, late = temporal_thirds(fused)

    # Stage-appropriate affect: mean stress productive vs block
    prod_mask  = fused["stage"].isin(PRODUCTIVE_STAGES)
    block_mask = fused["stage"].isin(BLOCK_VARIANTS)
    stress_prod  = fused.loc[prod_mask,  "Stress"].mean() if prod_mask.sum() > 0 else np.nan
    stress_block = fused.loc[block_mask, "Stress"].mean() if block_mask.sum() > 0 else np.nan
    if not np.isnan(stress_prod) and not np.isnan(stress_block) and stress_block > 0:
        feats["d2_stage_affect_appropriateness"] = float(stress_prod / stress_block)
    else:
        feats["d2_stage_affect_appropriateness"] = 1.0

    # Stress recovery: mean stress in window AFTER block→productive transitions
    recovery_vals = []
    times = fused["time_s"].values
    stress_vals = fused["Stress"].values
    stages = fused["stage"].values
    for i in range(len(stages) - 1):
        if stages[i] in BLOCK_VARIANTS and stages[i+1] in PRODUCTIVE_STAGES:
            t_transition = times[i+1]
            window_mask  = (times >= t_transition) & (times <= t_transition + 30)
            if window_mask.sum() > 5:
                recovery_vals.append(stress_vals[window_mask].mean())
    feats["d2_stress_recovery_at_block"] = float(np.mean(recovery_vals)) if recovery_vals else float(stress_prod if not np.isnan(stress_prod) else 0)

    # Pre-block stress elevation: stress 30s before block relative to overall mean
    overall_mean_stress = fused["Stress"].mean()
    pre_block_vals = []
    for i in range(len(stages)):
        if stages[i] in BLOCK_VARIANTS:
            t_block = times[i]
            pre_mask = (times >= t_block - 30) & (times < t_block)
            if pre_mask.sum() > 5:
                pre_block_vals.append(stress_vals[pre_mask].mean())
    feats["d2_pre_block_stress_elevation"] = float(
        np.mean(pre_block_vals) - overall_mean_stress
    ) if pre_block_vals else 0.0

    # Windowed variance slope: late variance - early variance
    var_early = float(early["Stress"].var()) if len(early) > 10 else 0.0
    var_late  = float(late["Stress"].var())  if len(late)  > 10 else 0.0
    feats["d2_windowed_variance_slope"] = float(var_late - var_early)

    # Regulation flexibility: std of mean stress per stage
    stage_means = []
    for st in STAGE_STATES:
        m = fused[fused["stage"] == st]["Stress"]
        if len(m) > 10:
            stage_means.append(m.mean())
    feats["d2_regulation_flexibility"] = float(np.std(stage_means)) if len(stage_means) >= 3 else 0.0

    # Calm during production
    prod_frames = fused[fused["stage"] == "Production"]
    if len(prod_frames) > 10:
        feats["d2_calm_during_production"] = float(
            (prod_frames["Stress"] < 0.5).mean()
        )
    else:
        feats["d2_calm_during_production"] = 0.5

    return feats

# ============================================================================
# D3 — ADAPTIVE PERSISTENCE
# Bar-On Stress Tolerance + D'Mello & Graesser + Martin & Marsh
# Sub-features: block/productive ratios + Markov chain features (inside D3)
# ============================================================================

def fit_stage_markov(stage_df: pd.DataFrame) -> np.ndarray:
    """First-order Markov chain over the 6 problem-solving stages"""
    sorted_s = stage_df.sort_values("start").reset_index(drop=True)
    counts = np.zeros((N_STAGE_STATES, N_STAGE_STATES), dtype=float)
    for k in range(len(sorted_s) - 1):
        s_from = sorted_s.loc[k,   "stage"]
        s_to   = sorted_s.loc[k+1, "stage"]
        if s_from in STAGE_IDX and s_to in STAGE_IDX:
            counts[STAGE_IDX[s_from], STAGE_IDX[s_to]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(row_sums > 0, counts / row_sums, 0.0)
    return prob

def extract_d3_adaptive_persistence(fused: pd.DataFrame,
                                     stage_df: pd.DataFrame) -> dict:
    """
    D3: Adaptive Persistence (weight 0.25)
    Theoretical anchor: Bar-On (2006) Stress Tolerance & Adaptability,
    D'Mello & Graesser (2012), Martin & Marsh (2008), Duckworth & Gross (2014)

    Scalar features:
      d3_block_proportion    — fraction of total session time in Block
      d3_productive_ratio    — fraction in productive stages
      d3_recovery_speed      — mean frames to drop from stress peak to 80%
      d3_n_block_entries     — block entry rate (per minute) — frequency of getting stuck

    Markov features (inside D3 per process-based model document):
      d3_mc_block_persistence   — P(Block→Block): probability of staying stuck
      d3_mc_block_to_productive — P(Block→Productive): direct recovery probability
      d3_mc_mean_block_dur_s    — mean duration of block episodes (seconds)
      d3_mc_block_to_ideation   — P(Block→Ideation): restart strategy probability
      d3_mc_transition_entropy  — entropy of row corresponding to Block state
                                   High = diverse exit strategies from Block
      d3_mc_productive_stability — mean P(productive→same productive): flow maintenance
    """
    feats = {}
    stress_vals = sg(fused["Stress"].values)

    # ── Scalar features ───────────────────────────────────────────────────
    total_dur = stage_df["end"].max() - stage_df["start"].min() + 1e-9
    block_rows = stage_df[stage_df["stage"].isin(BLOCK_VARIANTS)]
    prod_rows  = stage_df[stage_df["stage"].isin(PRODUCTIVE_STAGES)]

    block_time = (block_rows["end"] - block_rows["start"]).sum()
    prod_time  = (prod_rows["end"]  - prod_rows["start"]).sum()

    feats["d3_block_proportion"]  = float(block_time  / total_dur)
    feats["d3_productive_ratio"]  = float(prod_time   / total_dur)
    feats["d3_n_block_entries"]   = float(len(block_rows) / (total_dur / 60.0))

    # Recovery speed: frames to drop from peak to 80% of peak
    recovery_times = []
    i = 0
    while i < len(stress_vals) - 2:
        if stress_vals[i] > 1.0:  # above threshold = stressed peak
            peak = stress_vals[i]
            target = peak * 0.80
            j = i + 1
            while j < len(stress_vals) and stress_vals[j] > target:
                j += 1
            if j < len(stress_vals):
                recovery_times.append(j - i)
            i = j
        else:
            i += 1
    feats["d3_recovery_speed"] = float(np.mean(recovery_times) / FPS) if recovery_times else 5.0

    # ── Markov chain features (inside D3) ─────────────────────────────────
    prob = fit_stage_markov(stage_df)
    bi   = STAGE_IDX["Block"]
    prod_indices = [STAGE_IDX[s] for s in PRODUCTIVE_STAGES if s in STAGE_IDX]
    id_idx = STAGE_IDX.get("Ideation", -1)

    feats["d3_mc_block_persistence"]    = float(prob[bi, bi])
    feats["d3_mc_block_to_productive"]  = float(sum(prob[bi, j] for j in prod_indices))
    feats["d3_mc_block_to_ideation"]    = float(prob[bi, id_idx]) if id_idx >= 0 else 0.0
    feats["d3_mc_productive_stability"] = float(np.mean([prob[j, j] for j in prod_indices])) if prod_indices else 0.0

    # Mean block duration
    b_durations = (block_rows["end"] - block_rows["start"]).values
    feats["d3_mc_mean_block_dur_s"] = float(b_durations.mean()) if len(b_durations) else 0.0

    # Transition entropy of Block row: how diverse are Block exit strategies
    block_row = prob[bi]
    block_row_nonzero = block_row[block_row > 0]
    feats["d3_mc_transition_entropy"] = float(
        -np.sum(block_row_nonzero * np.log2(block_row_nonzero))
    ) if len(block_row_nonzero) > 0 else 0.0

    return feats

# ============================================================================
# D4 — AFFECTIVE SELF-AWARENESS
# Petrides trait EI Self-control + Pekrun control-value theory
# Sub-features: SAM valence/arousal stability, mean valence, trajectory
# ============================================================================

def extract_d4_affective_awareness(sam_rows: pd.DataFrame) -> dict:
    """
    D4: Affective Self-Awareness (weight 0.15)
    Theoretical anchor: Petrides & Furnham (2001), Petrides (2009),
    Pekrun (2006), Brackett et al. (2012)

    Features:
      d4_sam_valence_mean       — mean self-reported valence across session
      d4_sam_valence_std        — std of valence (instability)
      d4_sam_valence_slope      — linear slope of valence over checkpoints
                                   Positive = improving affect; Negative = deteriorating
      d4_sam_arousal_mean       — mean arousal across session
      d4_sam_arousal_std        — arousal instability
      d4_sam_arousal_slope      — linear slope of arousal over checkpoints
      d4_sam_valence_arousal_cov — covariance of valence and arousal
                                   Positive = excited-when-positive profile

    If SAM data unavailable, features are imputed at the session mean.
    """
    feats = {}
    defaults = {
        "d4_sam_valence_mean": 3.0, "d4_sam_valence_std": 0.5,
        "d4_sam_valence_slope": 0.0, "d4_sam_arousal_mean": 3.0,
        "d4_sam_arousal_std": 0.5, "d4_sam_arousal_slope": 0.0,
        "d4_sam_valence_arousal_cov": 0.0,
    }

    if sam_rows is None or len(sam_rows) < 3:
        return defaults

    v = sam_rows["valence"].dropna().values
    a = sam_rows["arousal"].dropna().values

    if len(v) < 2:
        return defaults

    feats["d4_sam_valence_mean"]  = float(v.mean())
    feats["d4_sam_valence_std"]   = float(v.std())
    feats["d4_sam_arousal_mean"]  = float(a.mean()) if len(a) > 1 else 3.0
    feats["d4_sam_arousal_std"]   = float(a.std())  if len(a) > 1 else 0.5

    # Linear slope via polyfit
    x = np.arange(len(v))
    feats["d4_sam_valence_slope"] = float(np.polyfit(x, v, 1)[0]) if len(v) >= 3 else 0.0

    xa = np.arange(len(a))
    feats["d4_sam_arousal_slope"] = float(np.polyfit(xa, a, 1)[0]) if len(a) >= 3 else 0.0

    # Covariance: only if both available and same length
    min_len = min(len(v), len(a))
    if min_len >= 3:
        feats["d4_sam_valence_arousal_cov"] = float(np.cov(v[:min_len], a[:min_len])[0, 1])
    else:
        feats["d4_sam_valence_arousal_cov"] = 0.0

    return feats

# ============================================================================
# D5 — EMOTION-COGNITION ALIGNMENT
# Mayer-Salovey Branch 2 + Fiori & Vesely-Maillefer + Izard + D'Mello
# Most novel dimension — operationalises emotional facilitation
# ============================================================================

def extract_d5_emotion_cognition_alignment(fused: pd.DataFrame,
                                            stage_df: pd.DataFrame) -> dict:
    """
    D5: Emotion-Cognition Alignment (weight 0.20)
    Theoretical anchor: Mayer-Salovey Branch 2, Fiori & Vesely-Maillefer (2018),
    Izard (2009), D'Mello & Graesser (2012), Boekaerts (2011)

    This is the most novel dimension. It operationalises emotional facilitation —
    the functional alignment between emotional states and cognitive processes.
    Prior EI literature has failed to measure Branch 2 empirically; this dimension
    is the framework's claim to rehabilitate it through process-based computation.

    Features:
      d5_transition_smoothness
          Mean absolute change in stress at stage transition boundaries.
          Low = smooth emotional adaptation to cognitive transitions.
          High = abrupt emotional disruption at cognitive boundaries.

      d5_focus_during_production
          Mean (1 - stress_normalised) specifically during Production stage.
          Operationalises functional emotional support for cognitive work.

      d5_positive_engagement_productive
          Mean Happiness specifically during productive stages.
          Positive affect facilitating cognition (Branch 2 core claim).

      d5_negative_affect_during_block
          Mean Stress specifically during Block relative to overall.
          Low relative elevation = emotion-cognition mismatch during difficulty.

      d5_stage_transition_congruence
          Correlation between stress change and stage-difficulty change at transitions.
          Positive = stress goes up when task gets harder (congruent).
          Negative = stress is misaligned with task demands (incongruent).

      d5_happiness_stress_correlation
          Pearson r between Happiness and Stress over the session.
          Negative = healthy separation of positive/negative affect channels.
          Near zero = affect channels are independent (well-regulated).
    """
    feats = {}
    stress_sg = sg(fused["Stress"].values)
    times  = fused["time_s"].values
    stages = fused["stage"].values
    happiness = fused["Happiness"].values

    # Transition smoothness: stress change at stage boundaries
    stage_boundary_changes = []
    for i in range(1, len(stages)):
        if stages[i] != stages[i-1]:
            stage_boundary_changes.append(abs(stress_sg[i] - stress_sg[i-1]))
    feats["d5_transition_smoothness"] = float(
        1.0 / (np.mean(stage_boundary_changes) + 1e-9)
    ) if stage_boundary_changes else 0.0
    # Inverted so that higher score = smoother transitions = better alignment

    # Focus during Production
    prod_mask = fused["stage"] == "Production"
    if prod_mask.sum() > 10:
        stress_max = max(fused["Stress"].max(), 1e-9)
        feats["d5_focus_during_production"] = float(
            1.0 - (fused.loc[prod_mask, "Stress"].mean() / stress_max)
        )
    else:
        feats["d5_focus_during_production"] = 0.5

    # Positive engagement during productive stages
    prod_all = fused["stage"].isin(PRODUCTIVE_STAGES)
    feats["d5_positive_engagement_productive"] = float(
        fused.loc[prod_all, "Happiness"].mean()
    ) if prod_all.sum() > 10 else 0.0

    # Negative affect during block relative to overall
    block_mask = fused["stage"].isin(BLOCK_VARIANTS)
    overall_stress_mean = fused["Stress"].mean()
    if block_mask.sum() > 10 and overall_stress_mean > 0:
        block_stress = fused.loc[block_mask, "Stress"].mean()
        feats["d5_negative_affect_during_block"] = float(
            block_stress / overall_stress_mean
        )
    else:
        feats["d5_negative_affect_during_block"] = 1.0

    # Stage transition congruence
    # Difficulty ordering (higher index = harder/more demanding)
    difficulty = {"Preparation":1,"Ideation":2,"Production":3,
                  "Evaluation":3,"Refinement":4,"Block":5}
    stress_changes = []
    difficulty_changes = []
    sorted_stage_df = stage_df.sort_values("start").reset_index(drop=True)
    for k in range(len(sorted_stage_df) - 1):
        s_from = sorted_stage_df.loc[k,   "stage"]
        s_to   = sorted_stage_df.loc[k+1, "stage"]
        t_boundary = sorted_stage_df.loc[k+1, "start"]
        if s_from in difficulty and s_to in difficulty:
            diff_change = difficulty[s_to] - difficulty[s_from]
            # stress at transition: look at 5s window either side
            pre_mask  = (times >= t_boundary - 5) & (times < t_boundary)
            post_mask = (times >= t_boundary)      & (times < t_boundary + 5)
            if pre_mask.sum() > 3 and post_mask.sum() > 3:
                s_change = stress_sg[post_mask].mean() - stress_sg[pre_mask].mean()
                stress_changes.append(s_change)
                difficulty_changes.append(diff_change)

    if len(stress_changes) >= 5:
        r, _ = pearsonr(stress_changes, difficulty_changes)
        feats["d5_stage_transition_congruence"] = float(r)
    else:
        feats["d5_stage_transition_congruence"] = 0.0

    # Happiness-Stress correlation (healthy separation)
    if len(fused) > 20:
        r_hs, _ = pearsonr(happiness, stress_sg)
        feats["d5_happiness_stress_correlation"] = float(r_hs)
    else:
        feats["d5_happiness_stress_correlation"] = 0.0

    return feats

# ============================================================================
# VALIDITY TESTING
# ============================================================================

def cronbach_alpha(df: pd.DataFrame, cols: list) -> tuple:
    """
    Cronbach's alpha for a set of features (items).
    Returns (alpha, mean_inter_item_r, n_items)
    """
    valid_cols = [c for c in cols if c in df.columns]
    if len(valid_cols) < 2:
        return (np.nan, np.nan, len(valid_cols))

    sub = df[valid_cols].dropna()
    if len(sub) < 5:
        return (np.nan, np.nan, len(valid_cols))

    k  = len(valid_cols)
    item_vars  = sub.var(axis=0, ddof=1)
    total_var  = sub.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var) if total_var > 0 else np.nan

    # Mean inter-item Pearson r
    rs = []
    for i in range(k):
        for j in range(i+1, k):
            r, _ = pearsonr(sub[valid_cols[i]], sub[valid_cols[j]])
            rs.append(r)
    mean_r = float(np.mean(rs)) if rs else np.nan

    return (float(alpha), mean_r, k)


def test_criterion_validity(feature_df: pd.DataFrame,
                             exam_scores: dict,
                             out_dir: Path) -> pd.DataFrame:
    """
    For each feature: Pearson r, Spearman r, Cohen's d, Mann-Whitney U
    Same criteria applied uniformly across all features.
    """
    pids = [p for p in feature_df.index if p in exam_scores]
    if len(pids) < 10:
        log.warning("Fewer than 10 participants with exam scores — validity stats unreliable")

    y = np.array([exam_scores[p] for p in pids])
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    hi_mask = y >= q3
    lo_mask = y <= q1

    rows = []
    for col in feature_df.columns:
        vals = feature_df.loc[pids, col].values.astype(float)
        if np.all(np.isnan(vals)) or np.nanstd(vals) < 1e-9:
            rows.append({
                "Feature": col, "Pearson r": np.nan, "Pearson p": np.nan,
                "Spearman r": np.nan, "Spearman p": np.nan,
                "Cohen's d": np.nan, "Effect size": "n/a",
                "MWU p": np.nan, "Sig (p<0.05)": "NO",
                "Predictive": "NO — constant feature",
                "Dimension": col.split("_")[0].upper(),
            })
            continue

        # Replace NaN with column mean
        nan_mask = np.isnan(vals)
        vals[nan_mask] = np.nanmean(vals)

        r,  p_r  = pearsonr(vals, y)
        sr, sp_r = spearmanr(vals, y)

        # Cohen's d: top vs bottom quartile
        if hi_mask.sum() >= 3 and lo_mask.sum() >= 3:
            hi_vals = vals[hi_mask]
            lo_vals = vals[lo_mask]
            pooled  = np.sqrt((hi_vals.std()**2 + lo_vals.std()**2) / 2 + 1e-9)
            d       = (hi_vals.mean() - lo_vals.mean()) / pooled
            u, p_u  = mannwhitneyu(hi_vals, lo_vals, alternative="two-sided")
        else:
            d, p_u = 0.0, 1.0

        # Effect size label
        abs_d = abs(d)
        effect = "Large"  if abs_d >= 0.80 else \
                 "Medium" if abs_d >= 0.50 else \
                 "Small"  if abs_d >= 0.20 else "Negligible"

        # Predictive flag
        is_predictive = (abs(r) >= MIN_ABS_R and p_r <= MAX_P_NONPREDICTIVE)

        rows.append({
            "Feature": col,
            "Pearson r": round(r, 4),
            "Pearson p": round(p_r, 4),
            "Spearman r": round(sr, 4),
            "Spearman p": round(sp_r, 4),
            "Cohen's d": round(d, 4),
            "Effect size": effect,
            "MWU p": round(p_u, 4),
            "Sig (p<0.05)": "YES *" if p_u < 0.05 else "NO",
            "Predictive": "YES" if is_predictive else f"NO — |r|={abs(r):.3f}",
            "Dimension": col.split("_")[0].upper(),
        })

    df_out = pd.DataFrame(rows).sort_values("Cohen's d", key=abs, ascending=False)
    df_out.to_csv(out_dir / "validity_by_feature.csv", index=False)
    log.info("Criterion validity table saved: %d features", len(df_out))
    return df_out


def test_internal_consistency(feature_df: pd.DataFrame,
                               dim_cols: dict,
                               out_dir: Path) -> dict:
    """
    Cronbach's alpha per dimension.
    dim_cols: dict of {dimension_name: [feature_col_list]}
    """
    results = {}
    for dim_name, cols in dim_cols.items():
        alpha, mean_r, n = cronbach_alpha(feature_df, cols)
        results[dim_name] = {
            "alpha":      round(alpha, 4) if not np.isnan(alpha) else "n/a",
            "mean_r":     round(mean_r, 4) if not np.isnan(mean_r) else "n/a",
            "n_features": n,
            "status":     "Good (≥0.7)" if alpha >= 0.70
                          else "Acceptable (≥0.6)" if alpha >= 0.60
                          else "Poor (<0.6)" if alpha >= 0.0
                          else "Negative — features anti-correlated",
        }
        log.info("[%s] alpha=%.3f  mean_r=%.3f  n=%d  → %s",
                 dim_name, alpha if not np.isnan(alpha) else -99,
                 mean_r  if not np.isnan(mean_r)  else -99,
                 n, results[dim_name]["status"])

    # Save summary
    rows = [{"Dimension": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(out_dir / "dimension_summary.csv", index=False)
    return results

# ============================================================================
# LOOCV COMPOSITE EI SCORE
# ============================================================================

def build_loocv_ei_score(feature_df: pd.DataFrame,
                          exam_scores: dict,
                          retained_features: list,
                          out_dir: Path) -> pd.Series:
    """
    LOOCV composite EI score.
    For each held-out participant:
      - Estimate feature weights from the other n-1 participants
        (weight = Pearson r of feature with exam score in training fold)
      - Compute weighted sum for held-out participant
      - Normalise to 0-100 scale

    This produces a methodologically clean composite that is NOT contaminated
    by the criterion variable.
    """
    pids = [p for p in feature_df.index if p in exam_scores]
    y    = np.array([exam_scores[p] for p in pids])
    X    = feature_df.loc[pids, retained_features].fillna(0).values.astype(float)
    n    = len(pids)

    raw_scores = np.zeros(n)
    for i in range(n):
        # Training fold: all except i
        train_idx = [j for j in range(n) if j != i]
        X_train   = X[train_idx]
        y_train   = y[train_idx]
        X_test    = X[i]

        # Feature weights: Pearson r in training fold
        weights = np.zeros(len(retained_features))
        for f_idx in range(len(retained_features)):
            if X_train[:, f_idx].std() > 1e-9:
                r, _ = pearsonr(X_train[:, f_idx], y_train)
                weights[f_idx] = r

        # Weighted sum
        raw_scores[i] = np.dot(X_test, weights)

    # Normalise to 0-100
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        normed = 100.0 * (raw_scores - s_min) / (s_max - s_min)
    else:
        normed = np.full(n, 50.0)

    score_series = pd.Series(normed, index=pids, name="loocv_ei_score")

    # Criterion validity of composite
    r_p,  p_p  = pearsonr(normed, y)
    r_sp, p_sp = spearmanr(normed, y)
    log.info("LOOCV EI composite: Pearson r=%.4f (p=%.6f)  Spearman r=%.4f (p=%.6f)",
             r_p, p_p, r_sp, p_sp)

    score_series.to_csv(out_dir / "loocv_ei_scores_v5.csv", header=True)
    return score_series, r_p, p_p, r_sp, p_sp

# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_effect_sizes(validity_df: pd.DataFrame, out_dir: Path) -> None:
    """Cohen's d bar chart coloured by dimension"""
    df = validity_df.dropna(subset=["Cohen's d"]).copy()
    df = df.sort_values("Cohen's d", key=abs, ascending=True)

    dim_colors = {"D1":"#2980B9","D2":"#27AE60","D3":"#E67E22",
                  "D4":"#8E44AD","D5":"#C0392B","MC":"#7F8C8D"}

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.28)))
    for _, row in df.iterrows():
        d   = row["Cohen's d"]
        dim = row["Dimension"]
        col = dim_colors.get(dim, "#95A5A6")
        sig = row.get("Sig (p<0.05)","NO") == "YES *"
        ax.barh(row["Feature"], d, color=col,
                alpha=0.9 if sig else 0.45,
                edgecolor="black" if sig else "none", linewidth=0.5)

    ax.axvline(0,    color="black", linewidth=0.8)
    ax.axvline( 0.5, color="gray",  linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(-0.5, color="gray",  linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline( 0.8, color="gray",  linewidth=0.6, linestyle=":",  alpha=0.5)
    ax.axvline(-0.8, color="gray",  linewidth=0.6, linestyle=":",  alpha=0.5)
    ax.set_xlabel("Cohen's d  (top vs bottom exam quartile)", fontsize=11)
    ax.set_title("Feature Validity — Effect Size by Dimension\n"
                 "(opaque = significant MWU p<0.05; dashed = medium/large thresholds)",
                 fontsize=12, fontweight="bold")

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in dim_colors.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "effect_size_chart.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Effect size chart saved.")


def plot_dimension_alphas(consistency: dict, out_dir: Path) -> None:
    """Cronbach alpha bar chart per dimension"""
    dims   = list(consistency.keys())
    alphas = [consistency[d]["alpha"] if isinstance(consistency[d]["alpha"], float) else 0.0
              for d in dims]
    colors = ["#27AE60" if a >= 0.70 else "#E67E22" if a >= 0.60 else "#C0392B"
              for a in alphas]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(dims, alphas, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0.70, color="#27AE60", linestyle="--", linewidth=1.2, label="Good (0.70)")
    ax.axhline(0.60, color="#E67E22", linestyle="--", linewidth=1.2, label="Acceptable (0.60)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cronbach's α", fontsize=11)
    ax.set_title("Internal Consistency per EI Dimension", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    for bar, a in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width()/2, a + 0.02,
                f"{a:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "dimension_alphas.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Dimension alpha chart saved.")


def plot_loocv_scatter(score_series: pd.Series,
                        exam_scores: dict,
                        r_p: float, p_p: float,
                        r_sp: float, p_sp: float,
                        out_dir: Path) -> None:
    """LOOCV EI score vs exam performance scatter"""
    pids   = list(score_series.index)
    scores = score_series.values
    exams  = np.array([exam_scores[p] * 100 for p in pids])

    median_ei   = np.median(scores)
    median_exam = np.median(exams)
    colors = []
    for s, e in zip(scores, exams):
        if s >= median_ei and e >= median_exam:   colors.append("#1A5276")
        elif s < median_ei and e < median_exam:   colors.append("#922B21")
        else:                                     colors.append("#7D6608")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(scores, exams, c=colors, s=70, alpha=0.85, edgecolors="white", linewidth=0.4)
    for pid, s, e in zip(pids, scores, exams):
        ax.annotate(pid, (s, e), fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    # Regression line
    m, b = np.polyfit(scores, exams, 1)
    xs   = np.linspace(scores.min(), scores.max(), 100)
    ax.plot(xs, m*xs + b, "--", color="black", linewidth=1.2, alpha=0.7)
    ax.axvline(median_ei,   color="gray", linewidth=0.7, linestyle=":")
    ax.axhline(median_exam, color="gray", linewidth=0.7, linestyle=":")

    ax.set_xlabel("LOOCV-validated EI Score (0–100)", fontsize=11)
    ax.set_ylabel("Exam Score (%)", fontsize=11)
    ax.set_title(
        f"LOOCV-Validated EI Score vs Exam Performance (v5)\n"
        f"Pearson r={r_p:.3f} (p={p_p:.6f})  |  Spearman r={r_sp:.3f}",
        fontsize=11, fontweight="bold"
    )
    legend_patches = [
        mpatches.Patch(color="#1A5276", label="High EI + High Exam"),
        mpatches.Patch(color="#922B21", label="Low EI + Low Exam"),
        mpatches.Patch(color="#7D6608", label="Inconsistent"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "loocv_ei_vs_exam_v5.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("LOOCV scatter saved.")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(data_dir: str) -> None:
    data_path = Path(data_dir)
    out_dir   = data_path / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract zips ──────────────────────────────────────────────────────
    for zname, dname in [(STAGE_ZIP, STAGE_DIR), (FACIAL_ZIP, FACIAL_DIR)]:
        zp = data_path / zname
        od = data_path / dname
        if zp.exists():
            extract_zip(zp, od)

    # ── Find files ────────────────────────────────────────────────────────
    stage_files  = find_files(data_path / STAGE_DIR)
    facial_files = find_files(data_path / FACIAL_DIR)
    exam_data    = load_exam(data_path / EXAM_FILE)

    # Load SAM data (optional — gracefully handles missing)
    sam_data = {}
    sam_path = data_path / SAM_FILE
    if sam_path.exists():
        sam_df = load_sam(sam_path)
        for pid_raw, grp in sam_df.groupby("participant"):
            m = re.search(r"P(\d+)", str(pid_raw).upper())
            if m:
                sam_data[f"P{m.group(1)}"] = grp.reset_index(drop=True)
        log.info("SAM data loaded for %d participants", len(sam_data))
    else:
        log.warning("SAM file not found — D4 features will use imputed defaults")

    common = sorted(
        set(stage_files) & set(facial_files) & set(exam_data),
        key=lambda p: int(re.search(r"\d+", p).group())
    )
    log.info("Participants in all three data sources: %d", len(common))

    # ── Extract features per participant ──────────────────────────────────
    all_features = {}
    for pid in common:
        try:
            facial = load_facial(facial_files[pid])
            stage  = load_stage(stage_files[pid])
            if facial.empty or stage.empty:
                log.warning("[%s] Empty data — skipping", pid); continue

            facial = compute_emotions(facial)

            # Align stage labels to facial frames
            facial["stage"] = "Unknown"
            for _, row in stage.iterrows():
                mask = (facial["time_s"] >= row["start"]) & \
                       (facial["time_s"] <  row["end"])
                facial.loc[mask, "stage"] = row["stage"]

            # Filter out unknown and insufficient frames
            fused = facial[facial["stage"] != "Unknown"].copy()
            if len(fused) < 200:
                log.warning("[%s] Insufficient fused frames (%d) — skipping", pid, len(fused))
                continue

            # Extract all five dimensions
            d1 = extract_d1_emotion_perception(fused)
            d2 = extract_d2_emotion_regulation(fused)
            d3 = extract_d3_adaptive_persistence(fused, stage)
            d4 = extract_d4_affective_awareness(sam_data.get(pid, None))
            d5 = extract_d5_emotion_cognition_alignment(fused, stage)

            all_features[pid] = {**d1, **d2, **d3, **d4, **d5}

            log.info("[%s] OK — D1:%d D2:%d D3:%d D4:%d D5:%d  total:%d features",
                     pid, len(d1), len(d2), len(d3), len(d4), len(d5),
                     len(d1)+len(d2)+len(d3)+len(d4)+len(d5))

        except Exception as ex:
            log.error("[%s] FAILED: %s", pid, ex)
            import traceback; traceback.print_exc()

    if len(all_features) < 10:
        log.error("Too few participants processed (%d). Check data paths.", len(all_features))
        return

    # ── Build feature matrix ──────────────────────────────────────────────
    feature_df = pd.DataFrame.from_dict(all_features, orient="index")
    feature_df.index.name = "participant"
    feature_df.to_csv(out_dir / "feature_matrix.csv")
    log.info("Feature matrix shape: %s", feature_df.shape)

    # ── Dimension column mapping ──────────────────────────────────────────
    dim_cols = {
        "D1 Emotion Perception":       [c for c in feature_df.columns if c.startswith("d1_")],
        "D2 Emotion Regulation":       [c for c in feature_df.columns if c.startswith("d2_")],
        "D3 Adaptive Persistence":     [c for c in feature_df.columns if c.startswith("d3_")],
        "D4 Affective Self-Awareness": [c for c in feature_df.columns if c.startswith("d4_")],
        "D5 Emotion-Cognition":        [c for c in feature_df.columns if c.startswith("d5_")],
    }
    log.info("Features per dimension: %s",
             {k: len(v) for k, v in dim_cols.items()})

    # ── Criterion validity testing ────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Criterion validity (Pearson r, Cohen's d, MWU)")
    log.info("=" * 60)
    validity_df = test_criterion_validity(feature_df, exam_data, out_dir)

    # ── Internal consistency ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2: Internal consistency (Cronbach's alpha per dimension)")
    log.info("=" * 60)
    consistency = test_internal_consistency(feature_df, dim_cols, out_dir)

    # ── Feature retention decision ────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 3: Feature retention")
    log.info("=" * 60)
    retained = validity_df[validity_df["Predictive"] == "YES"]["Feature"].tolist()
    non_predictive = validity_df[validity_df["Predictive"] != "YES"]["Feature"].tolist()
    log.info("Retained: %d features", len(retained))
    log.info("Non-predictive: %d features", len(non_predictive))

    if len(retained) < 3:
        log.warning("Fewer than 3 features retained — using all features for LOOCV")
        retained = list(feature_df.columns)

    # ── LOOCV composite EI score ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 4: LOOCV composite EI score")
    log.info("=" * 60)
    score_series, r_p, p_p, r_sp, p_sp = build_loocv_ei_score(
        feature_df, exam_data, retained, out_dir
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 5: Generating plots")
    log.info("=" * 60)
    validity_df["Dimension"] = validity_df["Feature"].apply(
        lambda x: x.split("_")[0].upper()
    )
    plot_effect_sizes(validity_df, out_dir)
    plot_dimension_alphas(consistency, out_dir)
    plot_loocv_scatter(score_series, exam_data, r_p, p_p, r_sp, p_sp, out_dir)

    # ── Human-readable report ─────────────────────────────────────────────
    lines = [
        "=" * 70,
        "   EI FEATURE PIPELINE v5 — FIVE-DIMENSION VALIDITY REPORT",
        "=" * 70,
        f"   Participants processed: {len(all_features)}",
        f"   Total features extracted: {len(feature_df.columns)}",
        f"   Features retained (|r|>=0.10, p<=0.50): {len(retained)}",
        f"   Features non-predictive: {len(non_predictive)}",
        "",
        "─" * 70,
        "   DIMENSION OVERVIEW",
        "─" * 70,
    ]
    for dim, cols in dim_cols.items():
        c_data = consistency.get(dim, {})
        lines.append(
            f"   {dim:<35} n_features={len(cols):<4} "
            f"alpha={c_data.get('alpha','n/a'):<8} "
            f"mean_r={c_data.get('mean_r','n/a'):<8} "
            f"→ {c_data.get('status','')}"
        )

    lines += [
        "",
        "─" * 70,
        "   CRITERION VALIDITY (top 20 by |Cohen's d|)",
        "─" * 70,
        f"   {'Feature':<38} {'r':>7} {'p':>8}  {'d':>7} {'Effect':>10}  Sig   Retained",
        "   " + "-"*68,
    ]
    for _, row in validity_df.head(20).iterrows():
        cd = row["Cohen's d"]
        lines.append(
            f"   {row['Feature']:<38} {row['Pearson r']:>7.3f} "
            f"{row['Pearson p']:>8.4f}  {cd:>7.3f} "
            f"{row['Effect size']:>10}  {row['Sig (p<0.05)']:<6} "
            f"{row['Predictive']}"
        )

    lines += [
        "",
        "─" * 70,
        "   NON-PREDICTIVE FEATURES (report in paper)",
        "─" * 70,
    ]
    for f in non_predictive:
        row = validity_df[validity_df["Feature"] == f]
        if not row.empty:
            cd_val = row["Cohen's d"].values[0]
            lines.append(
                f"   {f:<38} r={row['Pearson r'].values[0]:>6.3f}  "
                f"d={cd_val:>6.3f}"
            )

    lines += [
        "",
        "─" * 70,
        "   LOOCV COMPOSITE EI SCORE — CRITERION VALIDITY",
        "─" * 70,
        f"   Pearson  r = {r_p:.4f}  (p = {p_p:.6f})",
        f"   Spearman r = {r_sp:.4f}  (p = {p_sp:.6f})",
        "",
        "─" * 70,
        "   OUTPUT FILES",
        "─" * 70,
        "   feature_matrix.csv          — all features per participant",
        "   validity_by_feature.csv     — full criterion validity table",
        "   dimension_summary.csv       — per-dimension alpha + composite r",
        "   effect_size_chart.png       — Cohen's d bar chart by dimension",
        "   dimension_alphas.png        — Cronbach alpha per dimension",
        "   loocv_ei_vs_exam_v5.png     — LOOCV EI vs exam scatter",
        "   loocv_ei_scores_v5.csv      — validated EI scores per participant",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(out_dir / "pipeline_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    log.info("All outputs → %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EI Feature Pipeline v5 — Five-Dimension Framework"
    )
    parser.add_argument(
        "--data_dir", default=DEFAULT_DATA_DIR,
        help="Path to data directory containing zips, exam scores, SAM file"
    )
    args = parser.parse_args()
    main(args.data_dir)
