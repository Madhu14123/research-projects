"""
Markov Chain Feature Generation — Stage 2, Week 4
===================================================
Generates transition-based candidate features from two Markov chain views:

  View 1 — Stage-level Markov chain
    States: the 6 problem-solving stages (Preparation, Ideation,
    Production, Evaluation, Refinement, Block).
    Transition matrix P[i,j] = P(stage j follows stage i).
    Scalar features extracted: block persistence, recovery probability,
    productive self-transition, mean time per stage.

  View 2 — Emotional-state Markov chain
    States: 4 discretised emotional states (Calm, Stressed, Happy, Mixed).
    Derived from the continuous AU intensity signal.
    Scalar features extracted: P(Stressed→Calm), P(Calm→Stressed),
    mean self-transition (stability), stressed state occupancy,
    mean sojourn time in stressed state.

All extracted features are treated as candidate sub-features and go
through the same validity tests as existing scalar features:
  - Criterion validity: Pearson r with exam score
  - Internal consistency: Cronbach alpha within dimension

Reference:
  Rabiner (1989) A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition. Proc. IEEE 77(2), 257-286.

Output → results_markov/
  markov_features.csv     — all Markov features per participant
  markov_validity.csv     — criterion validity table
  transition_matrices.png — stage transition matrices (hi vs lo scorers)
  emotional_state_transitions.png — emotional state matrices
  markov_report.txt       — summary

Usage:
    python ei_markov_features.py
    python ei_markov_features.py --data_dir "D:/A_emotions in exam/data"
"""

import re, zipfile, argparse, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
RESULTS_DIR      = "results_markov"

STAGE_ZIP  = "stage coded data-20260321T120851Z-1-001.zip"
FACIAL_ZIP = "facial expressions- data-20260321T182549Z-1-001.zip"
EXAM_FILE  = "Exam Scores.xlsx"
SAM_FILE   = "SAM Responses every 3 minutes.xlsx"
STAGE_DIR  = "stage_coded_data"
FACIAL_DIR = "facial_expressions"

FPS = 25

# ── Stage-level Markov ────────────────────────────────────────────────────
STAGE_STATES = [
    "Preparation", "Ideation", "Production",
    "Evaluation",  "Refinement", "Block",
]
STAGE_IDX = {s: i for i, s in enumerate(STAGE_STATES)}
N_STAGE_STATES = len(STAGE_STATES)

PRODUCTIVE_STAGES = {"Production", "Refinement", "Evaluation"}
BLOCK_STAGES      = {"Block", "Block ", "block"}

STAGE_MAP = {
    "preparation": "Preparation", "ideation":    "Ideation",
    "production":  "Production",  "prod":        "Production",
    "evaluation":  "Evaluation",  "refinement":  "Refinement",
    "block":       "Block",       "sam":         "SAM",
}

# ── Emotional-state Markov ────────────────────────────────────────────────
# Discretisation thresholds (AU intensity scale 0-5)
STRESS_THRESH   = 1.0   # above this = Stressed
HAPPY_THRESH    = 0.5   # above this = Happy
CALM_STRESS_MAX = 0.5   # below this (and low happy) = Calm

STATE_CALM     = 0
STATE_STRESSED = 1
STATE_HAPPY    = 2
STATE_MIXED    = 3
N_EMO_STATES   = 4
EMO_STATE_NAMES = ["Calm", "Stressed", "Happy", "Mixed"]

EMOTION_AU = {
    "Anger":     ["AU04_r","AU05_r","AU07_r","AU23_r"],
    "Fear":      ["AU01_r","AU04_r","AU05_r"],
    "Sadness":   ["AU01_r","AU04_r","AU15_r"],
    "Disgust":   ["AU04_r","AU07_r","AU09_r","AU10_r","AU17_r","AU20_r"],
    "Pride":     ["AU06_r","AU07_r","AU12_r"],
    "Happiness": ["AU06_r","AU12_r"],
    "Stress":    ["AU01_r","AU06_r","AU12_r","AU15_r"],
}

# ============================================================================
# DATA LOADING
# ============================================================================

def extract_zip(zp, od):
    if od.exists() and any(od.iterdir()): return
    od.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s ...", zp.name)
    with zipfile.ZipFile(zp, "r") as z: z.extractall(od)

def pid_from(fn):
    m = re.search(r"(P\d+)", fn, re.I)
    return m.group(1).upper() if m else None

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
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df.columns = df.columns.str.strip()
    if "cleaned time" in df.columns: df = df.rename(columns={"cleaned time": "time_s"})
    elif "timestamp"  in df.columns: df = df.rename(columns={"timestamp": "time_s"})
    r_cols = [c for c in df.columns if c.endswith("_r")]
    keep = ["time_s"] + [c for c in ["confidence","success"] if c in df.columns] + r_cols
    df = df[[c for c in keep if c in df.columns]].copy()
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
    for col in ["start", "end"]:
        if col not in df.columns: continue
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
        elif df[col].dtype == object:
            def _s(v):
                try:
                    p = str(v).split(":"); return sum(float(x)*60**(len(p)-i-1) for i,x in enumerate(p))
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
        try: scores[f"P{m.group(1)}"] = float(row.iloc[1]) / (float(row.iloc[2]) if len(row)>2 else 26.)
        except: pass
    return scores

def compute_emotions(df):
    for em, aus in EMOTION_AU.items():
        avail = [a for a in aus if a in df.columns]
        df[em] = df[avail].sum(axis=1) if avail else 0.0
    return df

def sg(series, window=75, poly=3):
    n = len(series); w = min(window, n//2*2+1)
    if w % 2 == 0: w += 1
    w = max(5, w)
    return savgol_filter(series, w, poly) if n >= w else series

# ============================================================================
# VIEW 1 — STAGE-LEVEL MARKOV CHAIN
# ============================================================================

def fit_stage_markov(stage_df: pd.DataFrame) -> np.ndarray:
    """
    Fit a first-order Markov chain over the sequence of problem-solving
    stages for one participant.

    Returns a (N_STAGE_STATES × N_STAGE_STATES) row-normalised
    transition probability matrix.

    P[i, j] = P(next stage = j | current stage = i)

    Reference: Rabiner (1989) Proc. IEEE 77(2), 257-286.
    """
    sorted_s = stage_df.sort_values("start").reset_index(drop=True)
    counts   = np.zeros((N_STAGE_STATES, N_STAGE_STATES), dtype=float)

    for k in range(len(sorted_s) - 1):
        s_from = sorted_s.loc[k,   "stage"]
        s_to   = sorted_s.loc[k+1, "stage"]
        if s_from in STAGE_IDX and s_to in STAGE_IDX:
            counts[STAGE_IDX[s_from], STAGE_IDX[s_to]] += 1

    # Row-normalise — rows with zero counts stay zero
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(row_sums > 0, counts / row_sums, 0.0)
    return prob


def extract_stage_markov_features(prob: np.ndarray,
                                   stage_df: pd.DataFrame) -> dict:
    """
    Extract interpretable scalar features from the stage transition matrix.

    Features and their EI interpretation
    ─────────────────────────────────────
    mc_s_block_persistence
        P(Block → Block): probability of staying in Block on the next
        stage transition. High value = gets stuck repeatedly; low value
        = escapes difficulty quickly. Maps to Behavioral Persistence.

    mc_s_block_to_productive
        P(Block → Production|Evaluation|Refinement): probability of
        directly entering a productive stage after Block. Higher = more
        resilient recovery. Maps to Behavioral Persistence.

    mc_s_block_to_ideation
        P(Block → Ideation): returns to idea-generation after Block.
        Indicates a strategy-reset response rather than direct re-attempt.

    mc_s_productive_stability
        Mean P(productive → same productive stage): how consistently the
        participant stays in productive work once there. Maps to
        Emotion-Cognition Alignment.

    mc_s_mean_sojourn_block
        Mean duration of Block episodes (seconds). Computed directly from
        stage annotations. Lower = faster escape from difficulty.

    mc_s_n_block_entries
        Count of distinct Block episode entries normalised by session
        duration (per minute). Captures frequency of getting stuck.
    """
    # Block state index
    bi = STAGE_IDX["Block"]

    # Block persistence: P(Block → Block)
    block_persistence = float(prob[bi, bi])

    # Block → productive (any of Production, Evaluation, Refinement)
    prod_indices = [STAGE_IDX[s] for s in PRODUCTIVE_STAGES if s in STAGE_IDX]
    block_to_productive = float(sum(prob[bi, j] for j in prod_indices))

    # Block → Ideation
    block_to_ideation = float(prob[bi, STAGE_IDX.get("Ideation", -1)]) \
                        if "Ideation" in STAGE_IDX else 0.0

    # Productive self-transition: mean P(prod_i → prod_i) for productive states
    prod_self = np.mean([prob[j, j] for j in prod_indices]) if prod_indices else 0.0

    # Mean sojourn time in Block
    b_rows = stage_df[stage_df["stage"].isin(BLOCK_STAGES)]
    durations = (b_rows["end"] - b_rows["start"]).values
    mean_block_dur = float(durations.mean()) if len(durations) else 0.0

    # Block entry rate (per minute)
    total_duration_min = (
        stage_df["end"].max() - stage_df["start"].min()
    ) / 60.0 + 1e-9
    n_block_entries = len(b_rows)
    block_entry_rate = float(n_block_entries / total_duration_min)

    return {
        "mc_s_block_persistence":    block_persistence,
        "mc_s_block_to_productive":  block_to_productive,
        "mc_s_block_to_ideation":    block_to_ideation,
        "mc_s_productive_stability": float(prod_self),
        "mc_s_mean_block_dur_s":     mean_block_dur,
        "mc_s_block_entry_rate":     block_entry_rate,
    }


# ============================================================================
# VIEW 2 — EMOTIONAL-STATE MARKOV CHAIN
# ============================================================================

def discretise_emotional_state(stress: float, happy: float) -> int:
    """
    Map continuous (Stress, Happiness) values to a discrete state.

    Thresholds chosen to match the AU intensity scale (0-5):
      Stressed : Stress >= 1.0  (moderate facial tension)
      Happy    : Happiness >= 0.5 and not Stressed
      Calm     : both below threshold
      Mixed    : everything else
    """
    if   stress >= STRESS_THRESH:                          return STATE_STRESSED
    elif happy  >= HAPPY_THRESH:                           return STATE_HAPPY
    elif stress <  CALM_STRESS_MAX and happy < HAPPY_THRESH: return STATE_CALM
    else:                                                  return STATE_MIXED


def fit_emotional_markov(fused: pd.DataFrame) -> tuple:
    """
    Fit a first-order Markov chain over the discretised emotional state
    sequence for one participant.

    Smooths the continuous AU signals first (Savitzky-Golay) to reduce
    noise-driven micro-transitions before discretisation.

    Returns:
        prob  — (N_EMO_STATES × N_EMO_STATES) transition probability matrix
        states — array of state labels per frame
    """
    stress = sg(fused["Stress"].values)
    happy  = sg(fused["Happiness"].values)
    states = np.array([discretise_emotional_state(s, h)
                       for s, h in zip(stress, happy)])

    counts = np.zeros((N_EMO_STATES, N_EMO_STATES), dtype=float)
    for k in range(len(states) - 1):
        counts[states[k], states[k+1]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(row_sums > 0, counts / row_sums, 0.0)
    return prob, states


def extract_emotional_markov_features(prob: np.ndarray,
                                       states: np.ndarray) -> dict:
    """
    Extract scalar features from the emotional-state transition matrix.

    Features and their EI interpretation
    ─────────────────────────────────────
    mc_e_stressed_to_calm
        P(Stressed → Calm): probability of recovering from stress to calm
        on the next frame transition. Core measure of emotional regulation
        speed. Maps directly to the Emotional Regulation dimension.

    mc_e_calm_to_stressed
        P(Calm → Stressed): probability of entering stress from calm.
        Low value = emotionally stable and not easily disturbed. Maps to
        Emotional Awareness and Regulation.

    mc_e_stressed_stability
        P(Stressed → Stressed): self-transition within stressed state.
        High value = stress persists once triggered. Low value = rapid
        escape. Captures exactly what c3_block_score does not: whether
        stress within a block episode is maintained or resolves.

    mc_e_calm_stability
        P(Calm → Calm): proportion of calm frames that are followed by
        another calm frame. High value = sustained composure.

    mc_e_mean_stability
        Mean of all four diagonal self-transition probabilities. Overall
        emotional stability index — how often the participant stays in
        whatever state they are currently in, regardless of which state.

    mc_e_stressed_occupancy
        Fraction of total frames spent in the Stressed state. Provides
        a baseline against which the transition rates are interpreted.

    mc_e_mean_sojourn_stressed
        Mean run length of consecutive Stressed frames, converted to
        seconds. Higher = stress episodes are longer and less fragmented.
    """
    p_stressed_to_calm    = float(prob[STATE_STRESSED, STATE_CALM])
    p_calm_to_stressed    = float(prob[STATE_CALM,     STATE_STRESSED])
    p_stressed_stability  = float(prob[STATE_STRESSED, STATE_STRESSED])
    p_calm_stability      = float(prob[STATE_CALM,     STATE_CALM])
    mean_stability        = float(np.diag(prob).mean())
    n                     = len(states)
    stressed_occupancy    = float((states == STATE_STRESSED).sum() / n)

    # Mean sojourn time in Stressed state (run-length encoding)
    in_run      = False
    run_lengths = []
    current_run = 0
    for s in states:
        if s == STATE_STRESSED:
            in_run = True; current_run += 1
        else:
            if in_run:
                run_lengths.append(current_run)
                current_run = 0
                in_run = False
    if in_run and current_run > 0:
        run_lengths.append(current_run)

    mean_sojourn_s = float(np.mean(run_lengths) / FPS) if run_lengths else 0.0

    return {
        "mc_e_stressed_to_calm":   p_stressed_to_calm,
        "mc_e_calm_to_stressed":   p_calm_to_stressed,
        "mc_e_stressed_stability": p_stressed_stability,
        "mc_e_calm_stability":     p_calm_stability,
        "mc_e_mean_stability":     mean_stability,
        "mc_e_stressed_occupancy": stressed_occupancy,
        "mc_e_mean_sojourn_s":     mean_sojourn_s,
    }


# ============================================================================
# VALIDITY TESTING — SAME CRITERIA AS ALL OTHER FEATURES
# ============================================================================

def test_criterion_validity(feature_df: pd.DataFrame,
                             exam_scores: dict,
                             out_dir: Path) -> pd.DataFrame:
    """
    Pearson and Spearman correlation of each Markov feature with exam score.
    Mann-Whitney U for top vs bottom quartile.
    Cohen's d effect size.
    Same criteria applied to all other sub-features in the project.
    """
    pids = list(feature_df.index)
    y    = np.array([exam_scores[p] for p in pids])
    q1   = np.percentile(y, 25)
    q3   = np.percentile(y, 75)
    hi   = y >= q3
    lo   = y <= q1

    rows = []
    for col in feature_df.columns:
        vals = feature_df[col].values
        r, p_r   = pearsonr(vals, y)
        sr, sp_r = spearmanr(vals, y)

        if hi.sum() >= 3 and lo.sum() >= 3:
            u, p_u = mannwhitneyu(vals[hi], vals[lo], alternative="two-sided")
            pooled = np.sqrt((vals[hi].std()**2 + vals[lo].std()**2) / 2 + 1e-9)
            d      = (vals[hi].mean() - vals[lo].mean()) / pooled
        else:
            p_u, d = 1.0, 0.0

        rows.append({
            "Feature":         col,
            "Pearson r":       round(float(r),   4),
            "Pearson p":       round(float(p_r), 4),
            "Spearman r":      round(float(sr),  4),
            "Spearman p":      round(float(sp_r),4),
            "Cohen's d":       round(float(d),   3),
            "MWU p":           round(float(p_u), 4),
            "Sig (r, p<0.05)": "YES *" if p_r  < 0.05 else "no",
            "Sig (MWU,p<0.05)":"YES *" if p_u  < 0.05 else "no",
            "Effect size":     ("large"  if abs(d) >= 0.8 else
                                "medium" if abs(d) >= 0.5 else
                                "small"  if abs(d) >= 0.2 else "negligible"),
        })

    df = pd.DataFrame(rows).sort_values("Cohen's d", key=abs, ascending=False)
    df.to_csv(out_dir / "markov_validity.csv", index=False)
    return df


def cronbach_alpha(df: pd.DataFrame) -> float:
    """Cronbach's alpha for a set of features."""
    n     = df.shape[1]
    if n < 2: return np.nan
    item_var  = df.var(axis=0, ddof=1).sum()
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0: return np.nan
    return float((n / (n-1)) * (1 - item_var / total_var))


def test_internal_consistency(feature_df: pd.DataFrame,
                               out_dir: Path) -> dict:
    """
    Cronbach's alpha and mean inter-feature correlation for stage-level
    and emotional-state Markov features separately.
    """
    groups = {
        "Stage Markov features (mc_s_*)":
            [c for c in feature_df.columns if c.startswith("mc_s_")],
        "Emotional Markov features (mc_e_*)":
            [c for c in feature_df.columns if c.startswith("mc_e_")],
    }
    results = {}
    for name, cols in groups.items():
        if len(cols) < 2:
            results[name] = {"alpha": np.nan, "mean_r": np.nan}
            continue
        sub   = feature_df[cols].dropna()
        alpha = cronbach_alpha(sub)
        corrs = sub.corr().values
        n     = len(corrs)
        off   = corrs[np.triu_indices(n, k=1)]
        results[name] = {
            "alpha":  round(float(alpha), 3),
            "mean_r": round(float(off.mean()), 3),
            "n_features": len(cols),
        }
        log.info("%s: alpha=%.3f  mean_r=%.3f", name, alpha, off.mean())
    return results


# ============================================================================
# PLOTS
# ============================================================================

def plot_transition_matrices(stage_matrices: dict,
                              emo_matrices:   dict,
                              exam_scores:    dict,
                              out_dir:        Path) -> None:
    """
    Average transition matrices for top vs bottom quartile scorers,
    both stage-level and emotional-state level.
    """
    pids   = sorted(stage_matrices.keys(),
                    key=lambda p: int(re.search(r"\d+",p).group()))
    y_arr  = np.array([exam_scores[p] for p in pids])
    q1, q3 = np.percentile(y_arr, 25), np.percentile(y_arr, 75)
    hi_idx = [i for i,y in enumerate(y_arr) if y >= q3]
    lo_idx = [i for i,y in enumerate(y_arr) if y <= q1]

    def _avg(matrices, idxs):
        mats = [matrices[pids[i]] for i in idxs if pids[i] in matrices]
        return np.mean(mats, axis=0) if mats else None

    # ── Stage-level ──────────────────────────────────────────────────────
    hi_s = _avg(stage_matrices, hi_idx)
    lo_s = _avg(stage_matrices, lo_idx)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, mat, title in [
        (axes[0], hi_s, f"High scorers (n={len(hi_idx)})"),
        (axes[1], lo_s, f"Low scorers  (n={len(lo_idx)})"),
        (axes[2], (hi_s - lo_s) if hi_s is not None and lo_s is not None
                  else np.zeros((N_STAGE_STATES, N_STAGE_STATES)),
         "Difference (High − Low)"),
    ]:
        if mat is None: ax.set_visible(False); continue
        cmap = "Blues" if "Diff" not in title else "RdBu"
        vmin = 0.0      if "Diff" not in title else -0.4
        vmax = 1.0      if "Diff" not in title else  0.4
        im   = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(N_STAGE_STATES))
        ax.set_xticklabels(STAGE_STATES, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(N_STAGE_STATES))
        ax.set_yticklabels(STAGE_STATES, fontsize=8)
        ax.set_xlabel("To stage", fontsize=9)
        ax.set_ylabel("From stage", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(N_STAGE_STATES):
            for j in range(N_STAGE_STATES):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if mat[i,j] > 0.6 else "black")
    fig.suptitle("Stage-Level Markov Transition Matrices — "
                 "High vs Low Exam Scorers",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "stage_transition_matrices.png",
                dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Stage transition matrices saved.")

    # ── Emotional-state level ─────────────────────────────────────────────
    hi_e = _avg(emo_matrices, hi_idx)
    lo_e = _avg(emo_matrices, lo_idx)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    EMO_NAMES = ["Calm", "Stressed", "Happy", "Mixed"]
    for ax, mat, title in [
        (axes[0], hi_e, f"High scorers (n={len(hi_idx)})"),
        (axes[1], lo_e, f"Low scorers  (n={len(lo_idx)})"),
        (axes[2], (hi_e - lo_e) if hi_e is not None and lo_e is not None
                  else np.zeros((N_EMO_STATES, N_EMO_STATES)),
         "Difference (High − Low)"),
    ]:
        if mat is None: ax.set_visible(False); continue
        cmap = "Blues" if "Diff" not in title else "RdBu"
        vmin = 0.0      if "Diff" not in title else -0.3
        vmax = 1.0      if "Diff" not in title else  0.3
        im   = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(N_EMO_STATES))
        ax.set_xticklabels(EMO_NAMES, fontsize=9)
        ax.set_yticks(range(N_EMO_STATES))
        ax.set_yticklabels(EMO_NAMES, fontsize=9)
        ax.set_xlabel("To state", fontsize=9)
        ax.set_ylabel("From state", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(N_EMO_STATES):
            for j in range(N_EMO_STATES):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if mat[i,j] > 0.6 else "black")
    fig.suptitle("Emotional-State Markov Transition Matrices — "
                 "High vs Low Exam Scorers",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "emotional_state_matrices.png",
                dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Emotional state matrices saved.")


def plot_validity_bars(validity_df: pd.DataFrame, out_dir: Path) -> None:
    """Cohen's d bar chart for all Markov features, colour-coded by significance."""
    df = validity_df.sort_values("Cohen's d", ascending=True)
    colors = [
        "#1A5276" if r["Sig (MWU,p<0.05)"] == "YES *" and r["Cohen's d"] > 0
        else "#922B21" if r["Sig (MWU,p<0.05)"] == "YES *" and r["Cohen's d"] < 0
        else "#BDC3C7"
        for _, r in df.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(10, max(5, len(df)*0.45)))
    bars = ax.barh(df["Feature"], df["Cohen's d"],
                   color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0,    color="black",  linewidth=0.8)
    ax.axvline( 0.5, color="#2ECC71", linewidth=0.8, linestyle="--",
               alpha=0.6, label="medium (d=0.5)")
    ax.axvline(-0.5, color="#2ECC71", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline( 0.8, color="#E74C3C", linewidth=0.8, linestyle=":",
               alpha=0.6, label="large  (d=0.8)")
    ax.axvline(-0.8, color="#E74C3C", linewidth=0.8, linestyle=":", alpha=0.6)

    for bar, (_, row) in zip(bars, df.iterrows()):
        if row["Sig (MWU,p<0.05)"] == "YES *":
            xpos = row["Cohen's d"]
            ax.text(xpos + (0.03 if xpos >= 0 else -0.03),
                    bar.get_y() + bar.get_height()/2,
                    f"*  {row['Effect size']}",
                    va="center",
                    ha="left" if xpos >= 0 else "right",
                    fontsize=8, fontweight="bold",
                    color="#1A5276" if xpos >= 0 else "#922B21")

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#1A5276", label="High > Low (sig.)"),
        mpatches.Patch(color="#922B21", label="Low > High (sig.)"),
        mpatches.Patch(color="#BDC3C7", label="Not significant"),
    ] + ax.get_legend_handles_labels()[0], fontsize=8, loc="lower right")

    ax.set_xlabel("Cohen's d  (top quartile vs bottom quartile)", fontsize=10)
    ax.set_title("Markov Feature Validity — Effect Size by Feature",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "markov_validity_bars.png",
                dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Validity bar chart saved.")


# ============================================================================
# MAIN
# ============================================================================

def main(data_dir: str) -> None:
    data_path = Path(data_dir)
    out_dir   = data_path / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract zips
    for zname, dname in [(STAGE_ZIP, STAGE_DIR), (FACIAL_ZIP, FACIAL_DIR)]:
        zp = data_path / zname; od = data_path / dname
        if zp.exists(): extract_zip(zp, od)

    stage_files  = find_files(data_path / STAGE_DIR)
    facial_files = find_files(data_path / FACIAL_DIR)
    exam_data    = load_exam(data_path / EXAM_FILE)

    common = sorted(
        set(stage_files) & set(facial_files) & set(exam_data),
        key=lambda p: int(re.search(r"\d+", p).group())
    )
    log.info("Participants: %d", len(common))

    all_features    = {}
    stage_matrices  = {}
    emo_matrices    = {}

    for pid in common:
        try:
            facial = load_facial(facial_files[pid])
            stage  = load_stage(stage_files[pid])
            if facial.empty or stage.empty: continue
            facial = compute_emotions(facial)
            # align stages to facial frames
            facial = facial.copy(); facial["stage"] = "Unknown"
            for _, row in stage.iterrows():
                mask = (facial["time_s"] >= row["start"]) & \
                       (facial["time_s"] <  row["end"])
                facial.loc[mask, "stage"] = row["stage"]
            facial = facial[np.isfinite(facial["time_s"].values)].copy()
            if len(facial) < 100: continue

            # ── Stage Markov ──────────────────────────────────────────────
            s_prob  = fit_stage_markov(stage)
            s_feats = extract_stage_markov_features(s_prob, stage)
            stage_matrices[pid] = s_prob

            # ── Emotional Markov ──────────────────────────────────────────
            e_prob, e_states = fit_emotional_markov(facial)
            e_feats          = extract_emotional_markov_features(e_prob, e_states)
            emo_matrices[pid] = e_prob

            all_features[pid] = {**s_feats, **e_feats}
            log.info("[%s] OK — %d stage transitions | %d emotion frames",
                     pid, len(stage)-1, len(facial))

        except Exception as ex:
            log.error("[%s] failed: %s", pid, ex)

    if len(all_features) < 5:
        log.error("Too few participants processed. Stopping."); return

    # ── Build feature matrix ──────────────────────────────────────────────
    feature_df = pd.DataFrame.from_dict(all_features, orient="index")
    feature_df.index.name = "participant"
    feature_df.to_csv(out_dir / "markov_features.csv")
    log.info("Feature matrix: %s", feature_df.shape)

    # ── Validity testing (same criteria as all other features) ────────────
    log.info("=" * 50)
    log.info("STEP 1: Criterion validity")
    log.info("=" * 50)
    validity_df = test_criterion_validity(feature_df, exam_data, out_dir)

    log.info("=" * 50)
    log.info("STEP 2: Internal consistency")
    log.info("=" * 50)
    consistency = test_internal_consistency(feature_df, out_dir)

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 3: Plots")
    log.info("=" * 50)
    plot_transition_matrices(stage_matrices, emo_matrices, exam_data, out_dir)
    plot_validity_bars(validity_df, out_dir)

    # ── Report ────────────────────────────────────────────────────────────
    sig_feats = validity_df[validity_df["Sig (MWU,p<0.05)"] == "YES *"]

    lines = [
        "=" * 66,
        "   MARKOV CHAIN FEATURE VALIDITY REPORT",
        "=" * 66,
        f"   Participants:        {len(all_features)}",
        f"   Stage Markov feats:  "
        f"{len([c for c in feature_df.columns if c.startswith('mc_s_')])}",
        f"   Emotion Markov feats:"
        f"{len([c for c in feature_df.columns if c.startswith('mc_e_')])}",
        "",
        "─" * 66,
        "   1. CRITERION VALIDITY (sorted by |Cohen's d|)",
        "─" * 66,
        f"   {'Feature':<30} {'r':>7} {'p':>7}  {'d':>7} {'Effect':>10}  Sig",
        "   " + "-"*64,
    ]
    for _, row in validity_df.iterrows():
        cd = row["Cohen's d"]
        lines.append(
            f"   {row['Feature']:<30} {row['Pearson r']:>7.3f} "
            f"{row['Pearson p']:>7.4f}  {cd:>7.3f} "
            f"{row['Effect size']:>10}  {row['Sig (MWU,p<0.05)']}"
        )

    lines += [
        "",
        "─" * 66,
        "   2. INTERNAL CONSISTENCY",
        "─" * 66,
    ]
    for name, stats in consistency.items():
        lines.append(
            f"   {name}")
        lines.append(
            f"     Cronbach alpha = {stats['alpha']}  "
            f"mean inter-r = {stats['mean_r']}  "
            f"n features = {stats['n_features']}"
        )
        interpretation = (
            "Good internal consistency (alpha >= 0.7)"
            if stats["alpha"] >= 0.7 else
            "Acceptable (alpha >= 0.6)"
            if stats["alpha"] >= 0.6 else
            "Poor internal consistency — features measure different things"
            if stats["alpha"] >= 0.0 else
            "Negative alpha — features are anti-correlated, do not group"
        )
        lines.append(f"     → {interpretation}")

    lines += [
        "",
        "─" * 66,
        "   3. RECOMMENDATION",
        "─" * 66,
    ]
    if len(sig_feats) == 0:
        lines.append("   No Markov features reach p<0.05 on MWU test.")
        lines.append("   Recommendation: report as null finding in paper.")
        lines.append("   Do NOT add to the feature matrix for regression.")
    else:
        lines.append(f"   {len(sig_feats)} feature(s) significant:")
        for _, row in sig_feats.iterrows():
            cd2 = row["Cohen's d"]
            lines.append(f"     {row['Feature']:<30} d={cd2:+.3f}  "
                         f"r={row['Pearson r']:+.3f}  p={row['Pearson p']:.4f}")
        lines.append("")
        lines.append("   Recommendation: add significant features to the")
        lines.append("   main feature matrix (ei_pipeline_v4.py) and re-run")
        lines.append("   the full pipeline to assess whether they improve")
        lines.append("   the composite EI score's criterion validity.")

    lines += [
        "",
        "─" * 66,
        "   OUTPUT FILES",
        "─" * 66,
        "   markov_features.csv          — all features per participant",
        "   markov_validity.csv          — full criterion validity table",
        "   stage_transition_matrices.png — stage Markov (hi vs lo)",
        "   emotional_state_matrices.png  — emotion Markov (hi vs lo)",
        "   markov_validity_bars.png      — effect size chart",
        "=" * 66,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(out_dir / "markov_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    log.info("All outputs → %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Markov Chain Feature Generation — Stage 2, Week 4"
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    main(args.data_dir)
