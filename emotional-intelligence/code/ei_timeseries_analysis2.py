"""
EI Time-Series Analysis & Visualisation
=========================================
Produces:
  1. Emotion trajectory plots — Stress + Happiness over task time
     with problem-solving stage bands, for high vs low scorers.
  2. Group-level statistical analysis — Mann-Whitney U tests,
     Cohen's d effect sizes, comparing top vs bottom quartile
     on every EI feature.
  3. Final EI score per participant derived from validated features
     (c3_block_score, c4_trend, c3_productive_ratio, c4_mean_valence)
     shown alongside trajectory plots.
  4. Summary: time-series makes the EI differences *visible*.

Results saved to: results_timeseries/

Usage:
    python ei_timeseries_analysis.py
    python ei_timeseries_analysis.py --data_dir "D:/A_emotions in exam/data"
"""

import re, zipfile, argparse, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

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
RESULTS_DIR      = "results_timeseries"

STAGE_ZIP  = "stage coded data-20260321T120851Z-1-001.zip"
FACIAL_ZIP = "facial expressions- data-20260321T182549Z-1-001.zip"
EXAM_FILE  = "Exam Scores.xlsx"
SAM_FILE   = "SAM Responses every 3 minutes.xlsx"
STAGE_DIR  = "stage_coded_data"
FACIAL_DIR = "facial_expressions"

FPS = 25

# Emotion composites
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

# Stage colour palette for background bands
STAGE_COLORS = {
    "Preparation": "#D4E8FF",
    "Ideation":    "#FFE8C4",
    "Production":  "#C4F0C4",
    "Evaluation":  "#E8D4FF",
    "Refinement":  "#C4F0F0",
    "Block":       "#FFCCCC",
    "SAM":         "#F0F0F0",
    "Unknown":     "#FAFAFA",
}

# EI score: validated 4-feature parsimonious set from v3 results
EI_FEATURES = ["c3_block_score","c4_trend","c3_productive_ratio","c4_mean_valence"]
EI_WEIGHTS  = [0.40, 0.25, 0.20, 0.15]   # block dominates per SHAP

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def extract_zip(zip_path, out_dir):
    if out_dir.exists() and any(out_dir.iterdir()):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s ...", zip_path.name)
    with zipfile.ZipFile(zip_path,"r") as z:
        z.extractall(out_dir)

def participant_id(fn):
    m = re.search(r"(P\d+)", fn, re.IGNORECASE)
    return m.group(1).upper() if m else None

def find_files(directory):
    mapping = {}
    for f in Path(directory).rglob("*"):
        if f.suffix.lower() in {".csv",".xlsx",".xls"}:
            pid = participant_id(f.name)
            if pid:
                mapping.setdefault(pid, f)
    return mapping

def clean_stage(raw):
    if pd.isna(raw): return "Unknown"
    return STAGE_MAP.get(str(raw).strip().lower(), str(raw).strip())

def load_facial(path):
    df = pd.read_csv(path) if path.suffix==".csv" else pd.read_excel(path)
    df.columns = df.columns.str.strip()
    if "cleaned time" in df.columns:
        df = df.rename(columns={"cleaned time":"time_s"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp":"time_s"})
    r_cols = [c for c in df.columns if c.endswith("_r")]
    keep   = ["time_s","confidence","success"] + r_cols
    df     = df[[c for c in keep if c in df.columns]].copy()
    if "confidence" in df.columns:
        df = df[df["confidence"] >= 0.80]
    if "success" in df.columns:
        df = df[df["success"] == 1]
    return df.reset_index(drop=True)

def load_stage(path):
    df = pd.read_excel(path) if path.suffix in {".xlsx",".xls"} \
         else pd.read_csv(path)
    df.columns = df.columns.str.strip()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "start" in cl and "start" not in col_map: col_map["start"] = c
        if "end"   in cl and "end"   not in col_map: col_map["end"]   = c
        if ("code" in cl or "stage" in cl) and "stage" not in col_map:
            col_map["stage"] = c
    df = df.rename(columns={v:k for k,v in col_map.items()})
    for col in ["start","end"]:
        if col not in df.columns: continue
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
        elif df[col].dtype == object:
            def _sec(v):
                try:
                    parts = str(v).split(":")
                    return sum(float(p)*60**(len(parts)-i-1)
                               for i,p in enumerate(parts))
                except: return np.nan
            df[col] = df[col].apply(_sec)
    if "stage" not in df.columns: return pd.DataFrame()
    df["stage"] = df["stage"].apply(clean_stage)
    df = df[df["stage"]!="SAM"].copy()
    return df.dropna(subset=["start","end","stage"]).reset_index(drop=True)

def load_exam(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    scores = {}
    for _, row in df.iterrows():
        m = re.search(r"P(\d+)", str(row.iloc[0]).upper())
        if not m: continue
        try:
            scores[f"P{m.group(1)}"] = float(row.iloc[1]) / \
                (float(row.iloc[2]) if len(row)>2 else 26.0)
        except: pass
    return scores

def load_sam(path):
    df = pd.read_excel(path, header=None)
    hrow = 0
    for i, row in df.iterrows():
        if any("participant" in str(v).lower() for v in row.values):
            hrow = i; break
    df.columns = df.iloc[hrow]
    df = df.iloc[hrow+1:].reset_index(drop=True)
    result = {}
    for _, row in df.iterrows():
        m = re.search(r"P(\d+)", str(row.iloc[0]).upper())
        if not m: continue
        pid = f"P{m.group(1)}"
        vals = row.iloc[1:].tolist()
        def _safe(v):
            try: return float(v) if not pd.isna(v) else np.nan
            except: return np.nan
        result[pid] = {
            "valence": [_safe(v) for v in vals[0::2]],
            "arousal": [_safe(v) for v in vals[1::2]],
        }
    return result

# ============================================================================
# EMOTION COMPUTATION + ALIGNMENT
# ============================================================================

def compute_emotions(facial):
    for em, aus in EMOTION_AU.items():
        avail = [a for a in aus if a in facial.columns]
        facial[em] = facial[avail].sum(axis=1) if avail else 0.0
    return facial

def align_stages(facial, stage):
    facial = facial.copy()
    facial["stage"] = "Unknown"
    for _, row in stage.iterrows():
        mask = (facial["time_s"] >= row["start"]) & \
               (facial["time_s"] <  row["end"])
        facial.loc[mask,"stage"] = row["stage"]
    return facial

def smooth(series, window=75, poly=3):
    """Savitzky-Golay smoothing — window ~3 seconds at 25fps."""
    n = len(series)
    w = min(window, n//2*2+1)
    if w % 2 == 0: w += 1
    w = max(5, w)
    if n < w: return series
    return savgol_filter(series, w, poly)

# ============================================================================
# EI FEATURE EXTRACTION (parsimonious validated set)
# ============================================================================

def _norm(v, lo, hi):
    return float(np.clip((v-lo)/(hi-lo+1e-9), 0, 1))

def extract_ei_features(fused, stage_df, sam_valence, sam_arousal):
    """Extract the 4 validated EI features."""
    total = stage_df["end"].max() - stage_df["start"].min() + 1e-9

    # c3_block_score
    b_rows = stage_df[stage_df["stage"].isin(BLOCK_STAGES)]
    block_time = (b_rows["end"] - b_rows["start"]).sum()
    block_score = _norm(1 - block_time/total, 0.8, 1.0)

    # c3_productive_ratio
    p_rows = stage_df[stage_df["stage"].isin(PRODUCTIVE_STAGES)]
    prod_time = (p_rows["end"] - p_rows["start"]).sum()
    productive_ratio = _norm(prod_time/total, 0.3, 0.7)

    # c4_trend — SAM valence slope
    def _fill(lst):
        arr = np.array(lst, dtype=float)
        if np.all(np.isnan(arr)): return np.full(len(arr), 3.0)
        arr[np.isnan(arr)] = np.nanmean(arr)
        return arr
    v = _fill(sam_valence)
    slope, *_ = np.polyfit(range(len(v)), v, 1) if len(v)>1 else [0]
    trend = _norm(slope, -0.5, 0.5)

    # c4_mean_valence
    mean_val = _norm(v.mean(), 1, 5)

    return {
        "c3_block_score":      block_score,
        "c3_productive_ratio": productive_ratio,
        "c4_trend":            trend,
        "c4_mean_valence":     mean_val,
    }

def compute_ei_score(features: dict) -> float:
    """Weighted EI score 0-100 from 4 validated features."""
    raw = sum(EI_WEIGHTS[i] * features[f]
              for i, f in enumerate(EI_FEATURES))
    return round(raw * 100, 1)

# ============================================================================
# STATISTICAL ANALYSIS — GROUP COMPARISON
# ============================================================================

def group_statistical_analysis(feature_df, y, out_dir):
    """
    Split participants into top and bottom quartile by exam score.
    Run Mann-Whitney U test on every feature.
    Compute Cohen's d effect size.
    """
    q1  = np.percentile(y, 25)
    q3  = np.percentile(y, 75)
    hi  = y >= q3
    lo  = y <= q1

    rows = []
    for col in feature_df.columns:
        vals_hi = feature_df.loc[hi, col].values
        vals_lo = feature_df.loc[lo, col].values
        if len(vals_hi) < 3 or len(vals_lo) < 3:
            continue
        u_stat, p_val = mannwhitneyu(vals_hi, vals_lo,
                                     alternative="two-sided")
        # Cohen's d
        pooled_std = np.sqrt(
            (vals_hi.std()**2 + vals_lo.std()**2) / 2 + 1e-9
        )
        d = (vals_hi.mean() - vals_lo.mean()) / pooled_std

        # Direction: positive d = high scorers show MORE of this feature
        direction = "high > low" if d > 0 else "low > high"
        magnitude = (
            "large"  if abs(d) >= 0.8 else
            "medium" if abs(d) >= 0.5 else
            "small"  if abs(d) >= 0.2 else
            "negligible"
        )
        rows.append({
            "Feature":       col,
            "Mean (hi Q)":   round(vals_hi.mean(), 4),
            "Mean (lo Q)":   round(vals_lo.mean(), 4),
            "Mann-Whitney U":round(u_stat, 1),
            "p (raw)":       round(p_val, 4),
            "Significant":   "YES *" if p_val < 0.05 else "no",
            "Cohen's d":     round(d, 3),
            "Effect size":   magnitude,
            "Direction":     direction,
        })

    df = pd.DataFrame(rows).sort_values("Cohen's d", key=abs, ascending=False)
    df.to_csv(out_dir / "group_comparison_stats.csv", index=False)

    log.info("Group comparison (top vs bottom quartile):")
    log.info("  Hi-scorers: n=%d  (exam >= %.2f)", hi.sum(), q3)
    log.info("  Lo-scorers: n=%d  (exam <= %.2f)", lo.sum(), q1)
    for _, row in df[df["Significant"]=="YES *"].iterrows():
        log.info("  SIGNIFICANT: %-28s  d=%+.2f  (%s)  p=%.4f",
                 row["Feature"], row["Cohen's d"],
                 row["Effect size"], row["p (raw)"])
    return df, hi, lo

# ============================================================================
# TIME-SERIES TRAJECTORY PLOT — SINGLE PARTICIPANT
# ============================================================================

def plot_participant_trajectory(pid, fused, stage_df, ei_score,
                                 exam_score, ax_stress, ax_happy,
                                 title_prefix=""):
    """
    Plot Stress and Happiness smoothed trajectories with stage colour bands
    on two vertically stacked axes for one participant.
    """
    # ── Sanitise: drop rows where time_s is NaN/Inf ──────────────────────
    fused = fused[np.isfinite(fused["time_s"].values)].copy()
    if len(fused) < 10:
        for ax in [ax_stress, ax_happy]:
            ax.text(0.5, 0.5, f"{pid}\nInsufficient data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
        return

    t      = fused["time_s"].values / 60.0        # minutes
    stress = smooth(fused["Stress"].values)
    happy  = smooth(fused["Happiness"].values)

    # Guard against all-NaN after smoothing
    if not np.isfinite(t).all() or not np.isfinite(stress).all():
        for ax in [ax_stress, ax_happy]:
            ax.text(0.5, 0.5, f"{pid}\nData quality issue",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
        return

    t_min, t_max = float(t.min()), float(t.max())
    if t_min == t_max:
        t_max = t_min + 30.0   # fallback: 30 minutes

    # Draw stage background bands
    sorted_s = stage_df.sort_values("start")
    for _, row in sorted_s.iterrows():
        t0  = max(float(row["start"]) / 60.0, t_min)
        t1  = min(float(row["end"])   / 60.0, t_max)
        if t1 <= t0:
            continue
        col = STAGE_COLORS.get(row["stage"], "#FAFAFA")
        for ax in [ax_stress, ax_happy]:
            ax.axvspan(t0, t1, facecolor=col, alpha=0.45, linewidth=0)

    # Trajectories
    ax_stress.plot(t, stress, color="#C0392B", linewidth=1.6, label="Stress")
    ax_stress.fill_between(t, 0, stress, color="#C0392B", alpha=0.15)
    ax_stress.set_xlim(t_min, t_max)
    ax_stress.set_ylim(0, max(stress.max() * 1.1, 0.1))
    ax_stress.set_ylabel("Stress intensity", fontsize=8)
    ax_stress.tick_params(labelsize=7)
    ax_stress.set_title(
        f"{title_prefix} {pid}\nEI score: {ei_score:.1f}/100 | "
        f"Exam: {exam_score*26:.0f}/26 ({exam_score*100:.0f}%)",
        fontsize=9, fontweight="bold", pad=4
    )

    ax_happy.plot(t, happy, color="#27AE60", linewidth=1.6, label="Happiness")
    ax_happy.fill_between(t, 0, happy, color="#27AE60", alpha=0.15)
    ax_happy.set_xlim(t_min, t_max)
    ax_happy.set_ylim(0, max(happy.max() * 1.1, 0.1))
    ax_happy.set_ylabel("Happiness intensity", fontsize=8)
    ax_happy.set_xlabel("Task time (minutes)", fontsize=8)
    ax_happy.tick_params(labelsize=7)

    # Annotate Block episodes on stress axis
    stress_ymax = float(ax_stress.get_ylim()[1])
    for _, row in sorted_s[sorted_s["stage"].isin(BLOCK_STAGES)].iterrows():
        t_mid = (float(row["start"]) + float(row["end"])) / 2 / 60.0
        if t_min <= t_mid <= t_max:
            ax_stress.axvline(t_mid, color="#C0392B", linewidth=0.8,
                              linestyle=":", alpha=0.6)
            ax_stress.text(t_mid, stress_ymax * 0.85, "B",
                           ha="center", fontsize=6, color="#922B21",
                           fontweight="bold")


def build_stage_legend():
    patches = [
        mpatches.Patch(facecolor=STAGE_COLORS[s], alpha=0.6, label=s)
        for s in ["Preparation","Ideation","Production",
                  "Evaluation","Refinement","Block"]
    ]
    return patches

# ============================================================================
# MAIN TRAJECTORY FIGURE — HIGH vs LOW SCORERS
# ============================================================================

def plot_trajectory_comparison(participant_data: dict,
                                 pids_hi: list, pids_lo: list,
                                 out_dir: Path) -> None:
    """
    Side-by-side trajectory plots: top 3 scorers (left) vs bottom 3 (right).
    Each participant gets two rows: Stress on top, Happiness on bottom.
    """
    n_show  = min(3, len(pids_hi), len(pids_lo))
    n_rows  = n_show * 2          # 2 emotion rows per participant
    n_cols  = 2                   # left=high, right=low

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 3.5 * n_show),
        squeeze=False,
        gridspec_kw={"hspace": 0.08, "wspace": 0.25}
    )

    for col_idx, (pids, label) in enumerate(
        [(pids_hi[:n_show], "High scorer"), (pids_lo[:n_show], "Low scorer")]
    ):
        for row_group, pid in enumerate(pids):
            if pid not in participant_data:
                continue
            d = participant_data[pid]
            row_stress = row_group * 2
            row_happy  = row_group * 2 + 1

            plot_participant_trajectory(
                pid,
                d["fused"], d["stage"],
                d["ei_score"], d["exam_score"],
                axes[row_stress][col_idx],
                axes[row_happy][col_idx],
                title_prefix=label,
            )

            # Only show x-label on bottom row of each group
            if row_group < n_show - 1:
                axes[row_happy][col_idx].set_xlabel("")

    # Column headers
    axes[0][0].set_title(
        f"HIGH SCORERS\n{axes[0][0].get_title()}",
        fontsize=10, fontweight="bold", color="#1A5276", pad=6
    )
    axes[0][1].set_title(
        f"LOW SCORERS\n{axes[0][1].get_title()}",
        fontsize=10, fontweight="bold", color="#922B21", pad=6
    )

    # Legend
    fig.legend(
        handles=build_stage_legend(),
        loc="lower center", ncol=6, fontsize=8,
        title="Problem-solving stage (background colour)",
        title_fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
        framealpha=0.9,
    )
    fig.suptitle(
        "Emotion Trajectories: High vs Low EI Scorers\n"
        "(Smoothed Stress & Happiness over 30-minute task; "
        "B = Block episode)",
        fontsize=11, fontweight="bold", y=1.01
    )

    fig.savefig(out_dir / "trajectory_high_vs_low.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Trajectory comparison plot saved.")


# ============================================================================
# STATISTICAL VISUALISATION — EFFECT SIZE PLOT
# ============================================================================

def plot_effect_sizes(group_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Horizontal bar chart of Cohen's d for every feature,
    colour-coded by magnitude and significance.
    """
    df = group_df.sort_values("Cohen's d", ascending=True)
    colors = []
    for _, row in df.iterrows():
        if row["Significant"] == "YES *":
            colors.append("#1A5276" if row["Cohen's d"] > 0 else "#922B21")
        else:
            colors.append("#BDC3C7")

    fig, ax = plt.subplots(figsize=(10, max(6, len(df)*0.45)))
    bars = ax.barh(df["Feature"], df["Cohen's d"],
                   color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0,    color="black",  linewidth=0.8)
    ax.axvline( 0.5, color="#2ECC71", linewidth=0.8, linestyle="--",
               alpha=0.6, label="medium effect (d=±0.5)")
    ax.axvline(-0.5, color="#2ECC71", linewidth=0.8, linestyle="--",
               alpha=0.6)
    ax.axvline( 0.8, color="#E74C3C", linewidth=0.8, linestyle=":",
               alpha=0.6, label="large effect (d=±0.8)")
    ax.axvline(-0.8, color="#E74C3C", linewidth=0.8, linestyle=":",
               alpha=0.6)

    # Annotate significant bars
    for bar, (_, row) in zip(bars, df.iterrows()):
        if row["Significant"] == "YES *":
            ax.text(
                row["Cohen's d"] + (0.03 if row["Cohen's d"] >= 0 else -0.03),
                bar.get_y() + bar.get_height()/2,
                f"*  {row['Effect size']}",
                va="center", ha="left" if row["Cohen's d"] >= 0 else "right",
                fontsize=8, color="#1A5276" if row["Cohen's d"]>0 else "#922B21",
                fontweight="bold",
            )

    blue_patch  = mpatches.Patch(color="#1A5276",
                                  label="High scorers > Low scorers (sig.)")
    red_patch   = mpatches.Patch(color="#922B21",
                                  label="Low scorers > High scorers (sig.)")
    gray_patch  = mpatches.Patch(color="#BDC3C7", label="Not significant")
    ax.legend(handles=[blue_patch, red_patch, gray_patch] +
              ax.get_legend_handles_labels()[0],
              loc="lower right", fontsize=8)

    ax.set_xlabel("Cohen's d  (top quartile vs bottom quartile)", fontsize=10)
    ax.set_title("Effect Size by Feature — High vs Low Exam Scorers",
                 fontweight="bold", fontsize=11)
    ax.set_xlim(min(df["Cohen's d"].min()-0.2, -1.1),
                max(df["Cohen's d"].max()+0.2,  1.1))
    plt.tight_layout()
    fig.savefig(out_dir / "effect_sizes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Effect size plot saved.")


# ============================================================================
# EI SCORE DISTRIBUTION PLOT
# ============================================================================

def plot_ei_distribution(ei_scores: dict, exam_scores: dict,
                          out_dir: Path) -> None:
    """
    Scatter: EI score vs exam score, with participant labels.
    Quadrant lines at median EI and median exam.
    """
    pids = sorted(ei_scores.keys(),
                  key=lambda p: int(re.search(r"\d+", p).group()))
    ei   = np.array([ei_scores[p] for p in pids])
    ex   = np.array([exam_scores[p] for p in pids]) * 100  # percent

    fig, ax = plt.subplots(figsize=(10, 7))

    # Quadrant shading
    med_ei = np.median(ei)
    med_ex = np.median(ex)
    ax.axhline(med_ex, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(med_ei, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Colour by quadrant
    colors_q = []
    for e, x in zip(ei, ex):
        if   e >= med_ei and x >= med_ex: colors_q.append("#1A5276")  # high EI, high exam
        elif e <  med_ei and x <  med_ex: colors_q.append("#922B21")  # low EI, low exam
        else:                              colors_q.append("#7D6608")  # inconsistent

    sc = ax.scatter(ei, ex, c=colors_q, s=80, edgecolors="black",
                    linewidths=0.5, zorder=3)

    # Label each point
    for pid, e, x, c in zip(pids, ei, ex, colors_q):
        ax.annotate(pid, (e, x), textcoords="offset points",
                    xytext=(5, 3), fontsize=6.5, color=c)

    # Trend line
    m, b = np.polyfit(ei, ex, 1)
    xl   = np.linspace(ei.min(), ei.max(), 100)
    ax.plot(xl, m*xl+b, "k--", linewidth=1.5, label="trend")

    r, p = pearsonr(ei, ex)
    ax.set_xlabel("EI Score (0–100)", fontsize=11)
    ax.set_ylabel("Exam Score (%)", fontsize=11)
    ax.set_title(
        f"EI Score vs Examination Performance\n"
        f"Pearson r = {r:.3f}  (p = {p:.4f})",
        fontsize=12, fontweight="bold"
    )

    # Quadrant labels
    ax.text(med_ei + 0.5, med_ex + 1, "High EI\nHigh Exam", fontsize=8,
            color="#1A5276", ha="left", va="bottom", alpha=0.7)
    ax.text(ei.min(), med_ex - 3, "Low EI\nLow Exam", fontsize=8,
            color="#922B21", ha="left", va="top", alpha=0.7)

    hi_patch   = mpatches.Patch(color="#1A5276", label="High EI + High Exam")
    lo_patch   = mpatches.Patch(color="#922B21", label="Low EI  + Low Exam")
    mix_patch  = mpatches.Patch(color="#7D6608", label="Inconsistent quadrant")
    ax.legend(handles=[hi_patch, lo_patch, mix_patch], fontsize=9,
              loc="upper left")

    plt.tight_layout()
    fig.savefig(out_dir / "ei_score_vs_exam.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    log.info("EI score scatter saved.")


# ============================================================================
# GROUP MEAN TRAJECTORY PLOT
# ============================================================================

def plot_group_mean_trajectories(participant_data: dict,
                                  pids_hi: list, pids_lo: list,
                                  out_dir: Path) -> None:
    """
    Average Stress and Happiness trajectories for top vs bottom quartile,
    resampled to 300 equal time steps, with ±1 std shading.
    This makes the group-level EI difference directly visible.
    """
    def _resample(series, n=300):
        x_old = np.linspace(0, 1, len(series))
        x_new = np.linspace(0, 1, n)
        return np.interp(x_new, x_old, series)

    def _collect(pids, emotion):
        out = []
        for pid in pids:
            if pid not in participant_data: continue
            fused = participant_data[pid]["fused"]
            # Drop NaN time rows
            fused = fused[np.isfinite(fused["time_s"].values)]
            if len(fused) < 10: continue
            s = smooth(fused[emotion].values)
            if not np.isfinite(s).all(): continue
            out.append(_resample(s))
        return np.array(out)

    n_steps = 300
    t_axis  = np.linspace(0, 30, n_steps)   # minutes

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, emotion, color_hi, color_lo, label in [
        (axes[0], "Stress",    "#C0392B", "#E8A89C", "Stress intensity"),
        (axes[1], "Happiness", "#1A5276", "#8FB3CE", "Happiness intensity"),
    ]:
        hi_mat = _collect(pids_hi, emotion)
        lo_mat = _collect(pids_lo, emotion)

        if len(hi_mat) > 0:
            hi_mean = hi_mat.mean(axis=0)
            hi_std  = hi_mat.std(axis=0)
            ax.plot(t_axis, hi_mean, color=color_hi, linewidth=2.2,
                    label=f"High scorers (n={len(hi_mat)})")
            ax.fill_between(t_axis, hi_mean-hi_std, hi_mean+hi_std,
                            color=color_hi, alpha=0.15)

        if len(lo_mat) > 0:
            lo_mean = lo_mat.mean(axis=0)
            lo_std  = lo_mat.std(axis=0)
            ax.plot(t_axis, lo_mean, color=color_lo, linewidth=2.2,
                    linestyle="--", label=f"Low scorers (n={len(lo_mat)})")
            ax.fill_between(t_axis, lo_mean-lo_std, lo_mean+lo_std,
                            color=color_lo, alpha=0.15)

        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, None)
        ax.grid(axis="x", alpha=0.3)

    axes[1].set_xlabel("Task time (minutes)", fontsize=10)
    fig.suptitle(
        "Mean Emotion Trajectories — Top vs Bottom Quartile Scorers\n"
        "(shading = ±1 std; smoothed with Savitzky-Golay filter)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(out_dir / "group_mean_trajectories.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Group mean trajectories saved.")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def write_report(ei_scores, exam_scores, group_df,
                 pids_hi, pids_lo, out_dir):
    pids = sorted(ei_scores.keys(),
                  key=lambda p: int(re.search(r"\d+", p).group()))
    ei_arr = np.array([ei_scores[p]      for p in pids])
    ex_arr = np.array([exam_scores[p]*100 for p in pids])
    r, p   = pearsonr(ei_arr, ex_arr)
    sr, sp = spearmanr(ei_arr, ex_arr)

    sig_feats = group_df[group_df["Significant"]=="YES *"]

    L = [
        "=" * 66,
        "   EI TIME-SERIES ANALYSIS — RESULTS REPORT",
        "=" * 66,
        f"   Participants analysed:  {len(pids)}",
        f"   EI score formula:       "
        f"0.40×block + 0.25×trend + 0.20×productive + 0.15×valence",
        "",
        "─" * 66,
        "   1. EI SCORE SUMMARY",
        "─" * 66,
        f"   Mean EI score:   {ei_arr.mean():.1f} / 100",
        f"   Std EI score:    {ei_arr.std():.1f}",
        f"   Min / Max:       {ei_arr.min():.1f} / {ei_arr.max():.1f}",
        f"   Top 5 EI:  "
        + str([(p, round(s,1)) for s,p in
               sorted(zip(ei_arr,pids), reverse=True)[:5]]),
        f"   Bottom 5:  "
        + str([(p, round(s,1)) for s,p in
               sorted(zip(ei_arr,pids))[:5]]),
        "",
        "─" * 66,
        "   2. CORRELATION: EI SCORE vs EXAM PERFORMANCE",
        "─" * 66,
        f"   Pearson  r = {r:.4f}  (p = {p:.4f})",
        f"   Spearman r = {sr:.4f}  (p = {sp:.4f})",
        "",
        "─" * 66,
        "   3. GROUP COMPARISON  (top vs bottom exam quartile)",
        "─" * 66,
        f"   High scorers: {len(pids_hi)} participants",
        f"   Low scorers:  {len(pids_lo)} participants",
        "",
        f"   {'Feature':<30} {'d':>7} {'Effect':>10} {'p':>8}",
        "   " + "-"*58,
    ]
    for _, row in sig_feats.iterrows():
        cohens_d = row["Cohen's d"]
        L.append(f"   {row['Feature']:<30} {cohens_d:>7.3f} "
                 f"{row['Effect size']:>10}  {row['p (raw)']:>7.4f} *")
    if len(sig_feats) == 0:
        L.append("   No features reach p<0.05 (small group sizes).")
    L += [
        "",
        "─" * 66,
        "   4. OUTPUT FILES",
        "─" * 66,
        "   trajectory_high_vs_low.png — individual traces, high vs low",
        "   group_mean_trajectories.png — group averages ±std",
        "   effect_sizes.png            — Cohen's d per feature",
        "   ei_score_vs_exam.png        — EI score scatter",
        "   ei_scores_all.csv           — EI score per participant",
        "   group_comparison_stats.csv  — full statistical table",
        "=" * 66,
    ]
    report = "\n".join(L)
    print("\n" + report)
    with open(out_dir / "timeseries_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Report saved → timeseries_report.txt")


# ============================================================================
# MAIN
# ============================================================================

def main(data_dir: str) -> None:
    data_path = Path(data_dir)
    out_dir   = data_path / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract zips
    for zname, dname in [(STAGE_ZIP, STAGE_DIR), (FACIAL_ZIP, FACIAL_DIR)]:
        zp = data_path / zname
        od = data_path / dname
        if zp.exists():
            extract_zip(zp, od)

    stage_files  = find_files(data_path / STAGE_DIR)
    facial_files = find_files(data_path / FACIAL_DIR)
    exam_data    = load_exam(data_path / EXAM_FILE)
    sam_data     = load_sam(data_path / SAM_FILE)

    common = sorted(
        set(stage_files) & set(facial_files) &
        set(exam_data)   & set(sam_data),
        key=lambda p: int(re.search(r"\d+", p).group())
    )
    log.info("Participants in all sources: %d", len(common))

    # ── Process all participants ──────────────────────────────────────────
    participant_data = {}
    ei_scores        = {}
    feature_rows     = []

    for pid in common:
        try:
            facial = load_facial(facial_files[pid])
            stage  = load_stage(stage_files[pid])
            if facial.empty or stage.empty:
                continue
            facial = compute_emotions(facial)
            fused  = align_stages(facial, stage)

            sam     = sam_data[pid]
            feats   = extract_ei_features(
                fused, stage, sam["valence"], sam["arousal"]
            )
            ei_score = compute_ei_score(feats)

            participant_data[pid] = {
                "fused":      fused,
                "stage":      stage,
                "ei_score":   ei_score,
                "exam_score": exam_data[pid],
            }
            ei_scores[pid] = ei_score
            feats["participant"] = pid
            feature_rows.append(feats)
            log.info("[%s] EI=%.1f  Exam=%d/26", pid,
                     ei_score, round(exam_data[pid]*26))

        except Exception as e:
            log.error("[%s] failed: %s", pid, e)

    # ── Feature matrix + EI scores ────────────────────────────────────────
    feature_df = pd.DataFrame(feature_rows).set_index("participant")
    ei_df      = pd.DataFrame(
        {"EI_score": ei_scores,
         "Exam_pct": {p: v*100 for p, v in exam_data.items()
                      if p in ei_scores}}
    )
    ei_df.to_csv(out_dir / "ei_scores_all.csv")

    # ── Sort participants by exam score ───────────────────────────────────
    y    = np.array([exam_data[p] for p in feature_df.index])
    pids = list(feature_df.index)

    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    pids_hi = [p for p, s in zip(pids, y) if s >= q3]
    pids_lo = [p for p, s in zip(pids, y) if s <= q1]

    # Sort by score for representative selection
    pids_hi = sorted(pids_hi, key=lambda p: exam_data[p], reverse=True)
    pids_lo = sorted(pids_lo, key=lambda p: exam_data[p])

    log.info("High scorers (Q3+): %s", pids_hi)
    log.info("Low  scorers (Q1-): %s", pids_lo)

    # ── Step 1: Statistical analysis ──────────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 1: Group statistical analysis")
    log.info("=" * 50)
    group_df, hi_mask, lo_mask = group_statistical_analysis(
        feature_df, y, out_dir
    )

    # ── Step 2: Individual trajectory plot ────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 2: Individual trajectory plots")
    log.info("=" * 50)
    plot_trajectory_comparison(
        participant_data, pids_hi, pids_lo, out_dir
    )

    # ── Step 3: Group mean trajectory ────────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 3: Group mean trajectories")
    log.info("=" * 50)
    plot_group_mean_trajectories(
        participant_data, pids_hi, pids_lo, out_dir
    )

    # ── Step 4: Effect size plot ──────────────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 4: Effect size visualisation")
    log.info("=" * 50)
    plot_effect_sizes(group_df, out_dir)

    # ── Step 5: EI score scatter ──────────────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 5: EI score vs exam scatter")
    log.info("=" * 50)
    plot_ei_distribution(ei_scores, exam_data, out_dir)

    # ── Report ────────────────────────────────────────────────────────────
    write_report(ei_scores, exam_data, group_df,
                 pids_hi, pids_lo, out_dir)

    log.info("All outputs → %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EI Time-Series Analysis & Visualisation"
    )
    parser.add_argument(
        "--data_dir", default=DEFAULT_DATA_DIR,
        help='Root data directory'
    )
    args = parser.parse_args()
    main(args.data_dir)
