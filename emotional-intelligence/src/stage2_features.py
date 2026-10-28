"""
Stage 2 — Comprehensive Candidate Feature Extraction
======================================================
Extracts all 116 candidate features across 5 EI dimensions from behavioral
data ONLY (facial AU, stage annotations, SAM ratings).
NO exam scores loaded or used here.

Outputs:
  outputs/feature_matrix_raw.csv / .pkl  — shape (42, 116)
"""

import os, pickle, warnings
import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings("ignore")

from config import (
    OUTPUT_DIR, EMOTION_COLS, POSITIVE_EMOS, NEGATIVE_EMOS,
    BLOCK_STAGE, PRODUCTIVE_STAGES, CORE_PRODUCTIVE, AU_COLS,
    SAM_CHECKPOINT_TIMES_SEC, ALL_CANDIDATES, CANDIDATE_FEATURES
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_pearsonr(x, y):
    """Pearson r; returns 0 if insufficient data or constant."""
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3:
        return 0.0
    xv, yv = x[valid], y[valid]
    if np.std(xv) < 1e-10 or np.std(yv) < 1e-10:
        return 0.0
    r, _ = pearsonr(xv, yv)
    return float(r) if not np.isnan(r) else 0.0

def _stage_mask(face_df, stage_name):
    return face_df["stage"] == stage_name

def _stage_masks(face_df, stage_list):
    return face_df["stage"].isin(stage_list)

def _downsample_1hz(series: pd.Series, fps: float = 25.0) -> np.ndarray:
    """Downsample a time series to ~1 Hz by taking every fps-th frame."""
    step = max(1, int(round(fps)))
    return series.iloc[::step].values.astype(float)

def _ar1(arr: np.ndarray) -> float:
    """Lag-1 autocorrelation (AR1) of a 1D array."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 4:
        return 0.0
    # Detrend first
    arr = arr - np.mean(arr)
    if np.std(arr) < 1e-10:
        return 0.0
    r = np.corrcoef(arr[:-1], arr[1:])[0, 1]
    return float(r) if not np.isnan(r) else 0.0

def _mssd(arr: np.ndarray) -> float:
    """Mean Squared Successive Differences."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    diffs = np.diff(arr)
    return float(np.mean(diffs ** 2))

def _icc_1_way(data: np.ndarray) -> float:
    """
    One-way random effects ICC(1,1).
    data shape: (n_epochs, n_channels).
    Returns 1 - ICC so that high differentiation (low ICC) → high value.
    """
    if data.shape[0] < 2 or data.shape[1] < 2:
        return 0.0
    n, k = data.shape
    grand_mean = data.mean()
    ss_between = k * np.sum((data.mean(axis=1) - grand_mean) ** 2)
    ss_within  = np.sum((data - data.mean(axis=1, keepdims=True)) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within  = ss_within  / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 0.0
    return float(np.clip(1.0 - icc, 0, 1))   # 1-ICC = differentiation

def _stage_durations(stage_df: pd.DataFrame) -> pd.Series:
    stage_df = stage_df.copy()
    stage_df["dur"] = stage_df["end_sec"] - stage_df["start_sec"]
    return stage_df.groupby("stage")["dur"].sum()

def _markov_matrix(stage_df: pd.DataFrame):
    stages = stage_df["stage"].tolist()
    states = sorted(set(stages))
    idx    = {s: i for i, s in enumerate(states)}
    n      = len(states)
    counts = np.zeros((n, n))
    for a, b in zip(stages[:-1], stages[1:]):
        counts[idx[a], idx[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = counts / row_sums
    return T, states

def _stationary(T: np.ndarray, states: list):
    n = T.shape[0]
    if n < 2:
        return np.ones(n) / n
    A = (T.T - np.eye(n))
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1); b[-1] = 1.0
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.abs(pi); pi /= pi.sum()
    return pi

def _get_facial_window(face_df, t_start, t_end):
    mask = (face_df["t_sec"] >= t_start) & (face_df["t_sec"] < t_end)
    return face_df.loc[mask]

def _facial_valence_index(face_df):
    """FV = Happiness - mean(Anger, Fear, Sadness, Disgust) per frame."""
    neg_mean = face_df[["Anger","Fear","Sadness","Disgust"]].mean(axis=1)
    return face_df["Happiness"] - neg_mean


# ════════════════════════════════════════════════════════════════════════════
# D1: EMOTION PERCEPTION — 22 features
# ════════════════════════════════════════════════════════════════════════════

def extract_D1(face_df, stage_df, sam_val, sam_aro):
    f = {}

    emo = face_df[EMOTION_COLS]

    # ── Emodiversity ────────────────────────────────────────────────────────
    # Shannon entropy of mean emotion distribution
    means = emo.mean(); total = means.sum()
    probs = (means / total).clip(lower=1e-10) if total > 0 else pd.Series(np.ones(7)/7, index=EMOTION_COLS)
    H = -np.sum(probs * np.log2(probs))
    f["c1_shannon_entropy"] = float(H / np.log2(7))

    # Simpson diversity: per frame find dominant emotion, compute 1-Σp²
    dominant = emo.idxmax(axis=1)
    counts = dominant.value_counts(normalize=True)
    f["c1_simpson_diversity"] = float(1.0 - np.sum(counts.values ** 2))

    # Gini of dominant emotion frame counts
    p = counts.values
    p_sorted = np.sort(p); n = len(p_sorted)
    cumsum = np.cumsum(p_sorted)
    gini = (2 * np.sum((np.arange(1, n+1)) * p_sorted) / (n * p_sorted.sum()) - (n+1)/n) if p_sorted.sum() > 0 else 0
    f["c1_gini_emodiversity"] = float(np.clip(gini, 0, 1))

    # Emotion richness: number of emotions with mean intensity > 0.5
    f["c1_emotion_richness"] = float((means > 0.5).sum() / 7)

    # ── Granularity via ICC ─────────────────────────────────────────────────
    # Segment into 30-second epochs, compute ICC across epochs
    epoch_dur = 30.0
    t_max = face_df["t_sec"].max()
    epoch_data = []
    t = 0
    while t + epoch_dur <= t_max:
        w = _get_facial_window(face_df, t, t + epoch_dur)
        if len(w) > 5:
            epoch_data.append(w[EMOTION_COLS].mean().values)
        t += epoch_dur
    if len(epoch_data) >= 3:
        epoch_arr = np.array(epoch_data)
        f["c1_emotion_differentiation_icc"] = _icc_1_way(epoch_arr)
        neg_epoch = epoch_arr[:, [EMOTION_COLS.index(e) for e in ["Anger","Fear","Sadness","Disgust","Stress"]]]
        f["c1_neg_emotion_differentiation"] = _icc_1_way(neg_epoch)
    else:
        f["c1_emotion_differentiation_icc"] = 0.5
        f["c1_neg_emotion_differentiation"] = 0.5

    # ── Temporal dynamics ───────────────────────────────────────────────────
    # Emotion transition entropy: 7×7 Markov on dominant emotion per frame
    dom_ds = _downsample_1hz(dominant.map({e: i for i, e in enumerate(EMOTION_COLS)}).fillna(0))
    n_emo = 7
    trans_counts = np.zeros((n_emo, n_emo))
    for a, b in zip(dom_ds[:-1].astype(int), dom_ds[1:].astype(int)):
        trans_counts[a, b] += 1
    row_s = trans_counts.sum(axis=1, keepdims=True); row_s[row_s == 0] = 1
    T_emo = trans_counts / row_s
    ht = -np.sum(T_emo * np.where(T_emo > 0, np.log2(T_emo + 1e-10), 0))
    f["c1_emotion_transition_entropy"] = float(ht / (n_emo * np.log2(n_emo + 1e-10)))

    # Emotion inertia: mean AR1 across 7 emotions at 1 Hz
    inertias = []
    for col in EMOTION_COLS:
        arr_1hz = _downsample_1hz(face_df[col])
        inertias.append(_ar1(arr_1hz))
    f["c1_emotion_inertia_composite"] = float(np.mean(inertias))

    # Emotion instability MSSD
    mssds = []
    for col in EMOTION_COLS:
        arr_1hz = _downsample_1hz(face_df[col])
        mssds.append(_mssd(arr_1hz))
    f["c1_emotion_instability_mssd"] = float(np.clip(np.mean(mssds) / 2.0, 0, 1))

    # ── Blending & co-occurrence ────────────────────────────────────────────
    # Mixed emotion rate: frames with ≥1 positive AND ≥1 negative > 1.0
    pos_active = (face_df[POSITIVE_EMOS] > 1.0).any(axis=1)
    neg_active = (face_df[NEGATIVE_EMOS] > 1.0).any(axis=1)
    f["c1_mixed_emotion_rate"] = float((pos_active & neg_active).mean())

    # Emotion blend count: unique binary patterns (threshold 1.0) / 127
    binary = (emo > 1.0).astype(int)
    patterns = binary.apply(lambda r: tuple(r), axis=1)
    unique_patterns = patterns.nunique()
    f["c1_emotion_blend_count"] = float(min(unique_patterns / 127, 1.0))

    # ── Cross-stage profile differentiation ────────────────────────────────
    stage_profiles = {}
    for s in stage_df["stage"].unique():
        m = face_df.loc[face_df["stage"] == s, EMOTION_COLS]
        if len(m) > 5:
            stage_profiles[s] = m.mean().values
    if len(stage_profiles) >= 2:
        keys = list(stage_profiles.keys())
        dists = []
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a, b = stage_profiles[keys[i]], stage_profiles[keys[j]]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                cos_d = 1.0 - (np.dot(a, b) / (na * nb + 1e-10)) if na > 0 and nb > 0 else 0.5
                dists.append(cos_d)
        f["c1_cross_stage_profile_diff"] = float(np.mean(dists))
    else:
        f["c1_cross_stage_profile_diff"] = 0.5

    # Stage emodiversity variance: SD of per-stage Shannon entropies
    stage_entropies = []
    for s in stage_df["stage"].unique():
        m = face_df.loc[face_df["stage"] == s, EMOTION_COLS]
        if len(m) > 5:
            sm = m.mean(); st = sm.sum()
            if st > 0:
                sp = (sm / st).clip(lower=1e-10)
                stage_entropies.append(-np.sum(sp * np.log2(sp)) / np.log2(7))
    f["c1_stage_emodiversity_variance"] = float(np.std(stage_entropies)) if len(stage_entropies) > 1 else 0.0

    # ── Frequency domain ────────────────────────────────────────────────────
    spectral_entropies, hf_ratios = [], []
    fps = 25.0
    for col in AU_COLS:
        sig = face_df[col].values.astype(float)
        if len(sig) < 50:
            continue
        try:
            freqs, psd = signal.welch(sig, fs=fps, nperseg=min(256, len(sig)//2))
            psd_norm = psd / (psd.sum() + 1e-10)
            H = -np.sum(psd_norm * np.log2(psd_norm + 1e-10)) / np.log2(len(psd))
            spectral_entropies.append(H)
            hf_mask = freqs >= 1.0
            hf_ratios.append(psd[hf_mask].sum() / (psd.sum() + 1e-10))
        except Exception:
            pass
    f["c1_spectral_entropy"]       = float(np.mean(spectral_entropies)) if spectral_entropies else 0.5
    f["c1_high_freq_power_ratio"]  = float(np.mean(hf_ratios))          if hf_ratios         else 0.5

    # ── Distributional ──────────────────────────────────────────────────────
    # Emotion kurtosis (excess) in 5-second windows
    window_5s = int(5 * fps)
    kurtoses, skewnesses = [], []
    for col in EMOTION_COLS:
        arr = face_df[col].values
        k_vals, s_vals = [], []
        for i in range(0, len(arr) - window_5s, window_5s):
            chunk = arr[i:i+window_5s]
            if np.std(chunk) > 1e-6:
                k_vals.append(stats.kurtosis(chunk, fisher=True))
                s_vals.append(stats.skew(chunk))
        kurtoses.append(np.mean(k_vals) if k_vals else 0)
        skewnesses.append(np.std(s_vals) if len(s_vals) > 1 else 0)
    f["c1_emotion_kurtosis"]   = float(np.clip(np.nanmean(kurtoses) / 10.0, -1, 1))
    f["c1_skewness_balance"]   = float(np.clip(np.nanmean(skewnesses), 0, 2) / 2.0)

    # ── AU network ──────────────────────────────────────────────────────────
    # AU activation breadth
    au_active = (face_df[AU_COLS] > 0.5).sum(axis=1)
    f["c1_au_activation_breadth"] = float(au_active.mean() / len(AU_COLS))

    # Expression distinctiveness: MI = H(AU) - H(AU|Stage)
    # Approximate via variance explained
    au_var_total = face_df[AU_COLS].var().mean()
    stage_groups = face_df.groupby("stage")[AU_COLS].var().mean()
    within_var = face_df.groupby("stage")[AU_COLS].var()
    mean_within = within_var.mean().mean()
    mi_proxy = 1.0 - (mean_within / (au_var_total + 1e-10)) if au_var_total > 0 else 0.5
    f["c1_expression_distinctiveness"] = float(np.clip(mi_proxy, 0, 1))

    # Expression ambiguity: H(AU|Stage) proxy (normalised mean within-stage AU variance)
    f["c1_expression_ambiguity"] = float(np.clip(mean_within / (au_var_total + 1e-10), 0, 1))

    # AU network density: pairwise co-activation probability
    au_bin = (face_df[AU_COLS] > 0.5).astype(float)
    n_au = len(AU_COLS)
    if len(au_bin) > 0:
        coprob = au_bin.T.dot(au_bin) / (len(au_bin) + 1e-10)
        np.fill_diagonal(coprob.values, 0)
        max_edges = n_au * (n_au - 1)
        density = (coprob > 0.1).values.sum() / max_edges
        f["c1_au_network_density"] = float(density)
    else:
        f["c1_au_network_density"] = 0.5

    # Positive/negative balance
    pos_mean = face_df[POSITIVE_EMOS].mean().mean()
    neg_mean = face_df[NEGATIVE_EMOS].mean().mean()
    total_emo = pos_mean + neg_mean
    f["c1_posneg_balance"] = float(pos_mean / total_emo) if total_emo > 0 else 0.5

    return f


# ════════════════════════════════════════════════════════════════════════════
# D2: EMOTION REGULATION — 22 features
# ════════════════════════════════════════════════════════════════════════════

def _find_stress_peaks(stress_1hz: np.ndarray):
    """Return indices of stress peaks (> personal mean + 1 SD)."""
    mean_s = np.nanmean(stress_1hz)
    sd_s   = np.nanstd(stress_1hz)
    threshold = mean_s + sd_s
    peaks, _ = signal.find_peaks(stress_1hz, height=threshold, distance=5)
    return peaks, mean_s, sd_s

def extract_D2(face_df, stage_df, sam_val, sam_aro):
    f = {}

    stress_1hz = _downsample_1hz(face_df["Stress"])
    fps = 25.0

    peaks, mean_s, sd_s = _find_stress_peaks(stress_1hz)

    # ── Recovery dynamics ────────────────────────────────────────────────────
    recovery_latencies, recovery_completeness_vals, peak_slopes = [], [], []
    for pk in peaks:
        pre_baseline = float(np.nanmean(stress_1hz[max(0, pk-10):pk]))
        # Find when stress returns within 0.5 SD of mean
        for i in range(pk, min(pk + 60, len(stress_1hz))):
            if stress_1hz[i] <= mean_s + 0.5 * sd_s:
                recovery_latencies.append(i - pk)
                break
        else:
            recovery_latencies.append(60)
        # Completeness: (peak - post_recovery) / (peak - pre_baseline)
        post = float(np.nanmean(stress_1hz[min(pk+5, len(stress_1hz)-1):min(pk+30, len(stress_1hz))]))
        peak_val = stress_1hz[pk]
        denom = peak_val - pre_baseline
        comp = (peak_val - post) / (denom + 1e-10) if denom > 0 else 0.5
        recovery_completeness_vals.append(np.clip(comp, 0, 2))
        # Peak-to-baseline slope
        window_end = min(pk + 30, len(stress_1hz))
        if window_end > pk + 2:
            segment = stress_1hz[pk:window_end]
            x = np.arange(len(segment), dtype=float)
            slope = np.polyfit(x, segment, 1)[0] if np.std(segment) > 1e-6 else 0
            peak_slopes.append(-slope)  # negate: faster descent = higher value

    f["c2_stress_recovery_latency"]    = float(np.mean(recovery_latencies)) if recovery_latencies else 30.0
    f["c2_recovery_completeness"]      = float(np.mean(recovery_completeness_vals)) if recovery_completeness_vals else 0.5
    f["c2_peak_to_baseline_slope"]     = float(np.mean(peak_slopes)) if peak_slopes else 0.0

    # ── Emotional inertia ────────────────────────────────────────────────────
    f["c2_stress_inertia"]             = _ar1(stress_1hz)
    neg_composite = face_df[["Anger","Fear","Sadness","Disgust"]].mean(axis=1)
    neg_z = (neg_composite - neg_composite.mean()) / (neg_composite.std() + 1e-10)
    f["c2_negative_emotion_inertia"]   = _ar1(_downsample_1hz(neg_z))

    # ── Affective variability ────────────────────────────────────────────────
    f["c2_stress_instability_mssd"]    = float(np.clip(_mssd(stress_1hz) / 2.0, 0, 1))
    f["c2_emotion_variability_sd"]     = float(np.clip(np.nanstd(stress_1hz) / 3.0, 0, 1))

    # ── Regulation flexibility ───────────────────────────────────────────────
    # Repertoire diversity: Shannon entropy of binarized emotion per stage
    rep_entropies = []
    for s in stage_df["stage"].unique():
        m = face_df.loc[face_df["stage"] == s, EMOTION_COLS]
        if len(m) > 5:
            active = (m > 1.0).mean()
            total = active.sum()
            if total > 0:
                p = (active / total).clip(lower=1e-10)
                rep_entropies.append(-np.sum(p * np.log2(p)) / np.log2(7))
    f["c2_emotion_repertoire_diversity"] = float(np.mean(rep_entropies)) if rep_entropies else 0.5

    # Cross-stage regulation variability: SD of mean stress across stages
    stage_stress = {}
    for s in stage_df["stage"].unique():
        m = face_df.loc[face_df["stage"] == s, "Stress"]
        if len(m) > 5:
            stage_stress[s] = m.mean()
    f["c2_cross_stage_regulation_variability"] = float(np.std(list(stage_stress.values()))) if len(stage_stress) > 1 else 0.0

    # Regulation strategy switching: dominant emotion transitions per second
    dom_1hz = _downsample_1hz(face_df[EMOTION_COLS].idxmax(axis=1).map(
        {e: i for i, e in enumerate(EMOTION_COLS)}).fillna(0))
    switches = np.sum(np.diff(dom_1hz.astype(int)) != 0)
    f["c2_regulation_strategy_switching"] = float(switches / (len(dom_1hz) + 1e-10))

    # ── Habituation / dampening ──────────────────────────────────────────────
    # Stress peak attenuation: slope of peak intensities over time
    if len(peaks) >= 3:
        peak_vals = stress_1hz[peaks]
        x = np.arange(len(peak_vals), dtype=float)
        slope = np.polyfit(x, peak_vals, 1)[0]
        f["c2_stress_peak_attenuation"] = float(-slope)  # negative slope = habituation = positive
    else:
        f["c2_stress_peak_attenuation"] = 0.0

    # Habituation ratio: second half / first half mean stress
    mid = len(stress_1hz) // 2
    first_half  = np.nanmean(stress_1hz[:mid])
    second_half = np.nanmean(stress_1hz[mid:])
    f["c2_stress_habituation_ratio"]   = float(second_half / (first_half + 1e-10))

    # ── AU suppression ───────────────────────────────────────────────────────
    # Lip press index: AU23 during high-stress vs low-stress
    high_stress_mask = face_df["Stress"] > (mean_s + sd_s)
    low_stress_mask  = face_df["Stress"] < mean_s
    au23_high = face_df.loc[high_stress_mask, "AU23_r"].mean() if high_stress_mask.any() else 0.0
    au23_low  = face_df.loc[low_stress_mask,  "AU23_r"].mean() if low_stress_mask.any()  else 0.0
    f["c2_lip_press_index"] = float(au23_high - au23_low)

    # Negative expression dampening: later vs earlier stress episodes
    if len(peaks) >= 2:
        early_peaks = peaks[:len(peaks)//2]
        late_peaks  = peaks[len(peaks)//2:]
        neg_aus = ["AU01_r","AU04_r","AU15_r","AU20_r","AU23_r"]
        neg_early = face_df.loc[face_df.index.isin(
            [int(p * fps) for p in early_peaks if int(p * fps) < len(face_df)]),
            [c for c in neg_aus if c in face_df.columns]].mean().mean()
        neg_late = face_df.loc[face_df.index.isin(
            [int(p * fps) for p in late_peaks if int(p * fps) < len(face_df)]),
            [c for c in neg_aus if c in face_df.columns]].mean().mean()
        f["c2_negative_expression_dampening"] = float(
            np.clip((neg_early - neg_late) / (neg_early + 1e-10), -1, 1))
    else:
        f["c2_negative_expression_dampening"] = 0.0

    # ── Temporal regulation ──────────────────────────────────────────────────
    # Anticipatory regulation: mean stress 30s before Block/Evaluation vs baseline
    anticipatory_vals = []
    for _, row in stage_df.iterrows():
        if row["stage"] in ["Block", "Evaluation"]:
            t_before_start = max(0, row["start_sec"] - 30)
            t_start = row["start_sec"]
            pre_window = _get_facial_window(face_df, t_before_start, t_start)
            if len(pre_window) > 5:
                anticipatory_vals.append(pre_window["Stress"].mean())
    overall_mean_stress = face_df["Stress"].mean()
    if anticipatory_vals:
        # Lower anticipatory stress = better regulation
        f["c2_anticipatory_regulation"] = float(
            np.clip(1.0 - np.mean(anticipatory_vals) / (overall_mean_stress * 2 + 1e-10), 0, 1))
    else:
        f["c2_anticipatory_regulation"] = 0.5

    # Early vs late regulation: last third / first third mean stress
    n = len(face_df)
    third = n // 3
    early_stress = face_df["Stress"].iloc[:third].mean()
    late_stress  = face_df["Stress"].iloc[2*third:].mean()
    f["c2_early_vs_late_regulation_ratio"] = float(late_stress / (early_stress + 1e-10))

    # ── Cross-stage regulation ───────────────────────────────────────────────
    # Block stage regulation efficiency: slope of stress within each Block
    block_slopes = []
    for _, row in stage_df[stage_df["stage"] == BLOCK_STAGE].iterrows():
        w = _get_facial_window(face_df, row["start_sec"], row["end_sec"])
        if len(w) > 10:
            x = np.arange(len(w), dtype=float)
            s = w["Stress"].values
            if np.std(s) > 1e-6:
                slope = np.polyfit(x, s, 1)[0]
                block_slopes.append(-slope)  # negative slope = decreasing = positive
    f["c2_block_stage_regulation_efficiency"] = float(np.mean(block_slopes)) if block_slopes else 0.0

    # Production stress floor: median stress during Production
    prod_mask = face_df["stage"] == "Production"
    f["c2_production_stress_floor"] = float(face_df.loc[prod_mask, "Stress"].median()) if prod_mask.any() else overall_mean_stress

    # ── Reappraisal ──────────────────────────────────────────────────────────
    # Neg-to-pos transition speed
    dom_series = face_df[EMOTION_COLS].idxmax(axis=1)
    dom_1hz_named = dom_series.iloc[::int(fps)]
    transitions_np = []
    pos_set = set(POSITIVE_EMOS)
    neg_set = set(NEGATIVE_EMOS)
    in_neg, start_neg = False, 0
    for i, d in enumerate(dom_1hz_named):
        if d in neg_set and not in_neg:
            in_neg = True; start_neg = i
        elif d in pos_set and in_neg:
            transitions_np.append(i - start_neg)
            in_neg = False
    f["c2_neg_to_pos_transition_speed"] = float(np.mean(transitions_np)) if transitions_np else 10.0

    # Reappraisal proxy coactivation: AU04 & AU12 during high-stress frames
    stress_high = face_df["Stress"] > (mean_s + sd_s)
    if stress_high.any():
        au04 = face_df.loc[stress_high, "AU04_r"] if "AU04_r" in face_df.columns else pd.Series([0.0])
        au12 = face_df.loc[stress_high, "AU12_r"] if "AU12_r" in face_df.columns else pd.Series([0.0])
        both_active = ((au04 > 1.0) & (au12 > 1.0)).mean()
        f["c2_reappraisal_proxy_coactivation"] = float(both_active)
    else:
        f["c2_reappraisal_proxy_coactivation"] = 0.0

    # ── Multi-channel integration ────────────────────────────────────────────
    # SAM-facial discrepancy reduction slope
    fv = _facial_valence_index(face_df)
    sam_times = SAM_CHECKPOINT_TIMES_SEC
    discrepancies = []
    for i, (t, v) in enumerate(zip(sam_times, sam_val)):
        w = _get_facial_window(face_df, max(0, t - 180), t)
        if len(w) > 10:
            fv_mean = (w["Happiness"] - w[["Anger","Fear","Sadness","Disgust"]].mean(axis=1)).mean()
            fv_z = (fv_mean - fv.mean()) / (fv.std() + 1e-10)
            v_z  = (v - np.nanmean(sam_val)) / (np.nanstd(sam_val) + 1e-10)
            discrepancies.append(abs(v_z - fv_z))
    if len(discrepancies) >= 3:
        x = np.arange(len(discrepancies), dtype=float)
        slope = np.polyfit(x, discrepancies, 1)[0]
        f["c2_sam_facial_discrepancy_reduction"] = float(-slope)  # decreasing = positive
    else:
        f["c2_sam_facial_discrepancy_reduction"] = 0.0

    # Valence-arousal regulation coupling: within-person r between Δvalence and Δstress
    val_diff = np.diff(sam_val)
    stress_at_checkpoints = []
    for t in sam_times:
        w = _get_facial_window(face_df, max(0, t-180), t)
        stress_at_checkpoints.append(w["Stress"].mean() if len(w) > 0 else np.nan)
    stress_diff = -np.diff(stress_at_checkpoints)  # negate: val increase + stress decrease
    f["c2_valence_arousal_regulation_coupling"] = _safe_pearsonr(val_diff, stress_diff)

    return f


# ════════════════════════════════════════════════════════════════════════════
# D3: ADAPTIVE PERSISTENCE — 25 features
# ════════════════════════════════════════════════════════════════════════════

def extract_D3(face_df, stage_df, sam_val, sam_aro):
    f = {}

    durs = _stage_durations(stage_df)
    total_dur = durs.sum()
    block_dur = durs.get(BLOCK_STAGE, 0.0)
    prod_dur  = sum(durs.get(s, 0.0) for s in CORE_PRODUCTIVE)
    all_prod  = sum(durs.get(s, 0.0) for s in PRODUCTIVE_STAGES)

    stage_seq = stage_df["stage"].tolist()
    T, states = _markov_matrix(stage_df)

    # ── Block-to-productive transitions ─────────────────────────────────────
    # Block exit diversity: entropy of post-Block stages
    post_block_stages = [stage_seq[i+1] for i in range(len(stage_seq)-1)
                         if stage_seq[i] == BLOCK_STAGE and stage_seq[i+1] != BLOCK_STAGE]
    if post_block_stages:
        counts = pd.Series(post_block_stages).value_counts(normalize=True)
        H = -np.sum(counts * np.log2(counts + 1e-10))
        f["c3_block_exit_diversity"] = float(H / np.log2(max(len(counts), 2)))
    else:
        f["c3_block_exit_diversity"] = 0.5

    # Block-to-production rate
    block_exits = [stage_seq[i+1] for i in range(len(stage_seq)-1) if stage_seq[i] == BLOCK_STAGE]
    f["c3_block_to_production_rate"] = float(block_exits.count("Production") / (len(block_exits) + 1e-10))

    # Recovery acceleration: slope of Block durations over ordinal number
    block_rows = stage_df[stage_df["stage"] == BLOCK_STAGE].copy()
    if len(block_rows) >= 3:
        dur_vals = (block_rows["end_sec"] - block_rows["start_sec"]).values
        x = np.arange(len(dur_vals), dtype=float)
        slope = np.polyfit(x, dur_vals, 1)[0]
        f["c3_recovery_acceleration"] = float(-slope)  # shorter blocks = positive
    else:
        f["c3_recovery_acceleration"] = 0.0

    # ── Productive struggle ──────────────────────────────────────────────────
    # Productive confusion index
    resolved_stress, unresolved_stress = [], []
    for i in range(len(stage_seq) - 1):
        if stage_seq[i] == BLOCK_STAGE:
            next_s = stage_seq[i+1]
            row = stage_df.iloc[i]
            w = _get_facial_window(face_df, row["start_sec"], row["end_sec"])
            mean_block_stress = w["Stress"].mean() if len(w) > 0 else 0.0
            if next_s in CORE_PRODUCTIVE:
                resolved_stress.append(mean_block_stress)
            else:
                unresolved_stress.append(mean_block_stress)
    r_mean = np.mean(resolved_stress) if resolved_stress else 0.0
    u_mean = np.mean(unresolved_stress) if unresolved_stress else r_mean
    f["c3_productive_confusion_index"] = float(r_mean / (u_mean + 1e-10))

    # Confusion resolution rate
    short_high_blocks = []
    if len(block_rows) > 0:
        dur_blocks = (block_rows["end_sec"] - block_rows["start_sec"]).values
        q1 = np.percentile(dur_blocks, 25)
        med_stress = face_df.loc[face_df["stage"] == BLOCK_STAGE, "Stress"].median() if (face_df["stage"] == BLOCK_STAGE).any() else 0
        for i, (_, row) in enumerate(block_rows.iterrows()):
            dur_b = row["end_sec"] - row["start_sec"]
            w = _get_facial_window(face_df, row["start_sec"], row["end_sec"])
            block_stress = w["Stress"].mean() if len(w) > 0 else 0.0
            if dur_b <= q1 and block_stress > med_stress:
                # Check if resolved
                idx_in_seq = stage_df.index.get_loc(row.name) if row.name in stage_df.index else -1
                if idx_in_seq >= 0 and idx_in_seq < len(stage_df) - 1:
                    if stage_seq[idx_in_seq + 1] != BLOCK_STAGE:
                        short_high_blocks.append(True)
                    else:
                        short_high_blocks.append(False)
    f["c3_confusion_resolution_rate"] = float(np.mean(short_high_blocks)) if short_high_blocks else 0.5

    # ── Markov features ──────────────────────────────────────────────────────
    pi = _stationary(T, states)
    # Productive absorbing probability (sum over core productive states)
    prod_prob = sum(pi[states.index(s)] for s in CORE_PRODUCTIVE if s in states)
    f["c3_productive_absorbing_probability"] = float(prod_prob)

    # Second-order Block escape
    if BLOCK_STAGE in states:
        bi = states.index(BLOCK_STAGE)
        # P(non-Block | Block_{t-1}, Block_{t-2}): probability of escaping from 2 consecutive blocks
        block_block_seqs = sum(1 for a, b in zip(stage_seq[:-2], stage_seq[1:-1])
                               if a == BLOCK_STAGE and b == BLOCK_STAGE)
        block_block_nonblock = sum(1 for a, b, c in zip(stage_seq[:-2], stage_seq[1:-1], stage_seq[2:])
                                   if a == BLOCK_STAGE and b == BLOCK_STAGE and c != BLOCK_STAGE)
        f["c3_second_order_block_escape"] = float(block_block_nonblock / (block_block_seqs + 1e-10))
    else:
        f["c3_second_order_block_escape"] = 0.8

    # Markov entropy (original D3 feature)
    n_st = len(states)
    max_H = np.log2(n_st) if n_st > 1 else 1.0
    row_entropies = []
    for row in T:
        row_p = row[row > 0]
        if len(row_p):
            H = -np.sum(row_p * np.log2(row_p))
            row_entropies.append(H / max_H)
    f["c3_markov_entropy"] = float(np.mean(row_entropies)) if row_entropies else 0.0

    # Stationary Block inv
    block_prob = pi[states.index(BLOCK_STAGE)] if BLOCK_STAGE in states else 0.0
    f["c3_stationary_block_inv"] = float(1.0 - block_prob)

    # ── Stage momentum ───────────────────────────────────────────────────────
    stage_ordinal = {"Preparation": 1, "Ideation": 2, "Production": 3,
                     "Evaluation": 4, "Refinement": 5, BLOCK_STAGE: 0}
    ord_vals = [stage_ordinal.get(s, 2) for s in stage_seq]
    non_block_diffs = [ord_vals[i+1] - ord_vals[i]
                       for i in range(len(ord_vals)-1)
                       if stage_seq[i] != BLOCK_STAGE and stage_seq[i+1] != BLOCK_STAGE]
    f["c3_forward_stage_momentum"] = float(np.mean(non_block_diffs)) if non_block_diffs else 0.0

    # ── Engagement sustainability ─────────────────────────────────────────────
    # Longest productive streak (seconds without Block)
    current_streak, max_streak = 0.0, 0.0
    for _, row in stage_df.iterrows():
        if row["stage"] in PRODUCTIVE_STAGES:
            current_streak += row["end_sec"] - row["start_sec"]
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0.0
    f["c3_longest_productive_streak"] = float(max_streak)

    # Productive streak variability (CV)
    streaks = []
    current_streak = 0.0
    for _, row in stage_df.iterrows():
        if row["stage"] in PRODUCTIVE_STAGES:
            current_streak += row["end_sec"] - row["start_sec"]
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0.0
    if streaks and np.mean(streaks) > 0:
        f["c3_productive_streak_variability"] = float(np.std(streaks) / np.mean(streaks))
    else:
        f["c3_productive_streak_variability"] = 0.5

    # Late task engagement ratio
    t_total = total_dur
    first_third_end = t_total / 3
    last_third_start = 2 * t_total / 3
    first_prod = stage_df.loc[
        stage_df["stage"].isin(PRODUCTIVE_STAGES)].apply(
        lambda r: max(0, min(r["end_sec"], first_third_end) - max(r["start_sec"], 0)), axis=1).sum()
    last_prod = stage_df.loc[
        stage_df["stage"].isin(PRODUCTIVE_STAGES)].apply(
        lambda r: max(0, min(r["end_sec"], t_total) - max(r["start_sec"], last_third_start)), axis=1).sum()
    f["c3_late_task_engagement_ratio"] = float(last_prod / (first_prod + 1e-10))

    # ── Frustration tolerance ────────────────────────────────────────────────
    # Block negative affect damping: slope of stress within blocks, averaged
    block_stress_slopes = []
    for _, row in block_rows.iterrows():
        w = _get_facial_window(face_df, row["start_sec"], row["end_sec"])
        if len(w) > 10:
            x = np.arange(len(w), dtype=float)
            s = w["Stress"].values
            if np.std(s) > 1e-6:
                block_stress_slopes.append(-np.polyfit(x, s, 1)[0])
    f["c3_block_negative_affect_damping"] = float(np.mean(block_stress_slopes)) if block_stress_slopes else 0.0

    # Block anger suppression: mean Anger in Block / mean Anger non-Block
    block_anger    = face_df.loc[face_df["stage"] == BLOCK_STAGE, "Anger"].mean()
    nonblock_anger = face_df.loc[face_df["stage"] != BLOCK_STAGE, "Anger"].mean()
    f["c3_block_anger_suppression"] = float(block_anger / (nonblock_anger + 1e-10))

    # Stress inertia during blocks
    block_stress_arr = face_df.loc[face_df["stage"] == BLOCK_STAGE, "Stress"]
    f["c3_stress_inertia_during_blocks"] = _ar1(_downsample_1hz(block_stress_arr)) if len(block_stress_arr) > 10 else 0.0

    # ── Buoyancy ──────────────────────────────────────────────────────────────
    # Bounce-back index: post-Block Happiness / pre-Block Happiness
    bounce_vals = []
    for _, row in block_rows.iterrows():
        t_pre_end   = row["start_sec"]
        t_pre_start = max(0, t_pre_end - 60)
        t_post_start = row["end_sec"]
        t_post_end   = t_post_start + 60
        pre_h  = _get_facial_window(face_df, t_pre_start, t_pre_end)["Happiness"].mean()
        post_h = _get_facial_window(face_df, t_post_start, t_post_end)["Happiness"].mean()
        if pre_h > 0:
            bounce_vals.append(post_h / pre_h)
    f["c3_bounce_back_index"] = float(np.mean(bounce_vals)) if bounce_vals else 1.0

    # Adversity composure index: 1 - (Var_stress_block / Var_stress_overall)
    var_block   = face_df.loc[face_df["stage"] == BLOCK_STAGE, "Stress"].var() if (face_df["stage"] == BLOCK_STAGE).any() else 0
    var_overall = face_df["Stress"].var()
    f["c3_adversity_composure_index"] = float(np.clip(1.0 - var_block / (var_overall + 1e-10), 0, 1))

    # ── Grit proxies ─────────────────────────────────────────────────────────
    f["c3_total_productive_time_ratio"]   = float(prod_dur  / (total_dur + 1e-10))
    f["c3_task_completion_persistence"]   = float(last_prod / (total_dur / 3 + 1e-10))

    non_block_dur = total_dur - block_dur
    f["c3_effective_time_efficiency"]     = float(prod_dur / (non_block_dur + 1e-10))

    # ── Stage quality ─────────────────────────────────────────────────────────
    # Productive iteration index: Production→Evaluation→Refinement cycles / total transitions
    cycles = sum(1 for i in range(len(stage_seq)-2)
                 if stage_seq[i] == "Production" and stage_seq[i+1] == "Evaluation")
    f["c3_productive_iteration_index"] = float(cycles / (len(stage_seq) + 1e-10))

    # Regressive transition ratio
    forward_pairs = {("Ideation","Production"), ("Production","Evaluation"),
                     ("Evaluation","Refinement"), ("Preparation","Ideation")}
    backward_pairs = {("Production","Preparation"), ("Evaluation","Ideation"),
                      ("Refinement","Production")}
    non_block_transitions = [(stage_seq[i], stage_seq[i+1]) for i in range(len(stage_seq)-1)
                             if BLOCK_STAGE not in (stage_seq[i], stage_seq[i+1])
                             and stage_seq[i] != stage_seq[i+1]]
    regressive = sum(1 for t in non_block_transitions if t in backward_pairs)
    f["c3_regressive_transition_ratio"] = float(regressive / (len(non_block_transitions) + 1e-10))

    # Block emotional trajectory slope
    block_traj_slopes = []
    for _, row in block_rows.iterrows():
        w = _get_facial_window(face_df, row["start_sec"], row["end_sec"])
        if len(w) > 10:
            x = np.arange(len(w), dtype=float)
            hs = w["Happiness"].values - w["Stress"].values
            if np.std(hs) > 1e-6:
                block_traj_slopes.append(np.polyfit(x, hs, 1)[0])
    f["c3_block_emotional_trajectory_slope"] = float(np.mean(block_traj_slopes)) if block_traj_slopes else 0.0

    # Post-Block engagement quality: mean Happiness in 120s after Block / session mean
    post_block_happy = []
    for _, row in block_rows.iterrows():
        w = _get_facial_window(face_df, row["end_sec"], row["end_sec"] + 120)
        if len(w) > 5:
            post_block_happy.append(w["Happiness"].mean())
    session_mean_happy = face_df["Happiness"].mean()
    f["c3_post_block_engagement_quality"] = float(
        np.mean(post_block_happy) / (session_mean_happy + 1e-10)) if post_block_happy else 1.0

    # Struggle benefit index: correlation of Block duration with post-Block productive streak
    block_durations_list, post_streak_lengths = [], []
    for _, row in block_rows.iterrows():
        bd = row["end_sec"] - row["start_sec"]
        # Find next productive streak length
        t = row["end_sec"]
        streak = 0.0
        for _, nrow in stage_df[stage_df["start_sec"] >= t].iterrows():
            if nrow["stage"] in PRODUCTIVE_STAGES:
                streak += nrow["end_sec"] - nrow["start_sec"]
            else:
                break
        block_durations_list.append(bd)
        post_streak_lengths.append(streak)
    f["c3_struggle_benefit_index"] = _safe_pearsonr(
        np.array(block_durations_list), np.array(post_streak_lengths))

    return f


# ════════════════════════════════════════════════════════════════════════════
# D4: AFFECTIVE SELF-AWARENESS — 25 features
# ════════════════════════════════════════════════════════════════════════════

def extract_D4(face_df, stage_df, sam_val, sam_aro):
    f = {}

    fv_series = _facial_valence_index(face_df)  # per frame
    sam_times = SAM_CHECKPOINT_TIMES_SEC

    # Compute facial valence and total AU per checkpoint window (~3 min each)
    fv_per_checkpoint, au_total_per_checkpoint = [], []
    neg_per_checkpoint = []
    for t in sam_times:
        w = _get_facial_window(face_df, max(0, t - 180), t)
        if len(w) > 5:
            fv_per_checkpoint.append((w["Happiness"] - w[["Anger","Fear","Sadness","Disgust"]].mean(axis=1)).mean())
            au_total_per_checkpoint.append(w[AU_COLS].mean().mean())
            neg_per_checkpoint.append(w[["Anger","Fear","Sadness","Disgust"]].mean().mean())
        else:
            fv_per_checkpoint.append(np.nan)
            au_total_per_checkpoint.append(np.nan)
            neg_per_checkpoint.append(np.nan)

    fv_arr   = np.array(fv_per_checkpoint)
    au_arr   = np.array(au_total_per_checkpoint)
    neg_arr  = np.array(neg_per_checkpoint)

    # ── SAM-AU concordance ───────────────────────────────────────────────────
    f["c4_valence_happiness_concordance"]   = _safe_pearsonr(sam_val, fv_arr)
    f["c4_valence_negemo_concordance"]      = _safe_pearsonr(6 - sam_val, neg_arr)  # invert scale
    f["c4_arousal_au_concordance"]          = _safe_pearsonr(sam_aro, au_arr)
    f["c4_valence_stress_inverse_concordance"] = _safe_pearsonr(
        sam_val, -np.array([_get_facial_window(face_df, max(0,t-180), t)["Stress"].mean()
                            if len(_get_facial_window(face_df, max(0,t-180),t)) > 5 else np.nan
                            for t in sam_times]))

    # ── SAM trajectory ───────────────────────────────────────────────────────
    # Valence curvature: quadratic coefficient
    x = np.arange(10, dtype=float)
    valid = ~np.isnan(sam_val)
    if valid.sum() >= 4:
        coeffs = np.polyfit(x[valid], sam_val[valid], 2)
        f["c4_valence_curvature"] = float(coeffs[0])
    else:
        f["c4_valence_curvature"] = 0.0

    # Valence turning points
    diffs = np.diff(sam_val[valid]) if valid.sum() >= 2 else np.array([0])
    turning_points = np.sum(np.diff(np.sign(diffs)) != 0)
    f["c4_valence_turning_points"] = float(turning_points / 8)

    # Arousal trend (OLS slope)
    if valid.sum() >= 3:
        aro_valid = ~np.isnan(sam_aro)
        if aro_valid.sum() >= 3:
            slope = np.polyfit(x[aro_valid], sam_aro[aro_valid], 1)[0]
            f["c4_arousal_trend"] = float(slope)
        else:
            f["c4_arousal_trend"] = 0.0
    else:
        f["c4_arousal_trend"] = 0.0

    # Arousal stability (negated SD)
    f["c4_arousal_stability"] = float(1.0 - min(np.nanstd(sam_aro) / 2.0, 1.0))

    # ── Interoceptive accuracy ───────────────────────────────────────────────
    # Facial valence accuracy: r between z-scored SAM valence and z-scored facial valence
    sv_z  = (sam_val  - np.nanmean(sam_val))  / (np.nanstd(sam_val)  + 1e-10)
    fv_z  = (fv_arr   - np.nanmean(fv_arr[~np.isnan(fv_arr)]))  / (np.nanstd(fv_arr[~np.isnan(fv_arr)])  + 1e-10) if not np.all(np.isnan(fv_arr)) else np.zeros(10)
    f["c4_facial_valence_accuracy"] = _safe_pearsonr(sv_z, fv_z)

    # Change detection alignment: r between Δ SAM valence and Δ facial valence
    delta_sam = np.diff(sam_val)
    delta_fv  = np.diff(fv_arr)
    f["c4_change_detection_alignment"] = _safe_pearsonr(delta_sam, delta_fv)

    # Block stage awareness: r between SAM drops at Block checkpoints and facial neg increases
    block_sam_drops, block_neg_increases = [], []
    for i, t in enumerate(sam_times[1:], 1):
        w = _get_facial_window(face_df, max(0, t-180), t)
        if len(w) > 5 and w["stage"].eq(BLOCK_STAGE).any():
            block_sam_drops.append(sam_val[i-1] - sam_val[i])
            block_neg_increases.append(w[NEGATIVE_EMOS].mean().mean())
    f["c4_block_stage_awareness"] = _safe_pearsonr(
        np.array(block_sam_drops), np.array(block_neg_increases)) if len(block_sam_drops) >= 3 else 0.0

    # ── Calibration ──────────────────────────────────────────────────────────
    bias_vals = []
    for sv, fv in zip(sv_z, fv_z):
        if not np.isnan(sv) and not np.isnan(fv):
            bias_vals.append(sv - fv)
    f["c4_positive_reporting_bias"]  = float(np.mean(bias_vals)) if bias_vals else 0.0
    f["c4_absolute_calibration_error"] = float(-np.mean(np.abs(bias_vals))) if bias_vals else 0.0

    # ── Arousal optimality ───────────────────────────────────────────────────
    f["c4_arousal_optimality"]        = float(-abs(np.nanmean(sam_aro) - 3.0))
    f["c4_arousal_midrange_proportion"] = float(np.sum((sam_aro >= 2) & (sam_aro <= 4)) / 10)

    # ── Self-efficacy ─────────────────────────────────────────────────────────
    # Valence recovery magnitude: avg increase after valence dips
    recoveries_val = []
    for i in range(1, len(sam_val) - 1):
        if not np.isnan(sam_val[i-1]) and not np.isnan(sam_val[i]) and not np.isnan(sam_val[i+1]):
            if sam_val[i] < sam_val[i-1] and sam_val[i+1] > sam_val[i]:
                recoveries_val.append(sam_val[i+1] - sam_val[i])
    f["c4_valence_recovery_magnitude"] = float(np.mean(recoveries_val)) if recoveries_val else 0.0

    # Pride during production relative to global
    prod_mask = face_df["stage"] == "Production"
    if prod_mask.any():
        pride_prod   = face_df.loc[prod_mask, "Pride"].mean()
        pride_global = face_df["Pride"].mean()
        f["c4_pride_during_production"] = float(pride_prod / (pride_global + 1e-10))
    else:
        f["c4_pride_during_production"] = 1.0

    # Stage emotion appropriateness (Pekrun 2006)
    scores_appr = []
    for stage_name, emo_name, direction in [
        ("Ideation",   "Happiness", +1),
        ("Evaluation", "Stress",    +1),
        ("Refinement", "Happiness", +1),
        (BLOCK_STAGE,  "Stress",    -1),
    ]:
        m = face_df.loc[face_df["stage"] == stage_name, emo_name]
        if len(m) > 5:
            z = (m.mean() - face_df[emo_name].mean()) / (face_df[emo_name].std() + 1e-10)
            scores_appr.append(direction * float(z))
    f["c4_stage_emotion_appropriateness_d4"] = float(np.mean(scores_appr)) if scores_appr else 0.0

    # Evaluation anxiety alignment: r between stress during Evaluation and SAM arousal at nearest checkpoint
    eval_mask = face_df["stage"] == "Evaluation"
    if eval_mask.any():
        eval_stress_per_checkpoint = []
        for t in sam_times:
            w = _get_facial_window(face_df, max(0, t-180), t)
            w_eval = w[w["stage"] == "Evaluation"]
            eval_stress_per_checkpoint.append(w_eval["Stress"].mean() if len(w_eval) > 3 else np.nan)
        f["c4_evaluation_anxiety_alignment"] = _safe_pearsonr(
            np.array(eval_stress_per_checkpoint), sam_aro)
    else:
        f["c4_evaluation_anxiety_alignment"] = 0.0

    # ── Alexithymia-reversed ─────────────────────────────────────────────────
    sam_range = np.nanmax(sam_val) - np.nanmin(sam_val)
    fv_range  = np.nanmax(fv_arr[~np.isnan(fv_arr)]) - np.nanmin(fv_arr[~np.isnan(fv_arr)]) if not np.all(np.isnan(fv_arr)) else 1.0
    f["c4_emotional_range_concordance"] = float(
        1.0 - abs(1.0 - sam_range / (fv_range + 1e-10)) if fv_range > 0 else 0.5)

    f["c4_sam_response_range"]      = float(sam_range / 4.0)
    # Anchoring resistance: 1 - AR1 of SAM valence
    f["c4_anchoring_resistance"]    = float(1.0 - abs(_ar1(sam_val)))
    # Valence-arousal discrimination: 1 - |r(valence, arousal)|
    f["c4_valence_arousal_discrimination"] = float(1.0 - abs(_safe_pearsonr(sam_val, sam_aro)))

    # Temporal coherence: max cross-correlation between ΔSAM valence and ΔFV at lags 0,±1
    if len(delta_sam) >= 3 and not np.all(np.isnan(delta_fv)):
        valid_both = ~(np.isnan(delta_sam) | np.isnan(delta_fv))
        if valid_both.sum() >= 3:
            xcorrs = []
            ds = delta_sam[valid_both]; df_v = delta_fv[valid_both]
            for lag in [-1, 0, 1]:
                if lag == 0:
                    xcorrs.append(_safe_pearsonr(ds, df_v))
                elif lag == 1 and len(ds) > 1:
                    xcorrs.append(_safe_pearsonr(ds[:-1], df_v[1:]))
                elif lag == -1 and len(ds) > 1:
                    xcorrs.append(_safe_pearsonr(ds[1:], df_v[:-1]))
            f["c4_temporal_coherence"] = float(np.max(xcorrs)) if xcorrs else 0.0
        else:
            f["c4_temporal_coherence"] = 0.0
    else:
        f["c4_temporal_coherence"] = 0.0

    # Original features retained
    f["c4_mean_valence"] = float((np.nanmean(sam_val) - 1) / 4.0)
    valid = ~np.isnan(sam_val)
    if valid.sum() >= 2:
        slope = np.polyfit(x[valid], sam_val[valid], 1)[0]
        f["c4_valence_trend"] = float(0.5 + np.clip(slope * 2.5, -0.5, 0.5))
    else:
        f["c4_valence_trend"] = 0.5

    return f


# ════════════════════════════════════════════════════════════════════════════
# D5: EMOTION-COGNITION ALIGNMENT — 22 features
# ════════════════════════════════════════════════════════════════════════════

def extract_D5(face_df, stage_df, sam_val, sam_aro):
    f = {}

    fps = 25.0
    sam_times = SAM_CHECKPOINT_TIMES_SEC
    stage_seq = stage_df["stage"].tolist()

    def stage_mean(emo_col, stage_name):
        m = face_df.loc[face_df["stage"] == stage_name, emo_col]
        return m.mean() if len(m) > 5 else np.nan

    def au_mean_stage(au_col, stage_name):
        m = face_df.loc[face_df["stage"] == stage_name, au_col]
        return m.mean() if len(m) > 5 else np.nan

    # ── Stage-appropriate emotion ────────────────────────────────────────────
    # Ideation positive affect (z-scored within participant)
    hap_ide = stage_mean("Happiness", "Ideation")
    hap_mean = face_df["Happiness"].mean(); hap_std = face_df["Happiness"].std()
    f["c5_ideation_positive_affect"] = float((hap_ide - hap_mean) / (hap_std + 1e-10)) if not np.isnan(hap_ide) else 0.0

    # Production concentration: mean(AU04 + AU07) during Production
    au04_prod = au_mean_stage("AU04_r", "Production")
    au07_prod = au_mean_stage("AU07_r", "Production")
    if not (np.isnan(au04_prod) or np.isnan(au07_prod)):
        conc_global = (face_df.get("AU04_r", 0) + face_df.get("AU07_r", 0)).mean()
        f["c5_production_concentration"] = float((au04_prod + au07_prod) / (conc_global + 1e-10))
    else:
        f["c5_production_concentration"] = 0.5

    # Evaluation critical engagement: AU04 + AU14 during Evaluation
    au04_eval = au_mean_stage("AU04_r", "Evaluation")
    au14_eval = au_mean_stage("AU14_r", "Evaluation")
    if not (np.isnan(au04_eval) or np.isnan(au14_eval)):
        f["c5_evaluation_critical_engagement"] = float(au04_eval + au14_eval)
    else:
        f["c5_evaluation_critical_engagement"] = 0.0

    # Stage emotion profile fit (cosine similarity to ideal profiles)
    # Ideal: Ideation: high Happiness/low Stress; Production: high AU04+AU07/moderate Happiness;
    #        Block: moderate AU04/low Anger; Evaluation: high AU04+AU14
    ideal_profiles = {
        "Ideation":   {"Happiness": 1.0, "Stress": -0.5, "Anger": -0.5},
        "Production": {"AU04_r": 1.0, "AU07_r": 0.7, "Happiness": 0.3},
        "Evaluation": {"AU04_r": 1.0, "AU14_r": 0.8, "Stress": 0.3},
        BLOCK_STAGE:  {"AU04_r": 0.5, "Anger": -0.5},
    }
    fit_scores = []
    for stage_name, ideal in ideal_profiles.items():
        observed = {}
        for feat, weight in ideal.items():
            col = feat if feat in face_df.columns else None
            if col:
                m_obs = face_df.loc[face_df["stage"] == stage_name, col]
                if len(m_obs) > 5:
                    obs_z = (m_obs.mean() - face_df[col].mean()) / (face_df[col].std() + 1e-10)
                    observed[feat] = obs_z
        if len(observed) >= 2:
            ideal_v = np.array([ideal[k] for k in observed])
            obs_v   = np.array([observed[k] for k in observed])
            ni, no = np.linalg.norm(ideal_v), np.linalg.norm(obs_v)
            if ni > 0 and no > 0:
                fit_scores.append(np.dot(ideal_v, obs_v) / (ni * no))
    f["c5_stage_emotion_profile_fit"] = float(np.mean(fit_scores)) if fit_scores else 0.0

    # ── Emotion-stage coupling ───────────────────────────────────────────────
    # Emotion-stage differentiation: η² from one-way ANOVA per emotion, averaged
    eta2_vals = []
    for emo in EMOTION_COLS:
        groups = [face_df.loc[face_df["stage"] == s, emo].values
                  for s in stage_df["stage"].unique()
                  if len(face_df.loc[face_df["stage"] == s, emo]) > 5]
        if len(groups) >= 2:
            try:
                fval, _ = stats.f_oneway(*groups)
                grand_mean = face_df[emo].mean()
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
                ss_total   = np.sum((face_df[emo].values - grand_mean)**2)
                eta2 = ss_between / (ss_total + 1e-10)
                eta2_vals.append(np.clip(eta2, 0, 1))
            except Exception:
                pass
    f["c5_emotion_stage_differentiation"] = float(np.mean(eta2_vals)) if eta2_vals else 0.0

    # Cognitive effort deployment: AU04 in Production+Evaluation vs Preparation+Refinement
    au04_high = np.nanmean([au_mean_stage("AU04_r", s) for s in ["Production","Evaluation"] if not np.isnan(au_mean_stage("AU04_r", s))])
    au04_low  = np.nanmean([au_mean_stage("AU04_r", s) for s in ["Preparation","Refinement"] if not np.isnan(au_mean_stage("AU04_r", s))])
    f["c5_cognitive_effort_deployment"] = float(au04_high - au04_low) if not np.isnan(au04_high) and not np.isnan(au04_low) else 0.0

    # Interest during ideation: AU01+AU02+AU07 during Ideation vs Production
    def au_sum_stage(aus, stage_name):
        cols = [a for a in aus if a in face_df.columns]
        if not cols:
            return np.nan
        m = face_df.loc[face_df["stage"] == stage_name, cols].mean().mean()
        return m if len(face_df.loc[face_df["stage"] == stage_name]) > 5 else np.nan
    interest_ide  = au_sum_stage(["AU01_r","AU02_r","AU07_r"], "Ideation")
    interest_prod = au_sum_stage(["AU01_r","AU02_r","AU07_r"], "Production")
    f["c5_interest_during_ideation"] = float(interest_ide - interest_prod) if not (np.isnan(interest_ide) or np.isnan(interest_prod)) else 0.0

    # ── Flow state ───────────────────────────────────────────────────────────
    # Flow proxy: z(AU07_prod) + z(Happiness_prod) - z(Stress_prod) - z(SD_AUs_prod)
    prod_mask = face_df["stage"] == "Production"
    if prod_mask.sum() > 20:
        def zs(val, full_series):
            return (val - full_series.mean()) / (full_series.std() + 1e-10)
        flow_components = [
            zs(face_df.loc[prod_mask, "AU07_r"].mean(), face_df["AU07_r"]),
            zs(face_df.loc[prod_mask, "Happiness"].mean(), face_df["Happiness"]),
            -zs(face_df.loc[prod_mask, "Stress"].mean(), face_df["Stress"]),
            -zs(face_df.loc[prod_mask, AU_COLS].std().mean(), face_df[AU_COLS].std()),
        ]
        f["c5_flow_proxy_production"] = float(np.mean(flow_components))
    else:
        f["c5_flow_proxy_production"] = 0.0

    # Flow stability: negated CV of flow proxy in 5-second windows during Production
    if prod_mask.sum() > 50:
        flow_vals = []
        prod_frames = face_df.loc[prod_mask].reset_index(drop=True)
        win = int(5 * fps)
        for i in range(0, len(prod_frames) - win, win):
            chunk = prod_frames.iloc[i:i+win]
            fv = (chunk["AU07_r"].mean() + chunk["Happiness"].mean()
                  - chunk["Stress"].mean() - chunk[AU_COLS].std().mean())
            flow_vals.append(fv)
        if flow_vals and np.mean(flow_vals) != 0:
            f["c5_flow_stability"] = float(-np.std(flow_vals) / (abs(np.mean(flow_vals)) + 1e-10))
        else:
            f["c5_flow_stability"] = 0.0
    else:
        f["c5_flow_stability"] = 0.0

    # ── Mastery mode ─────────────────────────────────────────────────────────
    def mastery_mode(frames):
        happy_mean = frames["Happiness"].mean(); stress_mean = frames["Stress"].mean()
        anger_mean = frames["Anger"].mean(); fear_mean = frames["Fear"].mean()
        win_s = int(5 * fps)
        mastery_count, coping_count = 0, 0
        for i in range(0, len(frames) - win_s, win_s):
            ch = frames.iloc[i:i+win_s]
            h = ch["Happiness"].mean(); s = ch["Stress"].mean()
            a = ch["Anger"].mean(); fe = ch["Fear"].mean()
            au12 = ch["AU12_r"].mean() if "AU12_r" in ch.columns else 0
            if h > happy_mean and s < stress_mean and au12 > 0.5:
                mastery_count += 1
            elif s > stress_mean and h < happy_mean and (a > anger_mean or fe > fear_mean):
                coping_count += 1
        total = mastery_count + coping_count
        return mastery_count / total if total > 0 else 0.5

    f["c5_mastery_mode_ratio"] = mastery_mode(face_df)
    challenge_mask = face_df["stage"].isin(["Production","Evaluation"])
    if challenge_mask.sum() > 50:
        f["c5_mastery_during_challenge"] = mastery_mode(face_df.loc[challenge_mask])
    else:
        f["c5_mastery_during_challenge"] = f["c5_mastery_mode_ratio"]

    # ── Confusion-resolution ─────────────────────────────────────────────────
    # Confusion episodes: AU04 > 75th pct AND AU01 > median, resolved in 15s
    if "AU04_r" in face_df.columns and "AU01_r" in face_df.columns:
        au04_75 = face_df["AU04_r"].quantile(0.75)
        au01_med = face_df["AU01_r"].median()
        au07_med = face_df["AU07_r"].median() if "AU07_r" in face_df.columns else 0
        confused = ((face_df["AU04_r"] > au04_75) & (face_df["AU01_r"] > au01_med))
        n_resolved = 0; n_confused = 0
        fps_int = int(fps)
        for idx in np.where(confused.values)[0][::fps_int]:
            n_confused += 1
            end_idx = min(idx + int(15 * fps), len(face_df))
            window_ahead = face_df.iloc[idx:end_idx]
            if len(window_ahead) > 5:
                au04_dec = window_ahead["AU04_r"].iloc[-1] < face_df["AU04_r"].iloc[idx]
                au07_el  = window_ahead["AU07_r"].mean() > au07_med if "AU07_r" in window_ahead.columns else False
                if au04_dec and au07_el:
                    n_resolved += 1
        f["c5_confusion_resolution_rate_d5"] = float(n_resolved / (n_confused + 1e-10))
    else:
        f["c5_confusion_resolution_rate_d5"] = 0.5

    # Block constructive confusion: AU04+AU01 vs Anger during Block
    block_mask = face_df["stage"] == BLOCK_STAGE
    if block_mask.any():
        au04_block = face_df.loc[block_mask, "AU04_r"].mean() if "AU04_r" in face_df.columns else 0
        au01_block = face_df.loc[block_mask, "AU01_r"].mean() if "AU01_r" in face_df.columns else 0
        anger_block = face_df.loc[block_mask, "Anger"].mean()
        f["c5_block_constructive_confusion"] = float((au04_block + au01_block) - anger_block)
    else:
        f["c5_block_constructive_confusion"] = 0.0

    # ── Stage transition quality ──────────────────────────────────────────────
    # Transition emotional preparation: neg Euclidean distance at stage boundaries
    trans_dists = []
    for i in range(len(stage_df) - 1):
        if stage_df.iloc[i]["stage"] == stage_df.iloc[i+1]["stage"]:
            continue
        t_end = stage_df.iloc[i]["end_sec"]
        t_start = stage_df.iloc[i+1]["start_sec"]
        w15s = 15.0
        before = _get_facial_window(face_df, t_end - w15s, t_end)
        after  = _get_facial_window(face_df, t_start, t_start + w15s)
        if len(before) > 5 and len(after) > 5:
            b_emo = before[EMOTION_COLS].mean().values
            a_emo = after[EMOTION_COLS].mean().values
            dist  = np.linalg.norm(b_emo - a_emo)
            trans_dists.append(dist)
    f["c5_transition_emotional_preparation"] = float(-np.mean(trans_dists)) if trans_dists else 0.0

    # Pride after production (first 30s Evaluation - last 30s Production)
    prod_rows = stage_df[stage_df["stage"] == "Production"]
    eval_rows = stage_df[stage_df["stage"] == "Evaluation"]
    if not prod_rows.empty and not eval_rows.empty:
        last_prod_end = prod_rows.iloc[-1]["end_sec"]
        pride_prod_last = _get_facial_window(face_df, last_prod_end - 30, last_prod_end)["Pride"].mean()
        first_eval_start = eval_rows.iloc[0]["start_sec"]
        pride_eval_first = _get_facial_window(face_df, first_eval_start, first_eval_start + 30)["Pride"].mean()
        f["c5_pride_after_production"] = float(pride_eval_first - pride_prod_last) if not np.isnan(pride_eval_first) and not np.isnan(pride_prod_last) else 0.0
    else:
        f["c5_pride_after_production"] = 0.0

    # ── Task-appropriate arousal (Yerkes-Dodson) ─────────────────────────────
    # Stress appropriate modulation: negated MAD from optimal stress per stage
    stage_optimal_stress = {
        "Preparation": 0.5, "Ideation": 1.0, "Production": 1.5,
        "Evaluation": 2.0,  "Refinement": 1.0, BLOCK_STAGE: 2.5
    }
    stress_global_std = face_df["Stress"].std()
    deviations = []
    for stage_name, opt in stage_optimal_stress.items():
        m = face_df.loc[face_df["stage"] == stage_name, "Stress"]
        if len(m) > 5:
            obs_z = (m.mean() - face_df["Stress"].mean()) / (stress_global_std + 1e-10)
            opt_z = (opt - face_df["Stress"].mean()) / (stress_global_std + 1e-10)
            deviations.append(abs(obs_z - opt_z))
    f["c5_stress_appropriate_modulation"] = float(-np.mean(deviations)) if deviations else 0.0

    # Arousal SAM task match: Spearman r between SAM arousal and stage complexity at checkpoints
    stage_complexity = {"Preparation": 2, "Ideation": 3, "Production": 4,
                        "Evaluation": 4, BLOCK_STAGE: 5, "Refinement": 3}
    complexity_at_checkpoints = []
    for t in sam_times:
        w = _get_facial_window(face_df, max(0, t-90), t)
        if len(w) > 5:
            dom_stage = w["stage"].mode().iloc[0] if not w["stage"].mode().empty else "Production"
            complexity_at_checkpoints.append(stage_complexity.get(dom_stage, 3))
        else:
            complexity_at_checkpoints.append(3)
    valid = ~np.isnan(sam_aro)
    if valid.sum() >= 3:
        try:
            rho, _ = spearmanr(sam_aro[valid], np.array(complexity_at_checkpoints)[valid])
            f["c5_arousal_sam_task_match"] = float(rho) if not np.isnan(rho) else 0.0
        except Exception:
            f["c5_arousal_sam_task_match"] = 0.0
    else:
        f["c5_arousal_sam_task_match"] = 0.0

    # ── Cross-stage differentiation ──────────────────────────────────────────
    # Positive creative vs focused production crossover
    hap_ideation = stage_mean("Happiness", "Ideation")
    hap_production = stage_mean("Happiness", "Production")
    au04_07_prod = (au_mean_stage("AU04_r","Production") or 0) + (au_mean_stage("AU07_r","Production") or 0)
    au04_07_ide  = (au_mean_stage("AU04_r","Ideation")  or 0) + (au_mean_stage("AU07_r","Ideation")  or 0)
    part1 = (hap_ideation - hap_production) if not (np.isnan(hap_ideation) or np.isnan(hap_production)) else 0
    part2 = au04_07_prod - au04_07_ide
    f["c5_positive_creative_vs_focused_prod"] = float(part1 + part2)

    # Negative affect block specificity
    neg_block    = face_df.loc[face_df["stage"] == BLOCK_STAGE, NEGATIVE_EMOS].mean().mean() if block_mask.any() else 0.0
    neg_nonblock = face_df.loc[~block_mask, NEGATIVE_EMOS].mean().mean()
    f["c5_negative_affect_block_specificity"] = float(neg_block - neg_nonblock)

    # Emotion complexity index: mean number of emotion composites > 1.0 per 10s window
    win_10s = int(10 * fps)
    complexity_vals = []
    for i in range(0, len(face_df) - win_10s, win_10s):
        chunk = face_df.iloc[i:i+win_10s]
        n_active = (chunk[EMOTION_COLS] > 1.0).any().sum()
        complexity_vals.append(n_active / 7)
    f["c5_emotion_complexity_index"] = float(np.mean(complexity_vals)) if complexity_vals else 0.5

    # SAM-AU coherence: r between SAM valence and (Happiness-Sadness) + r between SAM arousal and (Stress+Fear)
    hap_sad_checkpoints, stress_fear_checkpoints = [], []
    for t in sam_times:
        w = _get_facial_window(face_df, max(0, t-180), t)
        if len(w) > 5:
            hap_sad_checkpoints.append((w["Happiness"] - w["Sadness"]).mean())
            stress_fear_checkpoints.append((w["Stress"] + w["Fear"]).mean())
        else:
            hap_sad_checkpoints.append(np.nan)
            stress_fear_checkpoints.append(np.nan)
    r1 = _safe_pearsonr(sam_val, np.array(hap_sad_checkpoints))
    r2 = _safe_pearsonr(sam_aro, np.array(stress_fear_checkpoints))
    f["c5_sam_au_coherence"] = float((r1 + r2) / 2)

    # Engagement recovery speed: after high-neg episodes, time until re-engagement
    stress_75 = face_df["Stress"].quantile(0.75)
    anger_75  = face_df["Anger"].quantile(0.75)
    au07_med  = face_df["AU07_r"].median() if "AU07_r" in face_df.columns else 0
    stress_med = face_df["Stress"].median()
    recovery_times = []
    high_neg = ((face_df["Stress"] > stress_75) | (face_df["Anger"] > anger_75)).values
    i = 0
    while i < len(high_neg):
        if high_neg[i]:
            start_i = i
            while i < len(high_neg) and high_neg[i]:
                i += 1
            # Find recovery
            found = False
            for j in range(i, min(i + int(30 * fps), len(face_df))):
                au07_val = face_df.iloc[j]["AU07_r"] if "AU07_r" in face_df.columns else 0
                if au07_val > au07_med and face_df.iloc[j]["Stress"] < stress_med:
                    recovery_times.append((j - i) / fps)
                    found = True
                    break
            if not found:
                recovery_times.append(30.0)
        else:
            i += 1
    f["c5_engagement_recovery_speed"] = float(-np.mean(recovery_times)) if recovery_times else 0.0

    return f


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION LOOP
# ════════════════════════════════════════════════════════════════════════════

def extract_all(pid, data):
    face_df  = data["face_df"]
    stage_df = data["stage_df"]
    sam_val  = data["sam_valence"]
    sam_aro  = data["sam_arousal"]

    row = {"pid": pid}
    row.update(extract_D1(face_df, stage_df, sam_val, sam_aro))
    row.update(extract_D2(face_df, stage_df, sam_val, sam_aro))
    row.update(extract_D3(face_df, stage_df, sam_val, sam_aro))
    row.update(extract_D4(face_df, stage_df, sam_val, sam_aro))
    row.update(extract_D5(face_df, stage_df, sam_val, sam_aro))
    return row


def run(aligned=None):
    print("=" * 60)
    print("STAGE 2 — CANDIDATE FEATURE EXTRACTION (116 features)")
    print("=" * 60)

    if aligned is None:
        pkl_path = os.path.join(OUTPUT_DIR, "aligned_data.pkl")
        with open(pkl_path, "rb") as fh:
            aligned = pickle.load(fh)

    records, errors = [], []
    for pid in sorted(aligned.keys()):
        try:
            row = extract_all(pid, aligned[pid])
            records.append(row)
            print(f"  P{pid:02d}: {len(row)-1} features extracted")
        except Exception as e:
            import traceback
            errors.append((pid, str(e)))
            print(f"  P{pid:02d}: ERROR — {e}")
            traceback.print_exc()

    feat_df = pd.DataFrame(records).set_index("pid")
    print(f"\nFeature matrix: {feat_df.shape[0]} participants × {feat_df.shape[1]} features")
    print(f"Errors: {len(errors)}")
    print("\nMissing values per feature (top 10):")
    print(feat_df.isnull().sum().sort_values(ascending=False).head(10))

    # Save
    csv_path = os.path.join(OUTPUT_DIR, "feature_matrix_raw.csv")
    pkl_path = os.path.join(OUTPUT_DIR, "feature_matrix_raw.pkl")
    feat_df.to_csv(csv_path)
    with open(pkl_path, "wb") as fh:
        pickle.dump(feat_df, fh)
    print(f"\nSaved → {csv_path}")
    return feat_df


if __name__ == "__main__":
    run()
