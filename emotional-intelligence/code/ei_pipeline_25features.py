# -*- coding: utf-8 -*-
"""
EI Pipeline  --  25 Selected Features Only
===========================================
Exactly the 25 features selected from the candidate feature menu,
implemented with the formulas described in the feature-selection session.

COMPOSITES  (Project Brief definitions)
  Happiness = AU06 + AU12
  Anger     = AU04 + AU05 + AU07 + AU23
  Fear      = AU01 + AU04 + AU05
  Sadness   = AU01 + AU04 + AU15
  Disgust   = AU04 + AU07 + AU09 + AU10 + AU17 + AU20
  Pride     = AU06 + AU07 + AU12
  Stress    = AU01 + AU06 + AU07 + AU15
  NegAff    = mean(z-Anger, z-Fear, z-Sadness, z-Disgust)

All AU values are continuous regression outputs (0-5 range).

FEATURE MAP  (25 features)
  D2  Emotion Regulation       [8]  : F01-F08
  D3  Adaptive Persistence     [8]  : F09-F16
  D5  Emotion-Cognition Align  [9]  : F17-F25
"""

import os, re, warnings, traceback
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, ElasticNetCV, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
AU_DIR     = "D:/A_emotions in exam/data/facial_expressions/facial expressions- data"
STAGE_DIR  = "D:/A_emotions in exam/data/stage_coded_data/stage coded data"
SAM_FILE   = "D:/A_emotions in exam/data/SAM Responses every 3 minutes.xlsx"
SCORE_FILE = "D:/A_emotions in exam/data/Exam Scores.xlsx"
OUT_DIR    = "D:/A_emotions in exam/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
AU_COLS = [
    "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r",
    "AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r",
    "AU20_r","AU23_r","AU25_r","AU26_r","AU45_r"
]

STAGES    = ["Preparation","Ideation","Production","Evaluation","Refinement","Block"]
STAGE_ORD = {"Preparation":1,"Ideation":2,"Production":3,
             "Evaluation":4,"Refinement":5,"Block":0}
PROD_SET  = {"Production","Evaluation","Refinement"}

# Optimal Stress z-score per stage for F22 (Yerkes-Dodson matching)
OPT_STRESS = {
    "Preparation":-0.5, "Ideation":-0.5, "Production": 0.0,
    "Evaluation": 0.5,  "Refinement":-0.5, "Block": 1.0
}

# The 25 feature column names in computation order
F_COLS = [
    # -- D2 : Emotion Regulation (F01-F08) --
    "F01_stress_recovery_latency",        # negatively directed
    "F02_stress_inertia",                 # negatively directed
    "F03_stress_instability_MSSD",        # negatively directed
    "F04_stress_habituation_ratio",       # negatively directed
    "F05_block_regulation_efficiency",
    "F06_production_stress_floor",        # negatively directed
    "F07_anticipatory_regulation",
    "F08_reappraisal_coactivation",
    # -- D3 : Adaptive Persistence (F09-F16) --
    "F09_block_to_production_rate",
    "F10_recovery_acceleration",
    "F11_productive_absorbing_prob",
    "F12_productive_time_ratio",
    "F13_task_completion_persistence",
    "F14_late_engagement_ratio",
    "F15_bounce_back_index",
    "F16_forward_stage_momentum",
    # -- D5 : Emotion-Cognition Alignment (F17-F25) --
    "F17_ideation_positive_affect",
    "F18_production_concentration",
    "F19_evaluation_critical_engagement",
    "F20_flow_proxy_production",
    "F21_mastery_mode_ratio",
    "F22_stress_appropriate_modulation",
    "F23_confusion_resolution_rate",
    "F24_block_constructive_confusion",
    "F25_engagement_recovery_speed",
]

# Features where a LOWER raw value means better EI
#  -> sign-flipped before feature selection so all point "higher = better"
NEG_DIR = {
    "F01_stress_recovery_latency",
    "F02_stress_inertia",
    "F03_stress_instability_MSSD",
    "F04_stress_habituation_ratio",
    "F06_production_stress_floor",
}


# ==================================================================
# SECTION 1  --  DATA LOADING
# ==================================================================

def _pid(fname):
    m = re.match(r"(P\d+)", fname, re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_au(pid):
    """
    Returns DataFrame: time_s (relative seconds from 0)  +  AU01_r..AU45_r
    """
    files = [f for f in os.listdir(AU_DIR)
             if f.endswith(".xlsx") and _pid(f) == pid]
    if not files:
        return None
    df = pd.read_excel(os.path.join(AU_DIR, files[0]), engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    ts = next((c for c in df.columns if "timestamp" in c.lower()), None)
    if ts is None:
        return None

    present = [c for c in AU_COLS if c in df.columns]
    df = df[[ts] + present].copy()
    df.columns = ["ts"] + present
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df["time_s"] = df["ts"] - df["ts"].iloc[0]

    for c in AU_COLS:          # guarantee all 17 cols exist
        if c not in df.columns:
            df[c] = 0.0
    return df[["time_s"] + AU_COLS].copy()


def load_stage(pid):
    """
    Returns DataFrame: start_s | end_s | duration_s | stage
    """
    files = [f for f in os.listdir(STAGE_DIR)
             if f.endswith(".xlsx") and _pid(f) == pid]
    if not files:
        return None
    df = pd.read_excel(os.path.join(STAGE_DIR, files[0]), engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    end_col  = next((c for c in df.columns if "end"  in c.lower()), None)
    code_col = next((c for c in df.columns if "code" in c.lower()), None)
    if end_col is None or code_col is None:
        return None

    ends = []
    for v in df[end_col]:
        if hasattr(v, "total_seconds"):
            # timedelta object
            ends.append(v.total_seconds())
        elif hasattr(v, "hour"):
            # datetime.time object (HH:MM:SS format)
            try:
                ends.append(v.hour * 3600 + v.minute * 60 + v.second)
            except:
                ends.append(np.nan)
        else:
            # try numeric conversion as fallback
            try:
                ends.append(float(v))
            except:
                ends.append(np.nan)

    starts = [0.0] + ends[:-1]

    SMAP = {
        "preparation":"Preparation","prep":"Preparation",
        "ideation":"Ideation",
        "production":"Production","prod":"Production",
        "evaluation":"Evaluation","eval":"Evaluation",
        "refinement":"Refinement","refine":"Refinement","refinery":"Refinement",
        "block":"Block",
    }
    codes = []
    for v in df[code_col]:
        raw    = str(v).strip().lower()
        mapped = next((val for key, val in SMAP.items() if key in raw), None)
        codes.append(mapped)

    out = pd.DataFrame({
        "start_s": starts, "end_s": ends,
        "duration_s": [e-s if (pd.notna(e) and pd.notna(s)) else np.nan
                       for e,s in zip(ends,starts)],
        "stage": codes
    })
    return out.dropna(subset=["stage","end_s"]).query("duration_s > 0").reset_index(drop=True)


def load_scores():
    df = pd.read_excel(SCORE_FILE, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        m = re.match(r"(P\d+)", str(r.iloc[0]), re.IGNORECASE)
        if not m: continue
        pid   = m.group(1).upper()
        score = r.iloc[1];  total = r.iloc[2] if len(r) > 2 else 26
        try:    pct = float(score)/float(total)*100
        except: pct = np.nan
        rows.append({"pid":pid,"score":score,"total":total,"pct":pct})
    return pd.DataFrame(rows)


# ==================================================================
# SECTION 2  --  AU COMPOSITES
# ==================================================================

def add_composites(df):
    """
    Add 7 emotion composites + NegAff to an AU DataFrame.
    All are sums of continuous AU regression intensities (0-5 scale).
    """
    d = df.copy()
    d["Happiness"] = d["AU06_r"] + d["AU12_r"]
    d["Anger"]     = d["AU04_r"] + d["AU05_r"] + d["AU07_r"] + d["AU23_r"]
    d["Fear"]      = d["AU01_r"] + d["AU04_r"] + d["AU05_r"]
    d["Sadness"]   = d["AU01_r"] + d["AU04_r"] + d["AU15_r"]
    d["Disgust"]   = d["AU04_r"] + d["AU07_r"] + d["AU09_r"] + d["AU10_r"] + d["AU17_r"] + d["AU20_r"]
    d["Pride"]     = d["AU06_r"] + d["AU07_r"] + d["AU12_r"]
    d["Stress"]    = d["AU01_r"] + d["AU06_r"] + d["AU07_r"] + d["AU15_r"]
    for c in ["Anger","Fear","Sadness","Disgust"]:
        mu, sd    = d[c].mean(), d[c].std()
        d[f"z{c}"] = (d[c]-mu)/(sd if sd>0 else 1e-6)
    d["NegAff"] = d[["zAnger","zFear","zSadness","zDisgust"]].mean(axis=1)
    return d


def tag_stages(au, stage_df):
    """Add 'stage' column to AU DataFrame by matching time_s to stage intervals."""
    labels = np.full(len(au), None, dtype=object)
    t = au["time_s"].values
    for _, row in stage_df.iterrows():
        labels[(t >= row["start_s"]) & (t < row["end_s"])] = row["stage"]
    au = au.copy()
    au["stage"] = labels
    return au


# ==================================================================
# SECTION 3  --  HELPERS
# ==================================================================

def _div(a, b):
    try:
        b = float(b)
        return float(a)/b if b != 0 else np.nan
    except: return np.nan

def _ar1(arr):
    """AR(1) coefficient of a detrended series."""
    s = np.asarray(arr, float); s = s[~np.isnan(s)]
    if len(s) < 6: return np.nan
    idx = np.arange(len(s))
    s   = s - np.polyval(np.polyfit(idx, s, 1), idx)   # detrend
    return float(np.corrcoef(s[:-1], s[1:])[0,1])

def _mssd(arr):
    """Mean Squared Successive Differences."""
    s = np.asarray(arr, float); s = s[~np.isnan(s)]
    return float(np.mean(np.diff(s)**2)) if len(s)>=2 else np.nan

def _1hz(au, col):
    """Downsample column to 1 Hz by taking 1-second bin means."""
    tmp = au[["time_s", col]].copy()
    tmp["bin"] = tmp["time_s"].astype(int)
    return tmp.groupby("bin")[col].mean().values

def _zmean(au, col, stage):
    """Within-participant z-score of the mean of col during stage."""
    v = au[au["stage"]==stage][col].values
    if len(v) < 5: return np.nan
    mu, sd = au[col].mean(), au[col].std()
    return float((np.nanmean(v) - mu) / (sd if sd>0 else 1e-6))


# ==================================================================
# SECTION 4  --  COMPUTE 25 FEATURES
# ==================================================================

def compute_25(pid, au, stage_df):
    """
    Compute exactly the 25 selected EI features for one participant.

    Notation used in formula comments
    ----------------------------------
    S(t)   = Stress composite at frame t
    H(t)   = Happiness composite
    A(t)   = Anger composite
    mu_X   = participant-level mean of X
    sd_X   = participant-level std  of X
    fps    = frames per second (~25)
    T      = total task duration in seconds
    seq    = ordered list of stage labels
    """
    feat = {"pid": pid}

    # --- raw arrays ---
    t    = au["time_s"].values
    S    = au["Stress"].values
    H    = au["Happiness"].values
    A    = au["Anger"].values
    F_   = au["Fear"].values
    au4  = au["AU04_r"].values
    au7  = au["AU07_r"].values
    au1  = au["AU01_r"].values
    au12 = au["AU12_r"].values

    mu_S  = np.nanmean(S);   sd_S  = np.nanstd(S)  or 1e-6
    mu_H  = np.nanmean(H);   mu_A  = np.nanmean(A)
    mu_F  = np.nanmean(F_);  mu_au7= np.nanmean(au7)
    T     = float(t[-1] - t[0]) if len(t)>1 else np.nan
    fps   = len(t)/T if T and T>0 else 25.0

    # block episode (start_s, end_s) pairs
    blk_segs = [(r["start_s"], r["end_s"])
                for _, r in stage_df.iterrows() if r["stage"]=="Block"]
    # stage sequence (canonical labels only)
    seq = [s for s in stage_df["stage"].tolist() if s in STAGES]

    # ================================================================
    # D2  --  EMOTION REGULATION  (F01-F08)
    # ================================================================

    # F01  c2_stress_recovery_latency
    # Formula: L = mean( t_recover_k - t_peak_k )
    #   peak: S(t*) > mu_S + sd_S
    #   recovery: first t > t* where S(t) < mu_S + 0.5*sd_S
    # Expected: NEGATIVE (lower latency = faster recovery = better)
    lats = []
    i = 0
    while i < len(S):
        if S[i] > mu_S + sd_S:
            pk  = i
            rec = next((j for j in range(pk+1,len(S))
                        if S[j] < mu_S + 0.5*sd_S), None)
            if rec is not None:
                lats.append(t[rec] - t[pk])
            i = (rec+1) if rec else pk+1
        else:
            i += 1
    feat["F01_stress_recovery_latency"] = np.mean(lats) if lats else np.nan

    # F02  c2_stress_inertia
    # Formula: alpha_hat = corr(S_tilde(t), S_tilde(t-1))  at 1 Hz
    #          S_tilde = detrended Stress
    # Expected: NEGATIVE (lower inertia = less stuck emotion)
    feat["F02_stress_inertia"] = _ar1(_1hz(au,"Stress"))

    # F03  c2_stress_instability_MSSD
    # Formula: MSSD = mean( [S(t) - S(t-1)]^2 )  at 1 Hz
    # Expected: NEGATIVE (lower = more stable stress regulation)
    feat["F03_stress_instability_MSSD"] = _mssd(_1hz(au,"Stress"))

    # F04  c2_stress_habituation_ratio
    # Formula: R = mean(S in 2nd half) / mean(S in 1st half)
    # Expected: NEGATIVE (R < 1 means habituated to stress)
    mid = len(S)//2
    feat["F04_stress_habituation_ratio"] = _div(np.nanmean(S[mid:]), np.nanmean(S[:mid]))

    # F05  c2_block_stage_regulation_efficiency
    # Formula: BRE = mean( -slope_b )
    #   slope_b = OLS slope of S(t) ~ t within each Block episode b
    # Expected: POSITIVE (steeper drop inside block = faster recovery)
    slopes = []
    for bs,be in blk_segs:
        m = (t>=bs)&(t<be)
        if m.sum() < 4: continue
        tb = t[m] - t[m][0]
        slopes.append(-np.polyfit(tb, S[m], 1)[0])
    feat["F05_block_regulation_efficiency"] = np.mean(slopes) if slopes else np.nan

    # F06  c2_production_stress_floor
    # Formula: PSF = median( S(t)  for t in T_Production )
    # Expected: NEGATIVE (lower floor = calmer while productively working)
    pm = au["stage"]=="Production"
    feat["F06_production_stress_floor"] = (
        float(np.nanmedian(S[pm.values])) if pm.sum()>0 else np.nan)

    # F07  c2_anticipatory_regulation
    # Formula: AR = mean( mu_S - mean(S in [onset-30s, onset]) )
    #   over all Block and Evaluation stage onsets
    # Expected: POSITIVE (lower pre-challenge stress = proactive regulation)
    ant = []
    for _, row in stage_df.iterrows():
        if row["stage"] in ("Block","Evaluation"):
            mask = (t >= row["start_s"]-30) & (t < row["start_s"])
            if mask.sum() > 0:
                ant.append(mu_S - np.nanmean(S[mask]))
    feat["F07_anticipatory_regulation"] = np.mean(ant) if ant else np.nan

    # F08  c2_reappraisal_proxy_coactivation
    # Formula: RPC = sum( 1[AU04>1 AND AU12>1]  for t in E ) / |E|
    #   E = stress-peak frames where S(t) > mu_S + sd_S
    # Expected: POSITIVE (effort + smile during stress = cognitive reappraisal)
    E    = S > mu_S + sd_S
    coact = (au4 > 1.0) & (au12 > 1.0)
    feat["F08_reappraisal_coactivation"] = _div(np.sum(E & coact), np.sum(E))

    # ================================================================
    # D3  --  ADAPTIVE PERSISTENCE  (F09-F16)
    # ================================================================

    # F09  c3_block_to_production_rate
    # Formula: P(Block->Production) = #(Block->Production) / #(Block->any)
    # Expected: POSITIVE (direct return to productive work after block)
    exits = [seq[i+1] for i in range(len(seq)-1) if seq[i]=="Block"]
    feat["F09_block_to_production_rate"] = (
        _div(exits.count("Production"), len(exits)) if exits else np.nan)

    # F10  c3_recovery_acceleration
    # Formula: RA = -beta_1
    #   OLS: duration_k = beta_0 + beta_1 * k   (k = ordinal block number)
    # Expected: POSITIVE (blocks getting shorter = learning to cope)
    bdurs = stage_df[stage_df["stage"]=="Block"]["duration_s"].values
    if len(bdurs) >= 3:
        ks = np.arange(1, len(bdurs)+1)
        feat["F10_recovery_acceleration"] = float(-np.polyfit(ks, bdurs, 1)[0])
    else:
        feat["F10_recovery_acceleration"] = np.nan

    # F11  c3_productive_absorbing_prob
    # Formula: PAP = sum( pi_s  for s in {Production, Evaluation, Refinement} )
    #   pi = stationary distribution of 6-state stage Markov chain
    # Expected: POSITIVE (structural tendency toward productive work)
    if len(seq) >= 4:
        tc = {s:{s2:0 for s2 in STAGES} for s in STAGES}
        for i in range(len(seq)-1):
            if seq[i] in STAGES and seq[i+1] in STAGES:
                tc[seq[i]][seq[i+1]] += 1
        M = np.eye(len(STAGES))
        for i,s1 in enumerate(STAGES):
            rs = sum(tc[s1].values())
            if rs > 0:
                M[i] = [tc[s1][s2]/rs for s2 in STAGES]
        pi = np.ones(len(STAGES))/len(STAGES)
        for _ in range(2000):
            pn = pi @ M
            if np.allclose(pi,pn,atol=1e-9): break
            pi = pn
        pidx = [STAGES.index(s) for s in ["Production","Evaluation","Refinement"]]
        feat["F11_productive_absorbing_prob"] = float(pi[pidx].sum())
    else:
        feat["F11_productive_absorbing_prob"] = np.nan

    # F12  c3_total_productive_time_ratio
    # Formula: PTR = sum(duration_s for s in {Prod,Eval,Refine}) / T_total
    # Expected: POSITIVE
    pdur = stage_df[stage_df["stage"].isin(PROD_SET)]["duration_s"].sum()
    feat["F12_productive_time_ratio"] = _div(pdur, T)

    # F13  c3_task_completion_persistence
    # Formula: TCP = productive_time_in_[T-180s, T] / 180
    # Expected: POSITIVE (sustained effort near end of exam)
    if T:
        n = ((au["time_s"] >= t[-1]-180) & au["stage"].isin(PROD_SET)).sum()
        feat["F13_task_completion_persistence"] = _div(n/fps, 180)
    else:
        feat["F13_task_completion_persistence"] = np.nan

    # F14  c3_late_task_engagement_ratio
    # Formula: LER = productive_frames_in_last_third / productive_frames_in_first_third
    # Expected: POSITIVE (sustained engagement across session)
    t1, t2 = T/3, 2*T/3
    n1 = ((au["time_s"] < t1)  & au["stage"].isin(PROD_SET)).sum()
    n3 = ((au["time_s"] >= t2) & au["stage"].isin(PROD_SET)).sum()
    feat["F14_late_engagement_ratio"] = _div(n3, n1)

    # F15  c3_bounce_back_index
    # Formula: BBI = mean( mean_H[be, be+60s] / mean_H[bs-60s, bs] )
    #   over all Block episodes (bs=block_start, be=block_end)
    # Expected: POSITIVE (Happiness recovers after blocks)
    bbs = []
    for bs,be in blk_segs:
        pre  = au[(au["time_s"]>=bs-60) & (au["time_s"]<bs)]
        post = au[(au["time_s"]>=be)    & (au["time_s"]<be+60)]
        ph   = float(np.nanmean(H[pre.index]))  if len(pre)>0  else np.nan
        qh   = float(np.nanmean(H[post.index])) if len(post)>0 else np.nan
        if ph and ph > 0:
            bbs.append(qh/ph)
    feat["F15_bounce_back_index"] = float(np.nanmean(bbs)) if bbs else np.nan

    # F16  c3_forward_stage_momentum
    # Formula: FSM = mean( ord(stage_{i+1}) - ord(stage_i) )
    #   over all non-Block consecutive transitions
    # Expected: POSITIVE (net forward progress through task pipeline)
    nb = [(seq[i],seq[i+1]) for i in range(len(seq)-1)
          if seq[i]!="Block" and seq[i+1]!="Block"]
    feat["F16_forward_stage_momentum"] = (
        float(np.mean([STAGE_ORD[b]-STAGE_ORD[a] for a,b in nb])) if nb else np.nan)

    # ================================================================
    # D5  --  EMOTION-COGNITION ALIGNMENT  (F17-F25)
    # ================================================================

    # F17  c5_ideation_positive_affect
    # Formula: z[ mean(H(t) for t in T_Ideation) ]
    #   z-scored within participant
    # Expected: POSITIVE (positive affect facilitates creative/divergent thinking)
    feat["F17_ideation_positive_affect"] = _zmean(au,"Happiness","Ideation")

    # F18  c5_production_concentration
    # Formula: z[ mean(AU04(t) + AU07(t)  for t in T_Production) ]
    #   AU04=brow furrow (concentration), AU07=lid tighten (focused attention)
    # Expected: POSITIVE (concentrated effort during work = better output)
    au["_conc"] = au["AU04_r"] + au["AU07_r"]
    feat["F18_production_concentration"] = _zmean(au,"_conc","Production")

    # F19  c5_evaluation_critical_engagement
    # Formula: z[ mean(AU04(t) + AU14(t)  for t in T_Evaluation) ]
    #   AU14=dimpler, linked to learning gains during evaluation
    # Expected: POSITIVE
    au["_criteng"] = au["AU04_r"] + au["AU14_r"]
    feat["F19_evaluation_critical_engagement"] = _zmean(au,"_criteng","Evaluation")

    # F20  c5_flow_proxy_production
    # Formula: FPP = mean( [z(AU07) + z(H) - z(S) - z(SD_AU)] / 4 )
    #   for t in T_Production; SD_AU = std of all 17 AU values at frame t
    # Expected: POSITIVE (flow = high focus+affect, low stress+variability)
    prod_m = au["stage"]=="Production"
    if prod_m.sum() > 20:
        def _zp(col):
            mu_, sd_ = au[col].mean(), au[col].std() or 1e-6
            return (au.loc[prod_m, col] - mu_)/sd_
        sd_all  = au[AU_COLS].std(axis=1)
        mu_sd, sd_sd = sd_all.mean(), sd_all.std() or 1e-6
        z_sd    = (sd_all[prod_m] - mu_sd)/sd_sd
        flow    = (_zp("AU07_r") + _zp("Happiness") - _zp("Stress") - z_sd)/4
        feat["F20_flow_proxy_production"] = float(flow.mean())
    else:
        feat["F20_flow_proxy_production"] = np.nan

    # F21  c5_mastery_mode_ratio
    # In each 5-second window w:
    #   mastery(w) = 1  if mean_H > mu_H  AND  mean_S < mu_S  AND  mean_AU12 > 0.5
    #   coping(w)  = 1  if mean_S > mu_S  AND  mean_H < mu_H  AND  (mean_A>mu_A OR mean_F>mu_F)
    # Formula: MMR = sum(mastery) / (sum(mastery) + sum(coping))
    # Expected: POSITIVE
    WIN = max(2, int(5*fps))
    m_cnt = c_cnt = 0
    for i0 in range(0, len(au)-WIN, WIN):
        w    = au.iloc[i0:i0+WIN]
        wH   = w["Happiness"].mean();  wS  = w["Stress"].mean()
        wA   = w["Anger"].mean();      wF  = w["Fear"].mean()
        w12  = w["AU12_r"].mean()
        if wH > mu_H and wS < mu_S and w12 > 0.5:
            m_cnt += 1
        elif wS > mu_S and wH < mu_H and (wA > mu_A or wF > mu_F):
            c_cnt += 1
    feat["F21_mastery_mode_ratio"] = _div(m_cnt, m_cnt+c_cnt)

    # F22  c5_stress_appropriate_modulation
    # Formula: SAM = -mean( |z(mean_S_stage) - opt_z_stage|  for each stage )
    #   opt_z per stage: Prep=-0.5, Idea=-0.5, Prod=0, Eval=+0.5, Refine=-0.5, Block=+1
    # Expected: POSITIVE (closer to optimal arousal per stage = better performance)
    devs = []
    for sname, opt in OPT_STRESS.items():
        sv = au[au["stage"]==sname]["Stress"].values
        if len(sv) < 5: continue
        devs.append(abs((np.nanmean(sv)-mu_S)/sd_S - opt))
    feat["F22_stress_appropriate_modulation"] = float(-np.mean(devs)) if devs else np.nan

    # F23  c5_confusion_resolution_rate
    # Confusion episode: 3-s window where mean(AU04) > Q75(AU04) AND mean(AU01) > median(AU01)
    # Resolved: within next 15 s, AU04 mean decreases AND AU07 mean stays >= global mean
    # Formula: CRR = resolved / total_confusion_episodes
    # Expected: POSITIVE (productive confusion resolution -> learning)
    q75_au4  = np.nanpercentile(au4, 75)
    med_au1  = np.nanmedian(au1)
    W3  = max(1, int(3*fps));  W15 = max(1, int(15*fps))
    tot_conf = res_conf = 0
    i = 0
    while i < len(au)-W3:
        w = au.iloc[i:i+W3]
        if w["AU04_r"].mean() > q75_au4 and w["AU01_r"].mean() > med_au1:
            tot_conf += 1
            fut = au.iloc[i+W3: i+W3+W15]
            if len(fut) > 0:
                if (fut["AU04_r"].mean() < w["AU04_r"].mean() and
                        fut["AU07_r"].mean() >= au["AU07_r"].mean()):
                    res_conf += 1
            i += W3
        else:
            i += 1
    feat["F23_confusion_resolution_rate"] = _div(res_conf, tot_conf)

    # F24  c5_block_constructive_confusion
    # Formula: BCC = mean( AU04(t) + AU01(t) - Anger(t)  for t in T_Block )
    # Expected: POSITIVE (curiosity/effort exceeds frustration during blocks)
    bm = au["stage"]=="Block"
    feat["F24_block_constructive_confusion"] = (
        float((au.loc[bm,"AU04_r"] + au.loc[bm,"AU01_r"] - au.loc[bm,"Anger"]).mean())
        if bm.sum()>10 else np.nan)

    # F25  c5_engagement_recovery_speed
    # Disruption: S(t)>Q75(S) or A(t)>Q75(A) sustained >= 3 s
    # Recovery at t_r: first frame after episode where AU07>mu_AU07 AND S<mu_S
    # Formula: ERS = 1 / mean(t_r - t_episode_end)
    # Expected: POSITIVE (faster re-engagement = better cognitive resilience)
    q75S = np.nanpercentile(S,75);  q75A = np.nanpercentile(A,75)
    rec_ts = []
    i = 0
    while i < len(au)-int(3*fps):
        if S[i]>q75S or A[i]>q75A:
            ep = i
            while ep < len(au)-1 and (S[ep]>q75S or A[ep]>q75A):
                ep += 1
            for j in range(ep, min(ep+int(60*fps), len(au))):
                if au7[j]>mu_au7 and S[j]<mu_S:
                    rec_ts.append(t[j]-t[ep]);  break
            i = ep+1
        else:
            i += 1
    mr = np.mean(rec_ts) if rec_ts else np.nan
    feat["F25_engagement_recovery_speed"] = (1.0/mr if mr and mr>0 else np.nan)

    return feat


# ==================================================================
# SECTION 5  --  BUILD FEATURE MATRIX  (all 42 participants)
# ==================================================================

def build_matrix():
    scores  = load_scores()
    au_files= sorted([f for f in os.listdir(AU_DIR) if f.endswith(".xlsx")])

    rows, failed = [], []
    for fname in au_files:
        pid = _pid(fname)
        if not pid: continue
        print(f"  {pid} ...", end=" ", flush=True)
        try:
            au_df    = load_au(pid)
            stage_df = load_stage(pid)
            if au_df is None or stage_df is None:
                print("SKIP (file missing)"); failed.append(pid); continue
            au_df = add_composites(au_df)
            au_df = tag_stages(au_df, stage_df)
            rows.append(compute_25(pid, au_df, stage_df))
            print("OK")
        except Exception as e:
            print(f"ERROR -- {e}")
            traceback.print_exc()
            failed.append(pid)

    feat_df = pd.DataFrame(rows)
    feat_df = feat_df.merge(scores[["pid","pct","score"]], on="pid", how="left")
    print(f"\n  Done: {len(feat_df)} participants  |  Failed: {len(failed)}")
    if failed: print(f"  Failed: {failed}")
    feat_df.to_csv(f"{OUT_DIR}/features_25.csv", index=False)
    return feat_df


# ==================================================================
# SECTION 6  --  FEATURE SELECTION  (Revised Validation Pipeline)
# ==================================================================
#
# Based on validation criteria from candidate feature review:
#   Phase 1: Missingness filter + VIF check (formative dimensions D2, D5)
#   Phase 2: Criterion validity with:
#            - Threshold |r| >= 0.30 (minimum detectable at n=42, 80% power)
#            - Pearson AND Spearman r vs exam %
#            - 95% percentile bootstrap CIs (2,000 resamples)
#            - Benjamini-Hochberg FDR correction across ALL correlations
#            - Revised threshold: FDR p < 0.05 OR bootstrap CI excludes zero
#   Phase 3: Bootstrap stability selection:
#            - 1,000 LASSO resamples (not 100)
#            - Keep if selected in >= 75% of resamples (not 40%)
#            - Limit FINAL model to <= 4 predictors (10:1 OPR at n=42)
# ==================================================================

def _vif(X_df):
    """
    Compute Variance Inflation Factors for all columns in X_df.
    VIF_j = 1/(1-R2_j) where R2_j is R2 of regressing column j on all others.
    Returns dict {col: vif_value}.
    """
    from sklearn.linear_model import LinearRegression as _LR
    vifs = {}
    cols = list(X_df.columns)
    if len(cols) < 2:
        return {c: 1.0 for c in cols}
    for j, col in enumerate(cols):
        others = [c for c in cols if c != col]
        X_ = X_df[others].values
        y_ = X_df[col].values
        ok = ~np.isnan(X_).any(axis=1) & ~np.isnan(y_)
        if ok.sum() < 5:
            vifs[col] = np.nan
            continue
        r2 = _LR().fit(X_[ok], y_[ok]).score(X_[ok], y_[ok])
        vifs[col] = 1.0 / (1 - r2 + 1e-9)
    return vifs


def _bootstrap_ci(x, y, n_boot=2000, ci=0.95):
    """
    95% percentile bootstrap CI for Pearson r.
    Returns (r_obs, r_lo, r_hi, boot_p) where boot_p = proportion of
    bootstrap r values with opposite sign to r_obs (two-tailed).
    """
    r_obs, _ = stats.pearsonr(x, y)
    boot_rs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        try:
            r_b, _ = stats.pearsonr(x[idx], y[idx])
            boot_rs.append(r_b)
        except:
            pass
    boot_rs = np.array(boot_rs)
    lo = np.nanpercentile(boot_rs, (1-ci)/2*100)
    hi = np.nanpercentile(boot_rs, (1-(1-ci)/2)*100)
    # Two-tailed bootstrap p: proportion of bootstrap distribution on wrong side
    if r_obs >= 0:
        boot_p = float(np.mean(boot_rs < 0)) * 2
    else:
        boot_p = float(np.mean(boot_rs > 0)) * 2
    return r_obs, lo, hi, min(boot_p, 1.0)


def select_features(feat_df, miss_thresh=0.50, min_r=0.30, stab=0.75, max_pred=4):
    """
    Revised validation pipeline (per candidate feature review criteria).

    Phase 1  Missingness + VIF:
        Drop if >50% missing.
        For formative dimensions (D2: F01-F08, D5: F17-F25): flag VIF >= 5.0.
        VIF >= 5.0 means two features are near-collinear; consider dropping higher-VIF one.

    Phase 2  Criterion validity:
        Compute Pearson r AND Spearman rho vs exam%.
        Compute 95% percentile bootstrap CI (2,000 resamples).
        Apply Benjamini-Hochberg FDR correction across all features.
        Selection criterion: |r| >= 0.30 AND (FDR p < 0.05 OR CI excludes 0).

    Phase 3  Bootstrap stability selection:
        1,000 LASSO resamples on 80% subsamples.
        Keep if selected in >= 75% of resamples.
        Hard cap: at most max_pred=4 final features (10:1 OPR at n~42).

    NOTE: n=42 has zero power for |r| < 0.20, marginal power at r=0.30.
    All findings should be framed as PILOT/EXPLORATORY.

    Returns: selected (list), corr_df (table), X_imp (imputed DataFrame)
    """
    y = feat_df["pct"].values
    X = feat_df[F_COLS].copy()

    # sign-flip negative-direction features (higher = better EI)
    for col in NEG_DIR:
        if col in X.columns:
            X[col] = X[col] * -1

    # ---- Phase 1: Missingness filter ----
    X = X.loc[:, X.isnull().mean() <= miss_thresh]
    print(f"  Phase 1 - After missingness ({miss_thresh*100:.0f}% threshold): {X.shape[1]} features")

    # Median imputation before VIF
    X_imp = X.copy()
    for col in X_imp.columns:
        X_imp[col] = X_imp[col].fillna(X_imp[col].median())

    # VIF check for formative dimensions (D2 = F01-F08, D5 = F17-F25)
    formative_D2 = [c for c in X_imp.columns if c.startswith("F0") and
                    int(c[1:3]) <= 8 and int(c[1:3]) >= 1]
    formative_D5 = [c for c in X_imp.columns if c.startswith("F1") or
                    (c.startswith("F2") and int(c[1:3]) <= 25)]

    vif_cols = formative_D2 + formative_D5
    if len(vif_cols) >= 2:
        vif_vals = _vif(X_imp[vif_cols])
        high_vif = {k:v for k,v in vif_vals.items() if pd.notna(v) and v >= 5.0}
        print(f"\n  VIF check (formative dims D2+D5) -- features with VIF >= 5.0:")
        if high_vif:
            for feat, vif_v in sorted(high_vif.items(), key=lambda x:-x[1]):
                print(f"    {feat:<40s}  VIF={vif_v:.2f}  (near-collinear, review)")
        else:
            print("    All VIF < 5.0 -- no near-collinearity detected")

    # ---- Phase 2: Criterion validity with bootstrap CI + FDR ----
    ok_y = ~np.isnan(y)
    corr_rows = []
    print(f"\n  Phase 2 - Computing bootstrap correlations (2,000 resamples)...")

    for col in X_imp.columns:
        ok = ok_y & ~np.isnan(X_imp[col].values)
        n_ok = ok.sum()
        if n_ok < 8:
            corr_rows.append({
                "feature": col, "n": n_ok,
                "Pearson_r": np.nan, "Pearson_p": 1.0,
                "Spearman_r": np.nan, "Spearman_p": 1.0,
                "CI_lo": np.nan, "CI_hi": np.nan, "boot_p": 1.0,
                "CI_excludes_zero": False
            })
            continue
        xv = X_imp.loc[ok, col].values
        yv = y[ok]

        # Pearson r with bootstrap CI
        rp, ci_lo, ci_hi, boot_p = _bootstrap_ci(xv, yv)
        # Spearman rho
        rs, ps = stats.spearmanr(xv, yv)
        # Regular parametric p
        _, pp = stats.pearsonr(xv, yv)

        ci_excl_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)

        corr_rows.append({
            "feature": col, "n": n_ok,
            "Pearson_r": round(rp, 4), "Pearson_p": round(pp, 4),
            "Spearman_r": round(rs, 4), "Spearman_p": round(ps, 4),
            "CI_lo": round(ci_lo, 4), "CI_hi": round(ci_hi, 4),
            "boot_p": round(boot_p, 4),
            "CI_excludes_zero": ci_excl_zero
        })

    corr_df = pd.DataFrame(corr_rows)

    # Benjamini-Hochberg FDR correction across all features
    valid_mask = corr_df["Pearson_p"].notna() & (corr_df["Pearson_p"] < 1.0)
    if valid_mask.sum() > 1:
        _, p_fdr, _, _ = multipletests(corr_df.loc[valid_mask, "Pearson_p"], method="fdr_bh")
        corr_df.loc[valid_mask, "p_FDR"] = np.round(p_fdr, 4)
    else:
        corr_df["p_FDR"] = corr_df["Pearson_p"]

    corr_df = corr_df.sort_values("Pearson_r", key=abs, ascending=False).reset_index(drop=True)

    # Selection criterion: |r| >= min_r AND (FDR p < 0.05 OR CI excludes 0)
    def _meets_criterion(row):
        if pd.isna(row["Pearson_r"]):
            return False
        r_ok = abs(row["Pearson_r"]) >= min_r
        fdr_ok = pd.notna(row.get("p_FDR")) and row.get("p_FDR", 1.0) < 0.05
        ci_ok = row.get("CI_excludes_zero", False)
        return r_ok and (fdr_ok or ci_ok)

    corr_df["meets_criterion"] = corr_df.apply(_meets_criterion, axis=1)
    keep = corr_df[corr_df["meets_criterion"]]["feature"].tolist()

    # Fallback: if fewer than 2 pass strict criterion, take top features by |r| >= min_r
    if len(keep) < 2:
        fallback = corr_df[corr_df["Pearson_r"].abs() >= min_r]["feature"].tolist()
        if len(fallback) >= 2:
            keep = fallback
            print(f"  Phase 2 - Criterion relaxed to |r|>={min_r} only: {len(keep)} features")
        else:
            keep = corr_df.dropna(subset=["Pearson_r"]).head(4)["feature"].tolist()
            print(f"  Phase 2 - Criterion relaxed to top-4 by |r|: {len(keep)} features")
    else:
        print(f"  Phase 2 - After criterion (|r|>={min_r}, FDR p<0.05 or CI excl. 0): "
              f"{len(keep)} features")

    print(f"\n  Correlation summary for candidate features:")
    print(f"  {'Feature':<35} {'r':>7} {'rho':>7} {'CI_lo':>7} {'CI_hi':>7} "
          f"{'FDR_p':>7} {'CI_excl0':>9} {'Pass':>6}")
    print("  " + "-"*85)
    for _, row in corr_df.iterrows():
        if pd.isna(row["Pearson_r"]):
            continue
        passes = "*" if row["meets_criterion"] else ""
        print(f"  {row['feature']:<35} {row['Pearson_r']:>7.3f} {row['Spearman_r']:>7.3f} "
              f"{row['CI_lo']:>7.3f} {row['CI_hi']:>7.3f} "
              f"{row.get('p_FDR',np.nan):>7.4f} "
              f"{'YES' if row['CI_excludes_zero'] else 'no':>9} "
              f"{passes:>6}")

    # ---- Phase 3: Bootstrap stability selection ----
    print(f"\n  Phase 3 - Bootstrap stability selection (1,000 resamples, stab>={stab})...")
    if len(keep) == 0:
        print("  WARNING: No features passed criterion. Returning top-2 by |r|.")
        keep = corr_df.dropna(subset=["Pearson_r"]).head(2)["feature"].tolist()

    ok_idx = np.where(ok_y)[0]
    Xs = StandardScaler().fit_transform(X_imp.loc[ok_y, keep])
    ys = y[ok_y]
    cnt = np.zeros(len(keep))
    cvk = min(5, max(2, int(len(ys) * 0.8) - 1))

    for _ in range(1000):
        idx = resample(np.arange(len(ys)), n_samples=int(0.8 * len(ys)), replace=False)
        try:
            lasso = LassoCV(cv=cvk, max_iter=10000)
            lasso.fit(Xs[idx], ys[idx])
            cnt += (np.abs(lasso.coef_) > 1e-4).astype(int)
        except:
            pass

    rates = cnt / 1000
    selected = [keep[i] for i in range(len(keep)) if rates[i] >= stab]

    if len(selected) == 0:
        # Fallback: take top-2 by stability rate
        top2_idx = np.argsort(rates)[::-1][:2]
        selected = [keep[i] for i in top2_idx]
        print(f"  Phase 3 - Stability fallback: top-2 by rate ({[f'{rates[i]:.2f}' for i in top2_idx]})")
    else:
        print(f"  Phase 3 - After stability>={stab}: {len(selected)} features")

    # Hard cap: 4 predictors maximum (10:1 OPR with n~42)
    if len(selected) > max_pred:
        # rank by stability rate, take top max_pred
        sel_rates = {keep[i]: rates[i] for i in range(len(keep)) if keep[i] in selected}
        selected = sorted(selected, key=lambda x: sel_rates.get(x, 0), reverse=True)[:max_pred]
        print(f"  Phase 3 - Capped at {max_pred} predictors (10:1 OPR rule): {selected}")

    print(f"\n  FINAL SELECTION ({len(selected)} features):")
    for feat in selected:
        idx_f = keep.index(feat)
        row = corr_df[corr_df["feature"] == feat].iloc[0]
        print(f"    {feat:<40s}  r={row['Pearson_r']:>6.3f}  "
              f"CI=[{row['CI_lo']:.3f},{row['CI_hi']:.3f}]  "
              f"stab={rates[idx_f]:.2f}")

    # Store stability rates in corr_df
    rate_d = {keep[i]: round(rates[i], 3) for i in range(len(keep))}
    corr_df["stability"] = corr_df["feature"].map(rate_d)
    corr_df["selected"] = corr_df["feature"].isin(selected)

    return selected, corr_df, X_imp


# ==================================================================
# SECTION 7  --  EI SCORE  (PC1 of selected features)
# ==================================================================

def build_ei(feat_df, selected, X_imp):
    """
    EI score = PC1 of selected features, sign-aligned so higher = better EI.
    Also stores a simple equal-weight z-composite as a backup metric.

    NOTE: This version computes a full-sample PCA for descriptive purposes only.
    For predictive evaluation, use ei_loocv_fold() within cross-validation loops.
    """
    X_sel  = X_imp[selected].values
    ok     = ~feat_df["pct"].isna()
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_sel)
    pca    = PCA(n_components=min(len(selected),3))
    pca.fit(X_sc[ok])
    pcs    = pca.transform(X_sc)[:,0]

    # align sign with exam %
    if stats.pearsonr(pcs[ok], feat_df.loc[ok,"pct"])[0] < 0:
        pcs *= -1; pca.components_[0] *= -1

    feat_df = feat_df.copy()
    feat_df["EI_score"]     = pcs
    feat_df["EI_composite"] = X_sc.mean(axis=1)

    print(f"\n  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of selected-feature variance")
    print("  Loadings (full-sample PCA for reference):")
    for f,l in zip(selected, pca.components_[0]):
        print(f"    {f:<42s}  {l:+.3f}")
    return feat_df, pca, scaler


def ei_loocv_fold(X_train, X_test, y_train):
    """
    Compute EI score for test fold using PCA fit ONLY on training data.
    Prevents data leakage where held-out participant influences their own score.

    Returns: ei_test_pred (score for test fold), loadings (for tracking stability)
    """
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    pca = PCA(n_components=min(X_train.shape[1], 3))
    pca.fit(X_train_sc)
    ei_train = pca.transform(X_train_sc)[:,0]
    ei_test  = pca.transform(X_test_sc)[:,0]

    # align sign with training data labels
    if np.corrcoef(ei_train, y_train)[0,1] < 0:
        ei_test *= -1
        loadings = -pca.components_[0]
    else:
        loadings = pca.components_[0]

    return ei_test[0], loadings


# ==================================================================
# SECTION 8  --  MODELS  (LOOCV, n~42)
# ==================================================================

def run_models(feat_df, selected, X_imp):
    """
    Seven models evaluated via LOOCV with effect sizes and confidence intervals.

    Models:
      - Linear Regression (baseline)
      - Ridge Regression (regularized)
      - ElasticNet (L1+L2)
      - Bayesian Ridge (probabilistic)
      - Random Forest (ensemble)
      - XGBoost (boosting)
      - Mean Baseline (sanity check)

    Statistics:
      - R² with permutation p-value
      - Pearson r with 95% Fisher z confidence interval
      - Cohen's f² (effect size)
      - RMSE and Spearman correlation
    """
    ok  = feat_df["pct"].notna()
    X   = StandardScaler().fit_transform(X_imp.loc[ok,selected])
    y   = feat_df.loc[ok,"pct"].values
    n   = len(y)
    loo = LeaveOneOut()

    try:
        import xgboost as xgb
        xgb_available = True
    except ImportError:
        xgb_available = False

    MODELS = {
        "Linear Regression":   LinearRegression(),
        "Ridge Regression":    RidgeCV(alphas=np.logspace(-2,3,20), cv=5),
        "ElasticNet":          ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, max_iter=5000),
        "Bayesian Ridge":      BayesianRidge(),
        "Random Forest":       RandomForestRegressor(200, max_depth=3, min_samples_leaf=3, random_state=42),
    }

    if xgb_available:
        MODELS["XGBoost"] = xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42)

    # Add mean baseline
    from sklearn.dummy import DummyRegressor
    MODELS["Mean Baseline"] = DummyRegressor(strategy="mean")

    results = {}
    all_preds = {}

    for name, mdl in MODELS.items():
        try:
            yp  = cross_val_predict(mdl, X, y, cv=loo)
            r2  = r2_score(y, yp)
            rmse = float(np.sqrt(mean_squared_error(y, yp)))
            rsp, psp = stats.spearmanr(y, yp)
            rpe, ppe = stats.pearsonr(y, yp)

            # Cohen's f² effect size
            f2 = r2 / (1 - r2 + 1e-9)
            if f2 >= 0.35:
                effect = "large"
            elif f2 >= 0.15:
                effect = "medium"
            else:
                effect = "small"

            # Fisher z 95% CI on Pearson r
            if abs(rpe) < 0.9999:
                z = 0.5 * np.log((1 + rpe) / (1 - rpe + 1e-9))
                se = 1.0 / np.sqrt(n - 3)
                r_lo = np.tanh(z - 1.96 * se)
                r_hi = np.tanh(z + 1.96 * se)
            else:
                r_lo, r_hi = np.nan, np.nan

            # Permutation test (10,000 permutations — per validation criteria for n=42)
            perm_r2 = []
            for _ in range(10000):
                y_perm = np.random.permutation(y)
                yp_perm = cross_val_predict(mdl, X, y_perm, cv=loo)
                perm_r2.append(r2_score(y_perm, yp_perm))
            p_perm = float(np.mean(np.array(perm_r2) >= r2))

            results[name] = {
                "R2": r2, "RMSE": rmse,
                "Pearson_r": rpe, "Pearson_p": ppe,
                "Pearson_r_CI": (r_lo, r_hi),
                "Spearman_r": rsp, "Spearman_p": psp,
                "Cohen_f2": f2, "Effect_size": effect,
                "Permutation_p": p_perm,
                "y_true": y, "y_pred": yp,
                "n_samples": n
            }
            all_preds[name] = yp

            print(f"\n  {name}:")
            print(f"    R2={r2:.3f}  f²={f2:.3f} ({effect})  RMSE={rmse:.2f}")
            print(f"    Pearson r={rpe:.3f} (p={ppe:.4f})")
            print(f"      95% CI: [{r_lo:.3f}, {r_hi:.3f}]")
            print(f"    Spearman r={rsp:.3f} (p={psp:.4f})")
            print(f"    Permutation p={p_perm:.4f}")

        except Exception as e:
            print(f"\n  {name}: FAILED -- {e}")

    # Pairwise Wilcoxon tests between models (sign test on residuals)
    print("\n" + "="*70)
    print("PAIRWISE MODEL COMPARISONS (Wilcoxon signed-rank test)")
    print("="*70)
    model_names = list(results.keys())
    comparison_results = []

    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            res1 = np.abs(results[m1]["y_true"] - results[m1]["y_pred"])
            res2 = np.abs(results[m2]["y_true"] - results[m2]["y_pred"])
            w_stat, w_pval = stats.wilcoxon(res1, res2)
            comparison_results.append({
                "Model_1": m1, "Model_2": m2,
                "W_statistic": w_stat, "p_value": w_pval,
                "Better": m1 if (res1.mean() < res2.mean()) else m2,
                "Delta_MAE": abs(res1.mean() - res2.mean())
            })
            sig = "*" if w_pval < 0.05 else ""
            better = m1 if (res1.mean() < res2.mean()) else m2
            print(f"  {m1:<20s} vs {m2:<20s}: W={w_stat:7.0f}  p={w_pval:.4f} {sig}  (better: {better})")

    return results, comparison_results


# ==================================================================
# SECTION 9  --  RESEARCH QUESTION ANALYSES
# ==================================================================

def rq1_stages(feat_df):
    """RQ1: How do participants move through stages?"""
    print("\n" + "-"*60)
    print("RQ1 -- Stage Distribution and Transition Patterns")
    print("-"*60)
    # reload stage data for time proportions
    pct_rows, all_trans = [], {s:{s2:0 for s2 in STAGES} for s in STAGES}
    for pid in feat_df["pid"]:
        sdf = load_stage(pid)
        if sdf is None: continue
        tot = sdf["duration_s"].sum()
        if tot == 0: continue
        row = {"pid":pid}
        for s in STAGES:
            row[s] = sdf[sdf["stage"]==s]["duration_s"].sum()/tot*100
        pct_rows.append(row)
        seq = [s for s in sdf["stage"].tolist() if s in STAGES]
        for i in range(len(seq)-1):
            if seq[i] in STAGES and seq[i+1] in STAGES:
                all_trans[seq[i]][seq[i+1]] += 1

    spdf = pd.DataFrame(pct_rows)
    print("\n  Mean % time per stage:")
    for s in STAGES:
        if s in spdf.columns:
            print(f"    {s:<15s}: {spdf[s].mean():5.1f}% +/- {spdf[s].std():4.1f}%")
    tdf  = pd.DataFrame(all_trans).T
    tprob= tdf.div(tdf.sum(axis=1),axis=0).fillna(0)
    print("\n  Stage Transition Probability Matrix:")
    print(tprob.round(3).to_string())
    return spdf, tprob


def rq2_emotions(feat_df):
    """RQ2: How do emotions vary across stages?"""
    print("\n" + "-"*60)
    print("RQ2 -- Facial Emotion by Stage")
    print("-"*60)
    COMPS = ["Happiness","Stress","Anger","Fear","Sadness","Pride"]
    recs = []
    for pid in feat_df["pid"]:
        au_df    = load_au(pid)
        stage_df = load_stage(pid)
        if au_df is None or stage_df is None: continue
        au_df = add_composites(au_df)
        au_df = tag_stages(au_df, stage_df)
        for stage in STAGES:
            sf = au_df[au_df["stage"]==stage]
            if len(sf) < 10: continue
            rec = {"pid":pid,"stage":stage}
            for c in COMPS: rec[c] = float(sf[c].mean())
            recs.append(rec)
    long = pd.DataFrame(recs)
    print("\n  Mean composite by stage:")
    print(long.groupby("stage")[COMPS].mean().round(3).to_string())
    print("\n  Kruskal-Wallis (composite ~ stage):")
    for c in COMPS:
        grps = [long[long["stage"]==s][c].dropna().values for s in STAGES]
        grps = [g for g in grps if len(g)>=3]
        if len(grps)>=2:
            H, p = stats.kruskal(*grps)
            sig  = "*" if p<0.05 else ""
            print(f"    {c:<12s}: H={H:.2f}  p={p:.4f} {sig}")
    return long


def rq3_regulation(feat_df):
    """RQ3: Which features indicate stronger/weaker EI?"""
    print("\n" + "-"*60)
    print("RQ3 -- Feature Correlations with Exam Performance")
    print("-"*60)
    ok = feat_df["pct"].notna(); y = feat_df.loc[ok,"pct"].values
    rows = []
    for f in F_COLS:
        if f not in feat_df.columns: continue
        x = feat_df.loc[ok,f].values; v = ~(np.isnan(x)|np.isnan(y))
        if v.sum()<8:
            rows.append({"feature":f,"Spearman_r":np.nan,"p":1.0,"n":v.sum()}); continue
        r,p = stats.spearmanr(x[v],y[v])
        rows.append({"feature":f,"Spearman_r":round(r,3),"p":round(p,4),"n":v.sum()})
    res = pd.DataFrame(rows).sort_values("Spearman_r",key=abs,ascending=False)
    vp  = res["p"].notna()&(res["p"]<1)
    if vp.sum()>1:
        _,pfdr,_,_ = multipletests(res.loc[vp,"p"],method="fdr_bh")
        res.loc[vp,"p_fdr"] = pfdr.round(4)
    print(res.to_string(index=False))
    return res


def rq4_ei_exam(feat_df, selected, X_imp):
    """RQ4: EI score <-> Exam performance, with partial correlation control."""
    print("\n" + "-"*60)
    print("RQ4 -- EI Score vs Exam Performance")
    print("-"*60)
    df  = feat_df[["EI_score","pct"]].dropna()
    rp, pp = stats.pearsonr(df["EI_score"], df["pct"])
    rs, ps = stats.spearmanr(df["EI_score"], df["pct"])

    # Bivariate regression
    X_  = sm.add_constant(df["EI_score"])
    ols = sm.OLS(df["pct"], X_).fit()
    print(f"\n  BIVARIATE ANALYSIS:")
    print(f"  Pearson  r = {rp:.3f}  (p={pp:.4f})")
    print(f"  Spearman r = {rs:.3f}  (p={ps:.4f})")
    print(f"\n  OLS:  Exam% = {ols.params.iloc[0]:.2f} + {ols.params.iloc[1]:.2f} * EI_score")
    print(f"  R2={ols.rsquared:.3f}  F={ols.fvalue:.2f}  p={ols.f_pvalue:.4f}")
    print(ols.summary().tables[1])

    # Partial correlation: EI vs Exam%, controlling for strongest feature
    print(f"\n  PARTIAL CORRELATION ANALYSIS:")
    ok = feat_df["pct"].notna()
    y = feat_df.loc[ok, "pct"].values
    ei = df["EI_score"].values

    # Find strongest individual feature
    best_feat = None
    best_r = 0
    for f in selected:
        if f in X_imp.columns:
            x = X_imp.loc[ok, f].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 8:
                continue
            r_f, _ = stats.pearsonr(x[valid], y[valid])
            if abs(r_f) > abs(best_r):
                best_r = r_f
                best_feat = f

    if best_feat is not None:
        x_ctrl = X_imp.loc[ok, best_feat].values
        valid = ~(np.isnan(ei) | np.isnan(x_ctrl) | np.isnan(y))
        if valid.sum() >= 8:
            # Partial correlation: r_ey.x = (r_ey - r_ex*r_xy) / sqrt((1-r_ex²)(1-r_xy²))
            r_ey, _ = stats.pearsonr(ei[valid], y[valid])
            r_ex, _ = stats.pearsonr(ei[valid], x_ctrl[valid])
            r_xy, _ = stats.pearsonr(x_ctrl[valid], y[valid])
            denom = np.sqrt((1 - r_ex**2) * (1 - r_xy**2) + 1e-9)
            r_partial = (r_ey - r_ex * r_xy) / denom

            # T-test for partial r significance
            df_t = valid.sum() - 3
            t_stat = r_partial * np.sqrt(df_t / (1 - r_partial**2 + 1e-9))
            p_partial = 2 * (1 - stats.t.cdf(abs(t_stat), df_t))

            print(f"  Controlling for: {best_feat} (r={best_r:.3f})")
            print(f"  Partial r(EI, Exam | {best_feat}) = {r_partial:.3f}  (p={p_partial:.4f})")
            print(f"  EI explains {(r_partial**2)*100:.1f}% of Exam% variance beyond {best_feat}")

    return rp, pp, rs, ps, ols


def rq5_high_low(feat_df):
    """RQ5: High-EI vs Low-EI on adaptive behaviour indicators."""
    print("\n" + "-"*60)
    print("RQ5 -- High vs Low EI Participants")
    print("-"*60)
    fd = feat_df.copy()
    fd["EI_group"] = np.where(fd["EI_score"]>=fd["EI_score"].median(),"High","Low")

    check = [
        ("F12_productive_time_ratio",    "Productive time ratio"),
        ("F09_block_to_production_rate", "Block->Prod rate"),
        ("F01_stress_recovery_latency",  "Stress recovery (s)   [lower=better]"),
        ("F04_stress_habituation_ratio", "Stress habituation    [lower=better]"),
        ("F11_productive_absorbing_prob","Productive Markov pi"),
        ("F14_late_engagement_ratio",    "Late/early engagement"),
        ("pct",                          "Exam score (%)"),
    ]
    check = [(c,l) for c,l in check if c in fd.columns]

    print(f"\n  {'Measure':<40s}  {'Hi-EI':>8} {'Lo-EI':>8} {'p(MW)':>8} {'d':>7}")
    print("  " + "-"*70)
    for col,lab in check:
        hi = fd[fd["EI_group"]=="High"][col].dropna()
        lo = fd[fd["EI_group"]=="Low"][col].dropna()
        if len(hi)<3 or len(lo)<3: continue
        _,p = stats.mannwhitneyu(hi,lo,alternative="two-sided")
        psd = np.sqrt((hi.std()**2+lo.std()**2)/2) or 1e-6
        d   = (hi.mean()-lo.mean())/psd
        sig = "*" if p<0.05 else ("~" if p<0.10 else " ")
        print(f"  {lab:<40s}  {hi.mean():>8.3f} {lo.mean():>8.3f} {p:>8.4f}{sig} {d:>7.3f}")
    return fd


# ==================================================================
# SECTION 10  --  FIGURES  (6 plots)
# ==================================================================

def make_figures(feat_df, corr_df, results, spdf, rq2_df, selected):
    plt.rcParams.update({"figure.dpi":130,"font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})

    # Fig 1 -- Feature correlation bar chart
    fig, ax = plt.subplots(figsize=(10,8))
    cd = corr_df.dropna(subset=["r"]).copy()
    clr = ["#27ae60" if s else "#bdc3c7" for s in cd["selected"]]
    ax.barh(cd["feature"], cd["r"], color=clr, edgecolor="none")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline( 0.3, color="#27ae60", lw=1, ls="--", alpha=0.6)
    ax.axvline(-0.3, color="#27ae60", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Pearson r with Exam %")
    ax.set_title("25-Feature Correlations with Exam Performance  (green = selected)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#27ae60",label="Selected"),
                       Patch(color="#bdc3c7",label="Not selected")])
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig1_correlations.png"); plt.close()

    # Fig 2 -- LOOCV scatter (3 models)
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax,(name,res) in zip(axes,results.items()):
        ax.scatter(res["y_true"],res["y_pred"],alpha=0.75,s=55,
                   color="steelblue",edgecolors="white",linewidths=0.5)
        lo=min(res["y_true"].min(),res["y_pred"].min())-2
        hi=max(res["y_true"].max(),res["y_pred"].max())+2
        ax.plot([lo,hi],[lo,hi],"r--",lw=1.5)
        ax.set_xlabel("Actual Exam %"); ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nR2={res['R2']:.3f}  rho={res['Spearman_r']:.3f}"
                     f"  p_perm={res['Permutation_p']:.3f}")
    fig.suptitle("LOOCV Predictions vs Actual Exam %", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig2_loocv_predictions.png"); plt.close()

    # Fig 3 -- EI score distribution + scatter
    fig,(a1,a2) = plt.subplots(1,2,figsize=(11,4))
    a1.hist(feat_df["EI_score"].dropna(),bins=14,color="steelblue",edgecolor="white")
    a1.set_xlabel("EI Score (PC1)"); a1.set_title("EI Score Distribution")
    d2 = feat_df[["EI_score","pct"]].dropna()
    a2.scatter(d2["EI_score"],d2["pct"],s=60,alpha=0.75,
               color="steelblue",edgecolors="white",linewidths=0.5)
    m,b = np.polyfit(d2["EI_score"],d2["pct"],1)
    xs  = np.linspace(d2["EI_score"].min(),d2["EI_score"].max(),80)
    a2.plot(xs,m*xs+b,"r-",lw=2)
    r,p = stats.pearsonr(d2["EI_score"],d2["pct"])
    a2.set_xlabel("EI Score"); a2.set_ylabel("Exam %")
    a2.set_title(f"EI Score vs Exam %\nr={r:.3f}  p={p:.3f}")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig3_ei_vs_exam.png"); plt.close()

    # Fig 4 -- Stage time distribution (RQ1)
    if spdf is not None and len(spdf)>0:
        fig,ax = plt.subplots(figsize=(9,5))
        cols_p = [s for s in STAGES if s in spdf.columns]
        means  = spdf[cols_p].mean(); sems = spdf[cols_p].sem()
        pal    = ["#3498db","#2ecc71","#e74c3c","#f39c12","#9b59b6","#e67e22"]
        ax.bar(cols_p,means,yerr=sems,capsize=5,
               color=pal[:len(cols_p)],edgecolor="white")
        ax.set_ylabel("% of Task Time")
        ax.set_title("RQ1 -- Mean % Time per Stage (+/- SEM)")
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig4_rq1_stages.png"); plt.close()

    # Fig 5 -- Emotion by stage heatmap (RQ2)
    if rq2_df is not None and len(rq2_df)>0:
        COMPS = ["Happiness","Stress","Anger","Fear","Sadness","Pride"]
        pivot = rq2_df.groupby("stage")[COMPS].mean()
        pivot = pivot.reindex([s for s in STAGES if s in pivot.index])
        zpiv  = (pivot-pivot.mean())/(pivot.std()+1e-6)
        fig,ax = plt.subplots(figsize=(10,5))
        sns.heatmap(zpiv,annot=True,fmt=".2f",cmap="RdYlGn",
                    center=0,ax=ax,linewidths=0.5,linecolor="white")
        ax.set_title("RQ2 -- Z-scored Emotion Composites by Stage")
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig5_rq2_emotions.png"); plt.close()

    # Fig 6 -- High vs Low EI boxplots (RQ5)
    if "EI_group" in feat_df.columns:
        vp = ["F12_productive_time_ratio","F01_stress_recovery_latency",
              "F09_block_to_production_rate","pct"]
        vp = [v for v in vp if v in feat_df.columns]
        fig,axes = plt.subplots(1,len(vp),figsize=(4*len(vp),5))
        if len(vp)==1: axes=[axes]
        for ax,var in zip(axes,vp):
            hi = feat_df[feat_df["EI_group"]=="High"][var].dropna()
            lo = feat_df[feat_df["EI_group"]=="Low"][var].dropna()
            ax.boxplot([hi,lo],labels=["High EI","Low EI"],patch_artist=True,
                       boxprops=dict(facecolor="#aed6f1"),
                       medianprops=dict(color="steelblue",lw=2))
            ax.set_title(var.replace("_"," ")[:28],fontsize=9)
        fig.suptitle("RQ5 -- High vs Low EI",fontsize=12)
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig6_rq5_hilo_ei.png"); plt.close()

    print(f"  Figures saved --> {OUT_DIR}/")


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":

    SEP = "=" * 62

    print(SEP)
    print("PILOT STUDY -- EI Pipeline (25 Features, n=42)")
    print("NOTE: n=42 is below minimum for formal psychometric validation.")
    print("All findings are exploratory. Replication with n>=150 required.")
    print("Min detectable effect at 80% power: |r|=0.42 (medium-large).")
    print(SEP)

    print(f"\n{SEP}")
    print("STEP 1 -- Feature extraction  (25 features x 42 participants)")
    print(SEP)
    feat_df = build_matrix()

    print(f"\n{SEP}")
    print("STEP 2 -- Revised validation pipeline (per candidate feature review)")
    print("         Phase 1: Missingness + VIF  |  Phase 2: Bootstrap r + FDR")
    print("         Phase 3: Stability selection (1000 resamples, >=75%, max 4)")
    print(SEP)
    selected, corr_df, X_imp = select_features(feat_df)
    print(f"\n  Selected ({len(selected)} features, max 4 per 10:1 OPR rule): {selected}")
    corr_df.to_csv(f"{OUT_DIR}/feature_selection.csv", index=False)

    print(f"\n{SEP}")
    print("STEP 3 -- EI score construction  (PCA -> PC1)")
    print(SEP)
    feat_df, pca_mdl, scaler_mdl = build_ei(feat_df, selected, X_imp)

    print(f"\n{SEP}")
    print("STEP 4 -- Model training and LOOCV evaluation")
    print(SEP)
    model_results, comparison_results = run_models(feat_df, selected, X_imp)

    print(f"\n{SEP}")
    print("STEP 5 -- Research question analyses")
    print(SEP)
    spdf, trans   = rq1_stages(feat_df)
    rq2_df        = rq2_emotions(feat_df)
    rq3_df        = rq3_regulation(feat_df)
    rp,pp,rs,ps,ols = rq4_ei_exam(feat_df, selected, X_imp)
    feat_df       = rq5_high_low(feat_df)

    # ---- save all outputs ----
    feat_df.to_csv(f"{OUT_DIR}/results_with_ei.csv", index=False)
    rq3_df.to_csv( f"{OUT_DIR}/rq3_spearman.csv",   index=False)
    trans.to_csv(  f"{OUT_DIR}/rq1_transitions.csv")
    rq2_df.to_csv( f"{OUT_DIR}/rq2_emotions.csv",   index=False)

    # ---- model summary ----
    print(f"\n{SEP}")
    print("MODEL SUMMARY WITH EFFECT SIZES")
    print(SEP)
    smrows = []
    for name, res in model_results.items():
        r_ci = res.get("Pearson_r_CI", (np.nan, np.nan))
        smrows.append({
            "Model": name,
            "R2_LOOCV": round(res["R2"], 3),
            "RMSE": round(res["RMSE"], 2),
            "Pearson_r": round(res["Pearson_r"], 3),
            "Pearson_r_CI_lower": round(r_ci[0], 3),
            "Pearson_r_CI_upper": round(r_ci[1], 3),
            "Spearman_r": round(res["Spearman_r"], 3),
            "Spearman_p": round(res["Spearman_p"], 4),
            "Cohen_f2": round(res.get("Cohen_f2", np.nan), 3),
            "Effect_size": res.get("Effect_size", ""),
            "Permutation_p": round(res["Permutation_p"], 4)
        })
    sumdf = pd.DataFrame(smrows)
    print(sumdf.to_string(index=False))
    sumdf.to_csv(f"{OUT_DIR}/model_summary.csv", index=False)

    # Save model comparison results
    if comparison_results:
        comp_df = pd.DataFrame(comparison_results)
        comp_df.to_csv(f"{OUT_DIR}/model_comparisons_wilcoxon.csv", index=False)
        print(f"\n  Model comparison p-values saved")


    print(f"\n{SEP}")
    print("STEP 6 -- Figures")
    print(SEP)
    make_figures(feat_df, corr_df, model_results, spdf, rq2_df, selected)

    print(f"\n{'+'*3} Pipeline complete. Outputs saved to: {OUT_DIR}/")
    print("  results_with_ei.csv   -- EI score + all features per participant")
    print("  feature_selection.csv -- r, stability, selected flag per feature")
    print("  model_summary.csv     -- R2, RMSE, Spearman r, permutation p")
    print("  rq1_transitions.csv   -- stage transition probability matrix")
    print("  rq2_emotions.csv      -- emotion composite means per stage")
    print("  rq3_spearman.csv      -- Spearman r per feature vs exam %")
    print("  fig1..fig6 (PNG)      -- all visualisations")
