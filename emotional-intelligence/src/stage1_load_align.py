"""
Stage 1 — Data Loading & Alignment
===================================
Loads facial AU data, stage-coding data, and SAM ratings for all 42 participants.
Outputs:
  outputs/aligned_data.pkl  — dict {pid: {face_df, stage_df, sam_valence, sam_arousal}}
"""

import os, re, datetime, warnings
import pandas as pd
import numpy as np
import pickle
warnings.filterwarnings("ignore")

from config import (
    FACIAL_DIR, STAGE_DIR, SAM_FILE, OUTPUT_DIR,
    AU_COLS, EMOTION_DEFS, EMOTION_COLS, STAGE_NORM, SAM_STAGE_PREFIX
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def _pid(filename: str) -> int:
    """Extract participant number from filename like 'P3 - name.xlsx'."""
    m = re.match(r"P(\d+)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def _time_to_sec(val) -> float:
    """Convert datetime.time / timedelta / string HH:MM:SS to float seconds."""
    if isinstance(val, datetime.time):
        return val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6
    if isinstance(val, (pd.Timedelta, datetime.timedelta)):
        return val.total_seconds()
    if isinstance(val, str):
        parts = val.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return np.nan


def _norm_stage(label) -> str:
    """Normalise stage labels to canonical names; SAM rows → 'SAM'."""
    if pd.isna(label):
        return "Unknown"
    s = str(label).strip()
    lower = s.lower()
    if lower.startswith(SAM_STAGE_PREFIX):
        return "SAM"
    return STAGE_NORM.get(lower, s)   # keep as-is if not in map


def compute_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7 emotion intensity columns to a facial AU dataframe."""
    emo = {}
    for name, aus in EMOTION_DEFS.items():
        present = [a for a in aus if a in df.columns]
        emo[name] = df[present].sum(axis=1) if present else 0.0
    return pd.DataFrame(emo, index=df.index)


# ── per-participant loaders ──────────────────────────────────────────────────

def load_facial(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df.columns = [c.strip() for c in df.columns]
    # Keep only AU columns + time column
    time_col = "cleaned time"
    if time_col not in df.columns:
        # fallback: first numeric column
        time_col = df.columns[0]
    keep = [time_col] + [c for c in AU_COLS if c in df.columns]
    df = df[keep].copy()
    df.rename(columns={time_col: "t_sec"}, inplace=True)
    df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")
    df.dropna(subset=["t_sec"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Fill missing AUs with 0
    for col in AU_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # Append emotion composites
    df = pd.concat([df, compute_emotions(df)], axis=1)
    return df


def load_stages(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df.columns = [c.strip() for c in df.columns]
    df["stage"]     = df["Codes"].apply(_norm_stage)
    df["start_sec"] = df["Timestamp (Start)"].apply(_time_to_sec)
    df["end_sec"]   = df["Timestamp (End)"].apply(_time_to_sec)
    # Drop SAM rows and unknown / NaN rows for stage analysis
    df = df[~df["stage"].isin(["SAM", "Unknown"])].copy()
    df.dropna(subset=["start_sec", "end_sec"], inplace=True)
    df = df[df["end_sec"] > df["start_sec"]].copy()
    df.reset_index(drop=True, inplace=True)
    return df[["start_sec", "end_sec", "stage"]].copy()


def assign_stages(face_df: pd.DataFrame, stage_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised assignment of stage label to each facial frame."""
    labels = np.full(len(face_df), "Unknown", dtype=object)
    for _, row in stage_df.iterrows():
        mask = (face_df["t_sec"] >= row["start_sec"]) & (face_df["t_sec"] < row["end_sec"])
        labels[mask.values] = row["stage"]
    face_df = face_df.copy()
    face_df["stage"] = labels
    return face_df


def load_sam(sam_df: pd.DataFrame, pid: int):
    """
    Extract valence and arousal arrays for one participant from the wide SAM table.
    sam_df  — the already-loaded SAM spreadsheet (row 0 = header descriptions).
    Returns (valence_arr, arousal_arr) as numpy float arrays; NaN where missing.
    """
    # Row index: pid-1 in the data rows (row 0 of sam_df is the header)
    # After dropping the header row, participant p<pid> is at integer position pid-1
    data_rows = sam_df.iloc[1:].reset_index(drop=True)   # rows 1..42
    idx = pid - 1
    if idx >= len(data_rows):
        return np.full(10, np.nan), np.full(10, np.nan)

    row = data_rows.iloc[idx]
    # The first column is participant label; then alternating valence/arousal for checkpoints 1-10
    vals, arousals = [], []
    # Columns: [pid_label, V1, A1, V2, A2, ..., V10, A10]  — 21 columns total
    for i in range(10):
        v_idx = 1 + i * 2       # 1,3,5,...,19
        a_idx = 1 + i * 2 + 1  # 2,4,6,...,20
        v = row.iloc[v_idx] if v_idx < len(row) else np.nan
        a = row.iloc[a_idx] if a_idx < len(row) else np.nan
        vals.append(float(v) if str(v) not in ("nan", "Missing", "NaN") else np.nan)
        arousals.append(float(a) if str(a) not in ("nan", "Missing", "NaN") else np.nan)
    return np.array(vals), np.array(arousals)


# ── file discovery ───────────────────────────────────────────────────────────

def build_file_map(directory: str) -> dict:
    """Return {pid_int: filepath} for all participant xlsx files in directory."""
    mapping = {}
    for fname in os.listdir(directory):
        if not fname.lower().endswith(".xlsx"):
            continue
        pid = _pid(fname)
        if pid > 0:
            mapping[pid] = os.path.join(directory, fname)
    return mapping


# ── main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("STAGE 1 — DATA LOADING & ALIGNMENT")
    print("=" * 60)

    facial_map = build_file_map(FACIAL_DIR)
    stage_map  = build_file_map(STAGE_DIR)
    sam_raw    = pd.read_excel(SAM_FILE)

    common_pids = sorted(set(facial_map) & set(stage_map))
    print(f"Participants with both facial & stage data: {len(common_pids)}")
    missing = (set(range(1, 43)) - set(common_pids))
    if missing:
        print(f"  Missing participant IDs: {sorted(missing)}")

    aligned = {}
    errors  = []

    for pid in common_pids:
        try:
            face_df  = load_facial(facial_map[pid])
            stage_df = load_stages(stage_map[pid])
            face_df  = assign_stages(face_df, stage_df)
            val, aro = load_sam(sam_raw, pid)

            # Impute missing SAM values with participant mean
            val_mean = np.nanmean(val) if not np.all(np.isnan(val)) else 3.0
            aro_mean = np.nanmean(aro) if not np.all(np.isnan(aro)) else 3.0
            val  = np.where(np.isnan(val), val_mean, val)
            aro  = np.where(np.isnan(aro), aro_mean, aro_mean)

            assigned_pct = (face_df["stage"] != "Unknown").mean() * 100

            aligned[pid] = {
                "face_df":     face_df,
                "stage_df":    stage_df,
                "sam_valence": val,
                "sam_arousal": aro,
            }
            print(f"  P{pid:02d}: {len(face_df):6d} frames | "
                  f"{len(stage_df):3d} stage rows | "
                  f"{assigned_pct:.1f}% frames assigned")
        except Exception as e:
            errors.append((pid, str(e)))
            print(f"  P{pid:02d}: ERROR — {e}")

    print(f"\nLoaded {len(aligned)} participants successfully, {len(errors)} errors.")
    if errors:
        for pid, msg in errors:
            print(f"  P{pid}: {msg}")

    out_path = os.path.join(OUTPUT_DIR, "aligned_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(aligned, f)
    print(f"\nSaved → {out_path}")
    return aligned


if __name__ == "__main__":
    run()
