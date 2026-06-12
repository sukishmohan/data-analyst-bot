"""
data_loader.py — Data loading, profiling, and automatic cleaning.

Responsibilities:
  - Load the Superstore CSV with robust encoding handling
  - Profile the dataset (types, nulls, stats, duplicates)
  - Auto-clean: parse dates, fill/drop nulls, remove duplicates
  - Produce a structured Dataset Report for downstream agents
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from tabulate import tabulate

from utils import get_logger, safe_print

log = get_logger("data_loader")


# ──────────────────────────────────────────────
# 1. Loading
# ──────────────────────────────────────────────

def load_csv(path: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load a CSV file with fallback encodings.
    Tries utf-8 -> latin-1 -> cp1252 in sequence.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    encodings = [encoding, "latin-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            log.info("Loaded %s rows x %s cols from %s (encoding=%s)",
                     len(df), len(df.columns), path.name, enc)
            return df
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"Failed to read {path} with encodings {encodings}")


# ──────────────────────────────────────────────
# 2. Data Profiling / Understanding
# ──────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Produce a structured profile of the DataFrame.

    Returns a dict with keys:
      shape, columns, dtypes, missing, duplicates,
      numeric_stats, sample_values, date_columns, categorical_columns
    """
    profile: dict = {}

    profile["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    profile["columns"] = list(df.columns)
    profile["dtypes"] = {col: str(dt) for col, dt in df.dtypes.items()}

    # Missing values
    missing = df.isnull().sum()
    profile["missing"] = {
        col: int(cnt) for col, cnt in missing.items() if cnt > 0
    }
    profile["total_missing"] = int(missing.sum())

    # Duplicates
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # Numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        stats = df[num_cols].describe().round(2)
        profile["numeric_stats"] = stats.to_dict()

    # Detect date-like columns
    date_cols = []
    for col in df.columns:
        if "date" in col.lower():
            date_cols.append(col)
            continue
        if df[col].dtype == "object":
            sample = df[col].dropna().head(20)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                date_cols.append(col)
            except (ValueError, TypeError):
                pass
    profile["date_columns"] = date_cols

    # Categorical columns
    cat_cols = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col not in date_cols:
            cat_cols.append(col)
    profile["categorical_columns"] = cat_cols

    # Sample unique values for categoricals
    profile["sample_values"] = {}
    for col in cat_cols:
        uniques = df[col].dropna().unique()
        profile["sample_values"][col] = list(uniques[:15])
    profile["numeric_columns"] = num_cols

    return profile


def print_dataset_report(profile: dict) -> str:
    """Format the profile dict into a human-readable report and print it."""
    lines = []
    lines.append("=" * 60)
    lines.append("  DATASET REPORT")
    lines.append("=" * 60)

    lines.append(f"\nShape: {profile['shape']['rows']:,} rows x "
                 f"{profile['shape']['columns']} columns\n")

    # Column info table
    col_table = []
    for col in profile["columns"]:
        dtype = profile["dtypes"][col]
        miss = profile.get("missing", {}).get(col, 0)
        col_table.append([col, dtype, miss])
    lines.append(tabulate(col_table,
                          headers=["Column", "Type", "Missing"],
                          tablefmt="grid"))

    lines.append(f"\nDuplicate rows: {profile['duplicate_rows']}")
    lines.append(f"Date columns detected: {profile['date_columns']}")
    lines.append(f"Categorical columns: {profile['categorical_columns']}")
    lines.append(f"Numeric columns: {profile.get('numeric_columns', [])}")

    # Numeric stats
    if "numeric_stats" in profile:
        lines.append("\nNumeric Summary:")
        stats_df = pd.DataFrame(profile["numeric_stats"]).T
        lines.append(tabulate(stats_df, headers="keys",
                              tablefmt="grid", floatfmt=".2f"))

    # Sample values
    if profile.get("sample_values"):
        lines.append("\nSample categorical values:")
        for col, vals in profile["sample_values"].items():
            lines.append(f"  - {col}: {vals[:8]}")

    report = "\n".join(lines)
    safe_print(report)
    return report


# ──────────────────────────────────────────────
# 3. Auto-Cleaning
# ──────────────────────────────────────────────

def auto_clean(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """
    Apply automatic cleaning steps:
      1. Parse detected date columns to datetime
      2. Remove exact duplicate rows
      3. Fill missing numerics with median, categoricals with mode
      4. Strip whitespace from string columns
    Returns a cleaned copy.
    """
    df = df.copy()
    changes: list[str] = []

    # 1. Parse dates
    for col in profile.get("date_columns", []):
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                changes.append(f"  [OK] Parsed '{col}' as datetime")
            except Exception:
                pass

    # 2. Remove duplicates
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        changes.append(f"  [OK] Removed {n_dup} duplicate rows")

    # 3. Fill missing values
    for col in df.columns:
        n_miss = df[col].isnull().sum()
        if n_miss == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            changes.append(f"  [OK] Filled {n_miss} nulls in '{col}' with median ({median_val:.2f})")
        else:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val.iloc[0], inplace=True)
                changes.append(f"  [OK] Filled {n_miss} nulls in '{col}' with mode ('{mode_val.iloc[0]}')")

    # 4. Strip whitespace on string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    if changes:
        log.info("Auto-clean applied:\n%s", "\n".join(changes))
        safe_print("\nAuto-Cleaning Applied:")
        safe_print("\n".join(changes))
    else:
        log.info("No cleaning needed.")
        safe_print("\nDataset is already clean -- no changes needed.")

    return df


# ──────────────────────────────────────────────
# 4. Convenience wrapper
# ──────────────────────────────────────────────

def load_and_prepare(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: load -> profile -> report -> clean."""
    df = load_csv(path)
    profile = profile_dataset(df)
    print_dataset_report(profile)
    df = auto_clean(df, profile)
    # Re-profile after cleaning
    profile = profile_dataset(df)
    return df, profile
