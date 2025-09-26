#!/usr/bin/env python3
"""
outlier_scan.py
----------------
Load a Cloudwalk-style transactional CSV and run amount outlier detection.

Usage:
  python outlier_scan.py --csv transactional-sample.csv --method iqr --alpha 1.5

Methods:
  - iqr  : Tukey's IQR rule (default). Robust to non-normal distributions.
  - mad  : Median Absolute Deviation (Leys et al.), threshold=3.5 by default.
  - z    : Standard z-score (less robust).

Outputs:
  - Prints summary to stdout
  - Writes flagged rows to outliers_<method>.csv in the current directory
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------- Helpers -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run outlier detection on transaction_amount.")
    p.add_argument("--csv", required=True, help="Path to the transactional CSV file.")
    p.add_argument("--method", choices=["iqr", "mad", "z"], default="iqr",
                   help="Outlier detection method.")
    p.add_argument("--alpha", type=float, default=1.5,
                   help="Multiplier: IQR whisker length (iqr) or z-score threshold (z). Ignored for MAD unless --mad_thresh is provided.")
    p.add_argument("--mad_thresh", type=float, default=3.5,
                   help="Threshold for MAD-based modified z-score (default 3.5).")
    p.add_argument("--date_col", default="transaction_date", help="Name of the datetime column.")
    p.add_argument("--amount_col", default="transaction_amount", help="Name of the amount column.")
    p.add_argument("--id_col", default="transaction_id", help="Primary key column for uniqueness check.")
    return p.parse_args()

def load_data(path: str, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse datetime (coerce invalid -> NaT so we can quantify data issues)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df

def summarize(df: pd.DataFrame, id_col: str, date_col: str, amount_col: str) -> None:
    print("==== Basic Summary ====")
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print("\nNull counts:")
    print(df.isna().sum().sort_values(ascending=False))
    if id_col in df.columns:
        dup = df[id_col].duplicated().sum()
        print(f"\nDuplicate {id_col}: {dup}")
    if date_col in df.columns:
        dt_valid = df[date_col].notna().mean()*100
        if dt_valid > 0:
            print(f"{date_col}: {dt_valid:.2f}% parseable (NaT indicates invalid or missing)")
            print(f"Date range: {df[date_col].min()}  →  {df[date_col].max()}")
    if amount_col in df.columns:
        des = df[amount_col].describe(percentiles=[.01,.05,.5,.95,.99])
        print("\nAmount distribution:")
        print(des)

def outliers_iqr(x: pd.Series, alpha: float = 1.5) -> pd.Series:
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - alpha * iqr
    upper = q3 + alpha * iqr
    return (x < lower) | (x > upper), (lower, upper)

def outliers_z(x: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=x.index), (mu - z_thresh*sd, mu + z_thresh*sd)
    z = (x - mu) / sd
    return (z.abs() > z_thresh), (mu - z_thresh*sd, mu + z_thresh*sd)

def outliers_mad(x: pd.Series, thresh: float = 3.5) -> pd.Series:
    # Modified z-score using MAD (median absolute deviation)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=x.index), (None, None)
    mod_z = 0.6745 * (x - med) / mad
    return (np.abs(mod_z) > thresh), (None, None)

def run_outlier_test(df: pd.DataFrame, amount_col: str, method: str, alpha: float, mad_thresh: float):
    x = pd.to_numeric(df[amount_col], errors="coerce")
    valid_mask = x.notna()
    x_valid = x[valid_mask]

    if method == "iqr":
        mask, bounds = outliers_iqr(x_valid, alpha=alpha)
        lower, upper = bounds
        label = f"IQR (alpha={alpha})"
    elif method == "z":
        mask, bounds = outliers_z(x_valid, z_thresh=alpha)
        lower, upper = bounds
        label = f"Z-score (|z|>{alpha})"
    else:
        mask, bounds = outliers_mad(x_valid, thresh=mad_thresh)
        lower, upper = bounds
        label = f"MAD (>|{mad_thresh}| modified z-score)"

    flagged_idx = x_valid[mask].index
    flagged = df.loc[flagged_idx].copy()

    print("\n==== Outlier Detection ====")
    print(f"Method: {label}")
    if lower is not None and upper is not None:
        print(f"Bounds: [{lower:.4f}, {upper:.4f}]")
    print(f"Valid rows analyzed: {len(x_valid):,}")
    print(f"Outliers flagged: {len(flagged):,} ({(len(flagged)/max(1,len(x_valid)))*100:.2f}%)")

    if len(flagged) > 0:
        print("\nTop 10 outliers by amount:")
        print(flagged.sort_values(amount_col, ascending=False).head(10)[[amount_col]].to_string())

    return flagged

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_data(str(csv_path), date_col=args.date_col)

    # Basic summary
    summarize(df, id_col=args.id_col, date_col=args.date_col, amount_col=args.amount_col)

    # Run outlier test
    if args.amount_col not in df.columns:
        raise KeyError(f"Amount column '{args.amount_col}' not found in CSV columns: {list(df.columns)}")

    flagged = run_outlier_test(df, amount_col=args.amount_col, method=args.method,
                               alpha=args.alpha, mad_thresh=args.mad_thresh)

    # Write results
    out_path = Path(f"outliers_{args.method}.csv")
    flagged.to_csv(out_path, index=False)
    print(f"\nSaved flagged rows to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
