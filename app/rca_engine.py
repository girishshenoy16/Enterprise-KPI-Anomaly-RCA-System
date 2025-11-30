import pandas as pd
import numpy as np
from typing import List
from app.config import RAW_DATA_FILE, SEGMENT_COLUMNS


def load_raw_events(path=None) -> pd.DataFrame:
    path = path or RAW_DATA_FILE
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def compute_segment_contributions(
    raw_df: pd.DataFrame,
    target_date: pd.Timestamp,
    kpi: str,
    segment_cols: List[str] | None = None,
    lookback_days: int = 7,
) -> pd.DataFrame:
    if segment_cols is None:
        segment_cols = SEGMENT_COLUMNS

    raw_df = raw_df.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    target_date = pd.to_datetime(target_date)

    current = raw_df[raw_df["date"] == target_date]

    if current.empty:
        return pd.DataFrame(
            columns=segment_cols
            + [
                "current_value",
                "baseline_value",
                "abs_change",
                "pct_change",
                "contribution_pct_of_change",
            ]
        )
    

    baseline_mask = (raw_df["date"] < target_date) & (
        raw_df["date"] >= target_date - pd.Timedelta(days=lookback_days)
    )

    baseline = raw_df[baseline_mask]


    if baseline.empty:
        baseline = raw_df[raw_df["date"] < target_date]


    if baseline.empty:
        return pd.DataFrame(
            columns=segment_cols
            + [
                "current_value",
                "baseline_value",
                "abs_change",
                "pct_change",
                "contribution_pct_of_change",
            ]
        )


    curr_seg = (
        current.groupby(segment_cols)
        .agg({kpi: "sum"})
        .rename(columns={kpi: "current_value"})
        .reset_index()
    )


    base_seg = (
        baseline.groupby(segment_cols)
        .agg({kpi: "mean"})
        .rename(columns={kpi: "baseline_value"})
        .reset_index()
    )


    merged = curr_seg.merge(base_seg, on=segment_cols, how="left")
    merged["baseline_value"] = merged["baseline_value"].fillna(0)

    merged["abs_change"] = merged["current_value"] - merged["baseline_value"]
    merged["pct_change"] = np.where(
        merged["baseline_value"] != 0,
        merged["abs_change"] / merged["baseline_value"] * 100,
        np.nan,
    )

    total_change = merged["abs_change"].sum() or 1.0
    merged["contribution_pct_of_change"] = (
        merged["abs_change"] / total_change * 100
    )

    merged = merged.sort_values("contribution_pct_of_change", ascending=False)

    return merged


def summarize_root_cause(
    segment_df: pd.DataFrame, kpi: str, top_n: int = 3
) -> str:
    if segment_df.empty:
        return "No significant segment-level contributors identified (insufficient history)."

    top = segment_df.head(top_n)
    lines = []

    for _, row in top.iterrows():
        seg_desc = ", ".join(
            f"{col}={row[col]}" for col in segment_df.columns if col in SEGMENT_COLUMNS
        )

        sign = "increase" if row["abs_change"] > 0 else "decrease"

        lines.append(
            f"- {seg_desc}: {sign} of {abs(row['abs_change']):.2f} in {kpi} "
            f"({row['pct_change']:.1f}% vs baseline, "
            f"{row['contribution_pct_of_change']:.1f}% of total change)"
        )


    if not lines:
        return "No significant segment-level contributors identified."


    return "\n".join(lines)
