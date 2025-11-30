from pathlib import Path
import zipfile
from datetime import datetime

import pandas as pd

from app.config import (
    KPI_COLUMNS,
    RAW_DATA_FILE,
    AGG_DAILY_FILE,
)

from app.anomaly_engine import build_hybrid_anomalies, load_daily_data
from app.rca_engine import compute_segment_contributions, load_raw_events
from app.pdf_generator import markdown_to_pdf


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
BUNDLE_DIR = REPORTS_DIR 

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)


def _kpi_overview_section(df_daily: pd.DataFrame) -> str:
    lines = ["# KPI Overview\n"]
    lines.append("This section summarizes the overall health of key KPIs.\n")

    stats = df_daily[KPI_COLUMNS].describe().T.reset_index()
    stats.rename(columns={"index": "kpi"}, inplace=True)

    lines.append("| KPI | Mean | Min | Max | Std |")
    lines.append("| --- | ---- | --- | --- | --- |")

    for _, row in stats.iterrows():
        lines.append(
            f"| {row['kpi']} | {row['mean']:.2f} | "
            f"{row['min']:.2f} | {row['max']:.2f} | {row['std']:.2f} |"
        )

    return "\n".join(lines)


def _anomaly_summary_section(df_anom: pd.DataFrame) -> str:
    lines = ["# Anomaly Summary\n"]

    lines.append(
        "This section highlights the most severe anomalies detected by the hybrid engine.\n"
    )

    top = (
        df_anom.sort_values("hybrid_anomaly_score", ascending=False)
        .head(10)
        .copy()
    )

    top["date"] = top["date"].dt.date

    lines.append("| Date | Score | Severity | Is Anomaly |")
    lines.append("| ---- | ------ | -------- | ---------- |")
    
    for _, row in top.iterrows():
        lines.append(
            f"| {row['date']} | {row['hybrid_anomaly_score']:.3f} | "
            f"{row.get('severity','')} | {bool(row['is_hybrid_anomaly'])} |"
        )
    
    return "\n".join(lines)


def _rca_summary_section(
        raw_df: pd.DataFrame, 
        df_anom: pd.DataFrame
) -> str:

    lines = ["# Root-Cause Summary\n"]

    lines.append(
        "This section lists top segment-level contributors for the most recent anomaly.\n"
    )

    anomalies = df_anom[df_anom["is_hybrid_anomaly"]].sort_values(
        "date", ascending=False
    )

    if anomalies.empty:
        lines.append("_No anomalies were detected in the current window._")

        return "\n".join(lines)

    latest = anomalies.iloc[0]
    latest_date = latest["date"]


    lines.append(f"- Latest anomaly date: **{latest_date.date()}**\n")


    for kpi in KPI_COLUMNS:
        if kpi not in df_anom.columns:
            continue

        seg_df = compute_segment_contributions(raw_df, latest_date, kpi)
       
        if seg_df.empty:
            continue

        lines.append(f"## {kpi.replace('_', ' ').title()}\n")
        lines.append(
            "| Channel | Region | Cohort | Current | Baseline | Δ | % Change | % of Total Change |"
        )

        lines.append(
            "| ------- | ------ | ------ | ------- | -------- |---|----------|-------------------|"
        )

        for _, row in seg_df.head(5).iterrows():
            lines.append(
                f"| {row['channel']} | {row['region']} | {row['cohort']} | "
                f"{row['current_value']:.2f} | {row['baseline_value']:.2f} | "
                f"{row['abs_change']:.2f} | "
                f"{row['pct_change']:.1f}% | "
                f"{row['contribution_pct_of_change']:.1f}% |"
            )

        lines.append("")

    return "\n".join(lines)


def _recommendations_section() -> str:

    return """# Recommendation Summary

This section consolidates key actions for Product, Growth, and Engineering teams.

## Product & UX
- Review journeys around checkout, payment and order confirmation.
- Prioritise fixes for segments (channel/region/cohort) that contributed the most to recent drops.

## Growth & Marketing
- Launch win-back campaigns targeting users affected during anomaly windows.
- Monitor performance of campaigns against KPIs like conversion rate and revenue.

## Engineering & Reliability
- Add monitoring around payment failures, latency, and error rates.
- Instrument alerts for major/critical severity anomalies in core KPIs.

## Business Analyst Takeaways
- Track post-action KPI movement to confirm if anomalies are resolved.
- Document each anomaly with date, impacted KPIs, and actions taken.
"""


def _executive_summary_section(
        df_daily: pd.DataFrame, 
        df_anom: pd.DataFrame
) -> str:
    latest_date = df_daily["date"].max().date()
    total_days = df_daily["date"].nunique()
    anomaly_days = int(df_anom["is_hybrid_anomaly"].sum())

    return f"""# Executive Summary

**Period analysed:** Last {total_days} days  
**Data refreshed on:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Latest date in data:** {latest_date}  

## High-level KPI Health

- Total KPIs monitored: **{len(KPI_COLUMNS)}**
- Days with anomalies (hybrid engine): **{anomaly_days}**
- Engine used: **Prophet + Isolation Forest (Hybrid)**

Overall, the system continuously monitors revenue, orders, conversion, DAU and other
business-critical KPIs. Any abnormal deviation from forecasted ranges is surfaced as
an anomaly, with severity levels (Normal / Minor / Major / Critical).
"""


def run_all_reports() -> dict:
    """
    Generate all markdown + PDF reports and return paths.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_daily = load_daily_data(AGG_DAILY_FILE)
    df_anom = build_hybrid_anomalies(df_daily)
    raw_df = load_raw_events(RAW_DATA_FILE)

    # --- Markdown files ---
    files = {}

    exec_md = REPORTS_DIR / "executive_summary.md"

    exec_md.write_text(
        _executive_summary_section(df_daily, df_anom), encoding="utf-8"
    )

    files["executive_summary_md"] = exec_md

    kpi_md = REPORTS_DIR / "business_overview.md"
    kpi_md.write_text(_kpi_overview_section(df_daily), encoding="utf-8")

    files["business_overview_md"] = kpi_md

    anomaly_md = REPORTS_DIR / "clv_summary.md"  
    anomaly_md.write_text(_anomaly_summary_section(df_anom), encoding="utf-8")

    files["anomaly_summary_md"] = anomaly_md

    rca_md = REPORTS_DIR / "persona_insights.md"
    rca_md.write_text(_rca_summary_section(raw_df, df_anom), encoding="utf-8")

    files["rca_summary_md"] = rca_md

    rec_md = REPORTS_DIR / "frm_report.md"
    rec_md.write_text(_recommendations_section(), encoding="utf-8")

    files["recommendations_md"] = rec_md

    # --- PDFs ---
    exec_pdf = REPORTS_DIR / "executive_summary.pdf"
    markdown_to_pdf(exec_md, exec_pdf)

    files["executive_summary_pdf"] = exec_pdf

    pdf_paths = [exec_pdf]

    for key in [
        "business_overview_md", 
        "anomaly_summary_md", 
        "rca_summary_md", 
        "recommendations_md"
    ]:
        md_file = files[key]
        pdf_file = md_file.with_suffix(".pdf")

        markdown_to_pdf(md_file, pdf_file)
        pdf_paths.append(pdf_file)


    # --- Bundle ZIP ---
    bundle_zip = BUNDLE_DIR / "reports_bundle.zip"

    with zipfile.ZipFile(
        bundle_zip, "w", 
        zipfile.ZIP_DEFLATED
    ) as zf:
        for p in pdf_paths:
            zf.write(p, p.name)
            
    files["bundle_zip"] = bundle_zip

    return files