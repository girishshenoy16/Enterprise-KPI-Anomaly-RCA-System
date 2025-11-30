import pandas as pd
from app.rca_engine import compute_segment_contributions, summarize_root_cause


def generate_insight_text(
    raw_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    target_date: pd.Timestamp,
    kpi: str,
) -> str:
    target_date = pd.to_datetime(target_date)

    daily_row = anomalies_df[anomalies_df["date"] == target_date]

    if daily_row.empty:
        return f"No anomaly details found for {target_date.date()}."

    daily_row = daily_row.iloc[0]
    actual_value = daily_row[kpi]

    forecast_col = f"{kpi}_yhat"
    forecast_value = daily_row.get(forecast_col, float("nan"))

    segment_df = compute_segment_contributions(raw_df, target_date, kpi)
    rca_text = summarize_root_cause(segment_df, kpi)

    direction = "dropped"
    try:
        if pd.notna(forecast_value) and actual_value >= forecast_value:
            direction = "increased"
    except Exception:
        pass

    insight = (
        f"{kpi.replace('_', ' ').title()} {direction} on {target_date.date()}.\n"
        f"- Actual: {actual_value:,.2f}\n"
        f"- Expected (forecast): {forecast_value:,.2f}\n\n"
        f"Top root-cause contributors:\n{rca_text}"
    )

    return insight


def suggest_actions(kpi: str) -> str:
    k = kpi.lower()
    if "revenue" in k or "orders" in k or "conversion" in k:

        return (
            "- Investigate traffic sources with highest drop.\n"
            "- Check payment failures and cart-abandon flows.\n"
            "- Launch recovery campaigns for users impacted during anomaly window."
        )
    
    if "payment" in k:

        return (
            "- Validate payment gateway uptime and error logs.\n"
            "- Coordinate with payment provider for UPI/card failure spikes.\n"
            "- Show alternative payment options and retries to affected users."
        )
    
    if "dau" in k or "sessions" in k:

        return (
            "- Check app crashes, release logs, and store reviews.\n"
            "- Inspect marketing spend and campaign pauses.\n"
            "- Re-engage churn-risk cohorts via notifications or email."
        )
    
    if "bounce" in k:

        return (
            "- Review page load times and core web vitals.\n"
            "- A/B test landing page content and CTAs.\n"
            "- Ensure traffic quality from ad campaigns."
        )

    return (
        "- Perform technical health checks (errors, latency, downtime).\n"
        "- Analyze segment-level impact and address top contributors first.\n"
        "- Align with product, engineering, and marketing on remedial plan."
    )