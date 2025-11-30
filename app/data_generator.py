import numpy as np
import pandas as pd
from app.config import (
    RAW_DATA_FILE, 
    PROCESSED_DATA_DIR, 
    AGG_DAILY_FILE, 
    RANDOM_SEED
)

np.random.seed(RANDOM_SEED)


def generate_synthetic_kpi_data(
    start_date: str = "2024-01-01",
    periods: int = 180,
    freq: str = "D",
    save_raw: bool = True,
    save_aggregated: bool = True,
):
    """Generate synthetic KPI dataset with segments and injected anomalies."""
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)

    channels = ["App", "Web"]
    regions = ["North", "South", "East", "West"]
    cohorts = ["New", "Repeat"]

    rows = []

    for ds in date_range:
        day_of_week = ds.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        weekend_boost = 1.2 if is_weekend else 1.0

        for ch in channels:
            for reg in regions:
                for coh in cohorts:
                    base_sessions = np.random.randint(500, 2000)
                    segment_multiplier = 1.0

                    if ch == "App":
                        segment_multiplier *= 1.2
                    if coh == "New":
                        segment_multiplier *= 1.1
                    if reg in ["North", "West"]:
                        segment_multiplier *= 1.05

                    sessions = int(base_sessions * segment_multiplier * weekend_boost)
                    dau = int(sessions * np.random.uniform(0.4, 0.7))

                    base_cr = 0.04

                    if coh == "Repeat":
                        base_cr += 0.02
                    if ch == "Web":
                        base_cr -= 0.005

                    conversion_rate = np.clip(
                        np.random.normal(base_cr, 0.005), 0.01, 0.10
                    )

                    orders = int(sessions * conversion_rate)

                    payment_success_rate = np.clip(
                        np.random.normal(0.93, 0.02), 0.85, 0.99
                    )

                    successful_payments = int(orders * payment_success_rate)

                    aov = np.random.uniform(300, 800)
                    revenue = successful_payments * aov

                    bounce_rate = np.clip(
                        np.random.normal(0.40, 0.05), 0.25, 0.70
                    )

                    cac = np.random.uniform(80, 150)

                    if coh == "New":
                        cac *= 1.2

                    rows.append(
                        {
                            "date": ds.date(),
                            "channel": ch,
                            "region": reg,
                            "cohort": coh,
                            "sessions": sessions,
                            "dau": dau,
                            "orders": orders,
                            "successful_payments": successful_payments,
                            "revenue": revenue,
                            "conversion_rate": orders / sessions if sessions > 0 else 0,
                            "payment_success_rate": payment_success_rate,
                            "bounce_rate": bounce_rate,
                            "cac": cac,
                            "is_injected_anomaly": 0,
                        }
                    )


    df = pd.DataFrame(rows)


    unique_dates = df["date"].unique()
    n_anom = min(8, len(unique_dates))

    if n_anom > 0:
        anomaly_dates = np.random.choice(unique_dates, size=n_anom, replace=False)
        first_half = anomaly_dates[: n_anom // 2]
        second_half = anomaly_dates[n_anom // 2 :]


        for ad in first_half:
            mask = df["date"] == ad
            df.loc[mask, "sessions"] *= 0.6
            df.loc[mask, "orders"] *= 0.5
            df.loc[mask, "successful_payments"] *= 0.5
            df.loc[mask, "revenue"] *= 0.5
            df.loc[mask, "conversion_rate"] *= 0.7
            df.loc[mask, "is_injected_anomaly"] = 1


        for ad in second_half:
            mask = df["date"] == ad
            df.loc[mask, "payment_success_rate"] *= 0.8
            df.loc[mask, "successful_payments"] *= 0.7
            df.loc[mask, "revenue"] *= 0.75
            df.loc[mask, "is_injected_anomaly"] = 1


    if save_raw:
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_FILE, index=False)


    daily_agg = (
        df.groupby("date")
        .agg(
            {
                "sessions": "sum",
                "dau": "sum",
                "orders": "sum",
                "successful_payments": "sum",
                "revenue": "sum",
                "payment_success_rate": "mean",
                "bounce_rate": "mean",
                "cac": "mean",
                "is_injected_anomaly": "max",
            }
        )
        .reset_index()
    )


    daily_agg["conversion_rate"] = (
        daily_agg["orders"] / daily_agg["sessions"]
    ).replace([np.inf, -np.inf], 0)

    daily_agg["date"] = pd.to_datetime(daily_agg["date"])


    if save_aggregated:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        daily_agg.to_csv(AGG_DAILY_FILE, index=False)


    return df, daily_agg


if __name__ == "__main__":
    raw, daily = generate_synthetic_kpi_data()
    print("Raw shape:", raw.shape)
    print("Daily shape:", daily.shape)
    print("Saved to:", RAW_DATA_FILE, "and", AGG_DAILY_FILE)