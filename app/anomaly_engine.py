import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from prophet import Prophet

from app.config import (
    AGG_DAILY_FILE,
    ANOMALY_OUTPUT_FILE,
    KPI_COLUMNS,
    Z_SCORE_THRESHOLD,
    IFOREST_CONTAMINATION,
    RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)


def load_daily_data(path=None) -> pd.DataFrame:
    path = path or AGG_DAILY_FILE
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day

    for col in ["revenue", "orders", "dau"]:
        if col in df.columns:
            df[f"{col}_7dma"] = df[col].rolling(7, min_periods=1).mean()
            df[f"{col}_30dma"] = df[col].rolling(30, min_periods=1).mean()

    df = df.fillna(0)

    return df


def fit_prophet_for_kpi(
        df_daily: pd.DataFrame, 
        kpi: str
) -> pd.DataFrame:
    df_kpi = df_daily[["date", kpi]].rename(columns={"date": "ds", kpi: "y"})

    model = Prophet(daily_seasonality=True, yearly_seasonality=False)
    model.fit(df_kpi)

    forecast = model.predict(df_kpi[["ds"]])

    merged = df_kpi.merge(
        forecast[[
            "ds", 
            "yhat", 
            "yhat_lower", 
            "yhat_upper"
        ]], on="ds", how="left"
    )

    merged["residual"] = merged["y"] - merged["yhat"]

    mu = merged["residual"].mean()
    sigma = merged["residual"].std() or 1.0
    merged["z_score"] = (merged["residual"] - mu) / sigma

    merged["prophet_is_anomaly"] = merged["z_score"].abs() > Z_SCORE_THRESHOLD
    merged = merged.rename(columns={"ds": "date"})
    
    return merged


def compute_isolation_forest_scores(df_daily: pd.DataFrame) -> pd.DataFrame:
    from app.model_registry import save_model, load_model

    feat_df = df_daily.copy().sort_values("date")

    # SAFETY FEATURES: ensure columns exist (important for tests and robustness)
    if "day_of_week" not in feat_df.columns:
        feat_df["day_of_week"] = feat_df["date"].dt.dayofweek

    if "day_of_month" not in feat_df.columns:
        feat_df["day_of_month"] = feat_df["date"].dt.day

    # Rolling windows may be missing if load_daily_data() wasn't used
    for col in ["revenue", "orders", "dau"]:
        if col in feat_df.columns:
            for win in [7, 30]:
                name = f"{col}_{win}dma"
                if name not in feat_df.columns:
                    feat_df[name] = feat_df[col].rolling(win, min_periods=1).mean()

    feature_cols = [
        *KPI_COLUMNS,
        "day_of_week",
        "day_of_month",
    ]
    for col in ["revenue", "orders", "dau"]:
        for win in [7, 30]:
            name = f"{col}_{win}dma"
            if name in feat_df.columns:
                feature_cols.append(name)

    features = feat_df[feature_cols].fillna(0)

    # 🔹 Try to load existing model; if not present, train and save
    iforest = load_model("isolation_forest")
    if iforest is None:
        iforest = IsolationForest(
            n_estimators=200,
            contamination=IFOREST_CONTAMINATION,
            random_state=RANDOM_SEED,
        )
        iforest.fit(features)
        save_model(iforest, "isolation_forest")
    else:
        # Optional: if you want to be extra safe, you could check feature count here
        pass

    scores = -iforest.decision_function(features)
    preds = iforest.predict(features)

    feat_df["iforest_score"] = scores
    feat_df["iforest_is_anomaly"] = preds == -1

    return feat_df[["date", "iforest_score", "iforest_is_anomaly"]]



def build_hybrid_anomalies(
        df_daily: pd.DataFrame | None = None
) -> pd.DataFrame:
    if df_daily is None:
        df_daily = load_daily_data()

    base = df_daily.copy().sort_values("date")

    for kpi in KPI_COLUMNS:
        if kpi not in base.columns:
            continue

        prophet_df = fit_prophet_for_kpi(base, kpi)
        prophet_df = prophet_df[
            [
                "date", 
                "yhat", 
                "residual", 
                "z_score", 
                "prophet_is_anomaly"
            ]
        ].rename(
            columns={
                "yhat": f"{kpi}_yhat",
                "residual": f"{kpi}_residual",
                "z_score": f"{kpi}_z",
                "prophet_is_anomaly": f"{kpi}_prophet_is_anomaly",
            }
        )

        base = base.merge(prophet_df, on="date", how="left")


    if_df = compute_isolation_forest_scores(base)
    base = base.merge(if_df, on="date", how="left")

    z_cols = [
        f"{k}_z" for k in KPI_COLUMNS 
        if f"{k}_z" in base.columns
    ]

    if z_cols:
        base["z_abs_mean"] = base[z_cols].abs().mean(axis=1)
    else:
        base["z_abs_mean"] = 0.0

    if_min = base["iforest_score"].min()
    if_max = base["iforest_score"].max()

    denom = (if_max - if_min) or 1.0

    base["iforest_score_norm"] = (base["iforest_score"] - if_min) / denom

    base["hybrid_anomaly_score"] = 0.6 * base["z_abs_mean"] + 0.4 * base[
        "iforest_score_norm"
    ]

    threshold = base["hybrid_anomaly_score"].quantile(0.90)

    base["is_hybrid_anomaly"] = base["hybrid_anomaly_score"] >= threshold

    try:
        base["severity"] = pd.qcut(
            base["hybrid_anomaly_score"],
            q=[0, 0.80, 0.90, 1.0],

            labels=[
                "Normal", 
                "Minor", 
                "Major", 
                "Critical"
            ],
        )
    except ValueError:
        base["severity"] = "Normal"
        base.loc[base["is_hybrid_anomaly"], "severity"] = "Major"

    return base


def save_anomalies_to_csv(output_path=None) -> pd.DataFrame:
    df_daily = load_daily_data()
    anomalies_df = build_hybrid_anomalies(df_daily)

    output_path = output_path or ANOMALY_OUTPUT_FILE
    anomalies_df.to_csv(output_path, index=False)

    return anomalies_df


if __name__ == "__main__":
    df = save_anomalies_to_csv()
    print("Anomaly file saved to:", ANOMALY_OUTPUT_FILE)
    print(df[["date", "hybrid_anomaly_score", "is_hybrid_anomaly", "severity"]].tail())