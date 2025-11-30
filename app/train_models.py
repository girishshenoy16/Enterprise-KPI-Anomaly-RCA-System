# app/train_models.py
from app.anomaly_engine import load_daily_data, compute_isolation_forest_scores
from app.config import AGG_DAILY_FILE


def train_isolation_forest():
    """
    Train or refresh the IsolationForest model and save it to /models.
    """
    df_daily = load_daily_data(AGG_DAILY_FILE)
    _ = compute_isolation_forest_scores(df_daily)
    print("IsolationForest anomaly model trained and saved.")
    return True


if __name__ == "__main__":
    train_isolation_forest()