from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANOMALIES_DIR = DATA_DIR / "anomalies"
MODELS_DIR = BASE_DIR / "models"

for p in [
    DATA_DIR, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    ANOMALIES_DIR, 
    MODELS_DIR
]:
    p.mkdir(parents=True, exist_ok=True)


RAW_DATA_FILE = RAW_DATA_DIR / "kpi_events.csv"
AGG_DAILY_FILE = PROCESSED_DATA_DIR / "kpi_daily_aggregated.csv"
ANOMALY_OUTPUT_FILE = ANOMALIES_DIR / "kpi_daily_anomalies.csv"

KPI_COLUMNS = [
    "revenue",
    "orders",
    "conversion_rate",
    "dau",
    "payment_success_rate",
    "bounce_rate",
    "cac",
]

SEGMENT_COLUMNS = [
    "channel", 
    "region", 
    "cohort"
]

Z_SCORE_THRESHOLD = 2.5
IFOREST_CONTAMINATION = 0.05

RANDOM_SEED = 42