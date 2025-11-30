import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from app.data_generator import generate_synthetic_kpi_data
from app.anomaly_engine import build_hybrid_anomalies


def test_build_hybrid_anomalies():
    _, daily = generate_synthetic_kpi_data(
        periods=60, save_raw=False, save_aggregated=False
    )
    
    df_anom = build_hybrid_anomalies(daily)

    assert "hybrid_anomaly_score" in df_anom.columns
    assert "is_hybrid_anomaly" in df_anom.columns
    assert "severity" in df_anom.columns
    assert df_anom["hybrid_anomaly_score"].notnull().all()