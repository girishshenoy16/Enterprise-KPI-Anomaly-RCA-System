import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
    
from app.data_generator import generate_synthetic_kpi_data
from app.rca_engine import compute_segment_contributions


def test_compute_segment_contributions():
    raw, _ = generate_synthetic_kpi_data(
        periods=20, save_raw=False, save_aggregated=False
    )

    any_date = raw["date"].iloc[10]
    seg_df = compute_segment_contributions(raw, any_date, "revenue")

    assert "current_value" in seg_df.columns
    assert "baseline_value" in seg_df.columns