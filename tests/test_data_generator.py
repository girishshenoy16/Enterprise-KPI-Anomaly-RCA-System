import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from app.data_generator import generate_synthetic_kpi_data


def test_generate_synthetic_kpi_data_shapes():
    raw, daily = generate_synthetic_kpi_data(
        periods=30, save_raw=False, save_aggregated=False
    )
    
    assert not raw.empty
    assert not daily.empty
    assert "revenue" in raw.columns
    assert "conversion_rate" in daily.columns