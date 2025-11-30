# app/model_registry.py
from pathlib import Path
import joblib
from typing import Any

from app.config import MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, name: str) -> Path:
    """
    Save a model under models/{name}.pkl and return the path.
    """
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def load_model(name: str) -> Any | None:
    """
    Load a model if it exists, otherwise return None.
    """
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        # If model is incompatible/corrupted, ignore and retrain
        return None