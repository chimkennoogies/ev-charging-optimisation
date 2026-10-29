"""
Utility functions for EV charging optimization project.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, filename: str) -> None:
    """Save a trained model to disk."""
    models_dir = get_project_root() / "models"
    ensure_dir(models_dir)
    filepath = models_dir / filename
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filename: str) -> Any:
    """Load a trained model from disk."""
    models_dir = get_project_root() / "models"
    filepath = models_dir / filename
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def save_metrics(metrics: Dict[str, float], filename: str = "metrics.json") -> None:
    """Save evaluation metrics to JSON file."""
    outputs_dir = get_project_root() / "outputs"
    ensure_dir(outputs_dir)
    filepath = outputs_dir / filename
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(filename: str = "metrics.json") -> Dict[str, float]:
    """Load evaluation metrics from JSON file."""
    outputs_dir = get_project_root() / "outputs"
    filepath = outputs_dir / filename
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def save_figure(fig, filename: str) -> None:
    """Save a matplotlib figure to disk."""
    outputs_dir = get_project_root() / "outputs" / "figures"
    ensure_dir(outputs_dir)
    filepath = outputs_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")


def load_data(filename: str = "ev_charging_data.csv") -> pd.DataFrame:
    """Load the raw dataset."""
    data_dir = get_project_root() / "data" / "raw"
    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return pd.read_csv(filepath)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save processed data to disk."""
    processed_dir = get_project_root() / "data" / "processed"
    ensure_dir(processed_dir)
    filepath = processed_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")
