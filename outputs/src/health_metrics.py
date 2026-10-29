"""
Battery health metric computation for EV charging sessions.

Metrics:
- battery_degradation (lower is better)
- capacity_retention_pct (higher is better)
- charging_efficiency (higher is better)
- thermal_stress_index (lower is better)
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


CHARGER_RATE_MAP = {
    "Level 1": 1.4,
    "Level 2": 7.2,
    "DC Fast": 50.0,
}


def _normalize_charger_type(value: str) -> str:
    text = str(value).lower()
    if "dc" in text or "fast" in text:
        return "DC Fast"
    if "level 1" in text:
        return "Level 1"
    if "level 2" in text:
        return "Level 2"
    return "Level 2"


def apply_charger_type(df: pd.DataFrame, charger_type: str) -> pd.DataFrame:
    """Apply a charger type and corresponding charging rate to a session row."""
    df = df.copy()
    normalized = _normalize_charger_type(charger_type)
    df["Charger Type"] = normalized
    df["Charging Rate (kW)"] = CHARGER_RATE_MAP.get(normalized, 7.2)
    return df


def compute_health_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregate battery health metrics from session data."""
    if df.empty:
        raise ValueError("Cannot compute health metrics on empty dataframe.")

    soc_start = pd.to_numeric(df["State of Charge (Start %)"], errors="coerce").fillna(50).clip(0, 100)
    soc_end = pd.to_numeric(df["State of Charge (End %)"], errors="coerce").fillna(80).clip(0, 100)
    temp = pd.to_numeric(df["Temperature (°C)"], errors="coerce").fillna(25)
    charge_rate = pd.to_numeric(df["Charging Rate (kW)"], errors="coerce").fillna(7).clip(lower=0)
    battery_cap = pd.to_numeric(df["Battery Capacity (kWh)"], errors="coerce").fillna(60).clip(lower=1e-3)
    energy = pd.to_numeric(df["Energy Consumed (kWh)"], errors="coerce").fillna(0).clip(lower=0)

    fast_charge = df["Charger Type"].astype(str).str.lower().str.contains("dc").astype(float)
    soc_gain = (soc_end - soc_start).clip(lower=0)
    high_soc = ((soc_end - 80) / 20).clip(lower=0, upper=1)
    temp_stress = ((temp - 25).abs() / 20).clip(lower=0, upper=1)
    c_rate = (charge_rate / battery_cap).clip(lower=0, upper=2) / 2

    degradation_index = (
        0.45 * fast_charge
        + 0.25 * high_soc
        + 0.20 * temp_stress
        + 0.10 * c_rate
    )
    battery_degradation = float(0.02 + 0.03 * degradation_index.mean())
    thermal_stress_index = float(temp_stress.mean())

    soc_gain_kwh = (soc_gain / 100.0) * battery_cap
    efficiency = (soc_gain_kwh / energy.replace(0, pd.NA)).astype(float).fillna(0.9).clip(0, 1)
    charging_efficiency = float(efficiency.mean())

    capacity_retention_pct = float(max(0.0, 100.0 - battery_degradation * 250.0))

    return {
        "battery_degradation": battery_degradation,
        "capacity_retention_pct": capacity_retention_pct,
        "charging_efficiency": charging_efficiency,
        "thermal_stress_index": thermal_stress_index,
    }
