import math
from typing import Dict, Tuple
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# Add the directory that contains the `src` package to the import path.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "outputs"))

from src.utils import load_model, get_project_root
from src.feature_engineering import FeatureEngineer
from results import render_results_tab

# ==========================
# Model Loading & Initialization
# ==========================

@st.cache_resource
def load_trained_model_and_features():
    """
    Load the trained ML model and feature engineer.
    Uses Streamlit cache to avoid reloading on every interaction.
    """
    try:
        model = load_model("trained_model.pkl")
        feature_engineer = FeatureEngineer()
        
        # Try to load feature columns if they were saved
        models_dir = get_project_root() / "models"
        import pickle
        feature_cols_path = models_dir / "feature_columns.pkl"
        if feature_cols_path.exists():
            with open(feature_cols_path, 'rb') as f:
                feature_columns = pickle.load(f)
        else:
            feature_columns = None
        
        # Load processed training data for reference in feature engineering
        try:
            training_data = pd.read_csv(get_project_root() / "data" / "processed" / "processed_data.csv")
            # Limit to avoid memory issues
            if len(training_data) > 500:
                training_data = training_data.sample(n=500, random_state=42)
        except:
            training_data = None
        
        return model, feature_engineer, feature_columns, training_data
    except Exception as e:
        st.error(f"Error loading trained model: {str(e)}")
        st.info("Make sure you've run main.py first to train the model.")
        return None, None, None, None


# ==========================
# Domain Logic / Utilities
# ==========================


def get_base_efficiency_kwh_per_100km(vehicle_model: str) -> float:
    """
    Estimate base energy consumption (kWh / 100 km) by vehicle model.

    This is a simple heuristic; in a real system you'd replace this with
    model-specific data or an ML model.
    """
    model_efficiency_map = {
        "Tesla Model 3": 15.0,
        "Tesla Model Y": 16.5,
        "Nissan Leaf": 17.0,
        "Hyundai Kona Electric": 15.5,
        "Kia e-Niro": 16.0,
        "BMW i3": 17.5,
        "Other / Unknown": 18.0,
    }
    return model_efficiency_map.get(vehicle_model, 18.0)


def estimate_trip_energy_kwh(distance_km: float, vehicle_model: str) -> float:
    """
    Estimate energy required for the trip in kWh.
    """
    efficiency = get_base_efficiency_kwh_per_100km(vehicle_model)
    energy_kwh = (distance_km * efficiency) / 100.0
    return max(energy_kwh, 0.0)


def compute_safety_buffer_factor(temperature_c: float, idle_hours: float) -> float:
    """
    Compute a multiplicative safety buffer factor based on:
    - Temperature (extreme hot/cold increases buffer)
    - Idle time (long idle → more buffer for auxiliary loads / self-discharge)

    Returns a factor >= 1.0.
    """
    # Base buffer: 10%
    buffer_pct = 0.10

    # Temperature effect
    # Very cold or very hot temperatures reduce effective range
    if temperature_c <= -10:
        buffer_pct += 0.25  # +25% buffer
    elif -10 < temperature_c <= 0:
        buffer_pct += 0.18
    elif 0 < temperature_c <= 10:
        buffer_pct += 0.10
    elif 25 < temperature_c <= 35:
        buffer_pct += 0.08
    elif temperature_c > 35:
        buffer_pct += 0.15

    # Idle time effect
    if idle_hours > 12:
        buffer_pct += 0.08
    elif idle_hours > 4:
        buffer_pct += 0.04

    # Cap buffer at a reasonable level
    buffer_pct = min(buffer_pct, 0.60)  # max +60%

    return 1.0 + buffer_pct


def soc_to_energy_kwh(soc_percent: float, battery_capacity_kwh: float) -> float:
    """Convert SOC (%) to energy (kWh)."""
    return (soc_percent / 100.0) * battery_capacity_kwh


def energy_kwh_to_soc_percent(energy_kwh: float, battery_capacity_kwh: float) -> float:
    """Convert energy (kWh) to SOC (%)."""
    if battery_capacity_kwh <= 0:
        return 0.0
    return (energy_kwh / battery_capacity_kwh) * 100.0


def get_charger_power_kw(charger_type: str) -> float:
    """Approximate charger power (kW) for different charger types."""
    charger_power_map = {
        "Level 1 (AC, ~1.4 kW)": 1.4,
        "Level 2 (AC, ~7.2 kW)": 7.2,
        "DC Fast (~50 kW)": 50.0,
    }
    return charger_power_map.get(charger_type, 7.2)


def estimate_charging_duration_hours(
    energy_to_charge_kwh: float,
    charger_type: str,
) -> float:
    """
    Estimate charging duration in hours given energy to charge and charger type.
    Applies a simple tapering factor to account for slowing near high SOC.
    """
    raw_power_kw = get_charger_power_kw(charger_type)

    # Simple tapering assumption: effective power reduced when charging above ~80%
    taper_factor = 0.90
    effective_power_kw = raw_power_kw * taper_factor

    if effective_power_kw <= 0:
        return 0.0

    return max(energy_to_charge_kwh / effective_power_kw, 0.0)


def recommend_charging_strategy(
    distance_km: float,
    temperature_c: float,
    idle_hours: float,
    current_soc: float,
    vehicle_model: str,
    battery_capacity_kwh: float,
    charger_type: str,
    max_preferred_soc: float = 85.0,
    min_health_soc: float = 20.0,
) -> Tuple[Dict, Dict]:
    """
    Recommend target SOC, energy to charge, and duration.

    Returns:
        results: dict with numeric outputs
        messages: dict with human-readable explanations
    """
    # Sanity checks
    battery_capacity_kwh = max(battery_capacity_kwh, 1.0)
    current_soc = min(max(current_soc, 0.0), 100.0)
    max_preferred_soc = min(max(max_preferred_soc, min_health_soc), 100.0)

    # 1) Energy needed for the trip
    trip_energy_kwh = estimate_trip_energy_kwh(distance_km, vehicle_model)

    # 2) Apply safety buffer
    safety_factor = compute_safety_buffer_factor(temperature_c, idle_hours)
    total_required_energy_kwh = trip_energy_kwh * safety_factor

    # 3) Convert to required SOC
    required_soc_for_trip = energy_kwh_to_soc_percent(
        total_required_energy_kwh, battery_capacity_kwh
    )

    # 4) Apply battery health constraints
    upper_bound_soc = min(max_preferred_soc, 85.0)
    lower_bound_soc = min_health_soc

    # Aim to finish charging with enough SOC for the trip + a buffer
    target_soc_raw = required_soc_for_trip + lower_bound_soc
    target_soc = min(max(target_soc_raw, lower_bound_soc), upper_bound_soc)

    if target_soc <= current_soc:
        target_soc = current_soc
        energy_to_charge_kwh = 0.0
        charging_duration_hours = 0.0
        needs_charging = False
    else:
        needs_charging = True
        current_energy_kwh = soc_to_energy_kwh(current_soc, battery_capacity_kwh)
        target_energy_kwh = soc_to_energy_kwh(target_soc, battery_capacity_kwh)
        energy_to_charge_kwh = max(target_energy_kwh - current_energy_kwh, 0.0)

        charging_duration_hours = estimate_charging_duration_hours(
            energy_to_charge_kwh, charger_type
        )

    # Battery health messaging
    health_msgs = []
    if upper_bound_soc < 100.0:
        health_msgs.append(
            f"To preserve battery health, the recommended upper SOC limit is about {upper_bound_soc:.0f}%."
        )
    if lower_bound_soc > 0.0:
        health_msgs.append(
            f"Keeping SOC above roughly {lower_bound_soc:.0f}% helps reduce deep discharge stress."
        )
    if not needs_charging:
        health_msgs.append(
            "Your current SOC already covers tomorrow’s trip with the safety buffer applied, "
            "so no additional charging is needed."
        )
    else:
        if target_soc >= upper_bound_soc - 1:
            health_msgs.append(
                "The recommended target SOC is near your configured maximum; "
                "consider slightly reducing it if you frequently charge to high levels."
            )
        if distance_km > 0 and required_soc_for_trip > 70:
            health_msgs.append(
                "Because tomorrow’s trip is relatively long, the system suggests a higher SOC "
                "while still staying within battery-friendly limits."
            )

    results = {
        "trip_energy_kwh": trip_energy_kwh,
        "safety_factor": safety_factor,
        "total_required_energy_kwh": total_required_energy_kwh,
        "required_soc_for_trip": required_soc_for_trip,
        "target_soc": target_soc,
        "energy_to_charge_kwh": energy_to_charge_kwh,
        "charging_duration_hours": charging_duration_hours,
        "needs_charging": needs_charging,
    }

    messages = {
        "health_advice": " ".join(health_msgs),
        "charger_power_kw": get_charger_power_kw(charger_type),
    }

    return results, messages


def format_duration(hours: float) -> str:
    """Convert decimal hours to a human-readable string."""
    if hours <= 0:
        return "0 h"
    h = int(hours)
    m = int(round((hours - h) * 60))
    if h > 0 and m > 0:
        return f"{h} h {m} min"
    if h > 0:
        return f"{h} h"
    return f"{m} min"


def predict_charging_cost_ml(
    distance_km: float,
    temperature_c: float,
    idle_hours: float,
    current_soc: float,
    vehicle_model: str,
    battery_capacity_kwh: float,
    charger_type: str,
    energy_to_charge_kwh: float = None,
    model = None,
    feature_engineer = None,
    feature_columns = None,
    training_data = None,
) -> Dict:
    """
    Use the trained ML model to predict charging cost.
    Falls back to rule-based estimation if model is unavailable.
    
    Returns:
        Dictionary with predicted cost and explanation
    """
    result = {
        "predicted_cost": None,
        "cost_available": False,
        "explanation": "Model not available",
        "error": None
    }
    
    if model is None or feature_engineer is None:
        result["explanation"] = "ML model not loaded. Using rule-based estimation."
        return result
    
    try:
        # Build a comprehensive feature DataFrame for prediction
        # Include all columns that the model might expect
        from datetime import datetime
        now = datetime.now()
        
        # Calculate derived values
        estimated_charging_hours = 2.0  # Default estimate
        if energy_to_charge_kwh and energy_to_charge_kwh > 0:
            estimated_charging_hours = energy_to_charge_kwh / 7.2  # Assume Level 2 as default
        
        target_soc = current_soc + 20  # Assume charging 20% more
        estimated_energy_consumed = energy_to_charge_kwh if energy_to_charge_kwh else 10.0
        estimated_charging_rate = estimated_energy_consumed / max(estimated_charging_hours, 0.1)
        
        # Get charger power
        charger_power_map = {
            "Level 1 (AC, ~1.4 kW)": 1.4,
            "Level 2 (AC, ~7.2 kW)": 7.2,
            "DC Fast (~50 kW)": 50.0,
        }
        charger_power = charger_power_map.get(charger_type, 7.2)
        
        session_data = pd.DataFrame({
            'Distance Driven (since last charge) (km)': [distance_km],
            'Temperature (°C)': [temperature_c],
            'State of Charge (Start %)': [current_soc],
            'State of Charge (End %)': [min(target_soc, 100)],
            'Battery Capacity (kWh)': [battery_capacity_kwh],
            'Energy Consumed (kWh)': [estimated_energy_consumed],
            'Charging Rate (kW)': [estimated_charging_rate],
            'Charging Duration (hours)': [estimated_charging_hours],
            'Vehicle Model': [vehicle_model],
            'Charger Type': [charger_type],
            'Vehicle Age (years)': [5.0],  # Default assumption
            'User Type': ['Commuter'],  # Default assumption
            'Charging Station Location': ['New York'],  # Default assumption
            'Time of Day': ['Evening'],  # Default assumption
            'User ID': ['User_999'],  # String ID to match training data format
            'Charging Station ID': ['Station_999'],  # String ID to match training data format
            'day_name': [now.strftime('%A')],
            'month': [now.month],
            'day_of_month': [now.day],
            'hour': [now.hour],
            'Charging Cost (USD)': [0.0],  # Dummy target (not used for prediction)
        })
        
        # For feature engineering, we need to pass training data to get proper user/station statistics
        # If training data is available, use it; otherwise feature engineer will create dummy features
        session_features = feature_engineer.engineer_features(
            session_data.copy(),
            train_df=training_data,  # Pass training data for reference
            encode_categorical=True,
            original_train_df=training_data if training_data is not None else session_data.copy()
        )
        
        # Get required feature columns
        if feature_columns:
            req_features = feature_columns
        else:
            # Fallback: get feature columns dynamically
            req_features = [col for col in session_features.columns 
                           if col != 'Charging Cost (USD)']
        
        # Ensure all required features are present
        for feat in req_features:
            if feat not in session_features.columns:
                session_features[feat] = 0.0
        
        # Select only required features in correct order
        X = session_features[[f for f in req_features if f in session_features.columns]].fillna(0)
        
        # Verify we have all features
        if len(X.columns) != len(req_features):
            missing = set(req_features) - set(X.columns)
            # Add missing features with 0
            for feat in missing:
                X[feat] = 0.0
            X = X[[f for f in req_features if f in X.columns]]
        
        # Make prediction
        predicted_cost = model.predict(X)[0]
        
        result["predicted_cost"] = max(predicted_cost, 0.0)  # Ensure non-negative
        result["cost_available"] = True
        result["explanation"] = "Prediction from trained ML model"
        
    except Exception as e:
        result["error"] = str(e)
        result["explanation"] = f"Error using ML model: {str(e)}"
    
    return result


# ==========================
# Streamlit UI
# ==========================


def render_recommendation_tab(
    model,
    feature_engineer,
    feature_columns,
    training_data,
) -> None:
    st.markdown("### Inputs")

    with st.form("ev_input_form"):
        col1, col2 = st.columns(2)

        with col1:
            distance_km = st.number_input(
                "Distance to be travelled tomorrow (km)",
                min_value=0.0,
                value=50.0,
                step=10.0,
                help="Approximate total distance you expect to drive tomorrow.",
            )

            temperature_c = st.number_input(
                "Expected outdoor temperature (°C)",
                min_value=-40.0,
                max_value=60.0,
                value=20.0,
                step=1.0,
                help="Colder or hotter temperatures reduce effective range and increase buffer.",
            )

            idle_hours = st.number_input(
                "Time vehicle will remain parked before the trip (hours)",
                min_value=0.0,
                value=10.0,
                step=1.0,
                help="Longer idle time increases the safety buffer.",
            )

            current_soc = st.slider(
                "Current battery SOC (%)",
                min_value=0,
                max_value=100,
                value=40,
                help="Approximate current state of charge.",
            )

            max_preferred_soc = st.slider(
                "Maximum preferred SOC for daily charging (%)",
                min_value=50,
                max_value=100,
                value=85,
                help="Charging above ~80–90% regularly can accelerate battery degradation.",
            )

        with col2:
            vehicle_model = st.selectbox(
                "Vehicle model",
                [
                    "Tesla Model 3",
                    "Tesla Model Y",
                    "Nissan Leaf",
                    "Hyundai Kona Electric",
                    "Kia e-Niro",
                    "BMW i3",
                    "Other / Unknown",
                ],
            )

            battery_capacity_kwh = st.number_input(
                "Battery capacity (kWh)",
                min_value=10.0,
                max_value=150.0,
                value=60.0,
                step=1.0,
                help="Nominal battery capacity of the vehicle.",
            )

            charger_type = st.selectbox(
                "Charger type",
                [
                    "Level 1 (AC, ~1.4 kW)",
                    "Level 2 (AC, ~7.2 kW)",
                    "DC Fast (~50 kW)",
                ],
                help="Type of charger you will use while the car is parked.",
            )

            parked_hours_for_charging = st.number_input(
                "Time you can keep the car plugged in (hours)",
                min_value=0.0,
                value=8.0,
                step=0.5,
                help="Used only to help interpret whether the charging window is sufficient.",
            )

        submitted = st.form_submit_button("Compute Recommendation")

    if not submitted:
        st.info("Fill in the inputs and click **Compute Recommendation** to see suggestions.")
        return

    # Compute recommendation
    results, messages = recommend_charging_strategy(
        distance_km=distance_km,
        temperature_c=temperature_c,
        idle_hours=idle_hours,
        current_soc=current_soc,
        vehicle_model=vehicle_model,
        battery_capacity_kwh=battery_capacity_kwh,
        charger_type=charger_type,
        max_preferred_soc=max_preferred_soc,
    )

    target_soc = results["target_soc"]
    energy_to_charge_kwh = results["energy_to_charge_kwh"]
    charging_duration_hours = results["charging_duration_hours"]
    needs_charging = results["needs_charging"]
    charger_power_kw = messages["charger_power_kw"]

    # Get ML cost prediction if model is available
    ml_prediction = None
    if model is not None and feature_engineer is not None:
        ml_prediction = predict_charging_cost_ml(
            distance_km=distance_km,
            temperature_c=temperature_c,
            idle_hours=idle_hours,
            current_soc=current_soc,
            vehicle_model=vehicle_model,
            battery_capacity_kwh=battery_capacity_kwh,
            charger_type=charger_type,
            energy_to_charge_kwh=energy_to_charge_kwh,
            model=model,
            feature_engineer=feature_engineer,
            feature_columns=feature_columns,
            training_data=training_data,
        )

    st.markdown("### Recommended Charging Plan")

    # Metric cards
    mcol1, mcol2, mcol3 = st.columns(3)

    with mcol1:
        st.metric(
            label="Recommended Target SOC",
            value=f"{target_soc:.0f} %",
            delta=f"{target_soc - current_soc:+.0f} %-points",
        )

    with mcol2:
        st.metric(
            label="Energy to Add",
            value=f"{energy_to_charge_kwh:.1f} kWh",
            help="Approximate energy needed to reach the recommended SOC.",
        )

    with mcol3:
        st.metric(
            label="Estimated Charging Duration",
            value=format_duration(charging_duration_hours),
            help=f"Assuming ~{charger_power_kw:.1f} kW effective power for {charger_type}.",
        )

    # ML Model Cost Prediction (if available)
    if ml_prediction is not None and ml_prediction["cost_available"]:
        st.markdown("### ML Model Cost Prediction")
        st.metric(
            label="Predicted Charging Cost (ML Model)",
            value=f"${ml_prediction['predicted_cost']:.2f}",
            help=ml_prediction["explanation"],
        )
        st.info(f"📊 {ml_prediction['explanation']}")
    elif ml_prediction is not None and ml_prediction.get("error"):
        st.warning(f"⚠️ Could not generate ML prediction: {ml_prediction['error']}")

    # SOC progress visualization
    st.markdown("### SOC Trajectory")
    soc_col1, soc_col2 = st.columns([2, 1])

    with soc_col1:
        st.write(
            f"Current SOC: **{current_soc:.0f}%** → Recommended SOC: **{target_soc:.0f}%**"
        )
        progress_value = min(max(target_soc / 100.0, 0.0), 1.0)
        st.progress(progress_value)

    with soc_col2:
        st.caption("Battery health guidance SOC band")
        st.write("Healthy daily range: **20% – 85%**")
        if target_soc >= 85:
            st.warning(
                "Target SOC is at the upper health limit. Consider using fast charging only when necessary."
            )
        elif target_soc <= 30:
            st.info(
                "Target SOC is on the lower side. This is fine for short trips but avoid frequent deep discharges."
            )

    # Battery health advice
    st.markdown("### Battery Health Advice")
    st.write(messages["health_advice"])

    # Detailed breakdown
    with st.expander("See detailed energy and buffer breakdown"):
        st.markdown("#### Trip Energy Estimate")
        st.write(f"- Distance: **{distance_km:.1f} km**")
        st.write(f"- Estimated trip energy: **{results['trip_energy_kwh']:.1f} kWh**")

        st.markdown("#### Safety Buffer")
        st.write(f"- Safety factor applied: **× {results['safety_factor']:.2f}**")
        st.write(
            f"- Total required energy (trip + buffer): **{results['total_required_energy_kwh']:.1f} kWh**"
        )
        st.write(
            f"- Required SOC for trip (with buffer): **{results['required_soc_for_trip']:.1f}%**"
        )

        st.markdown("#### Charging Window Check")
        if charging_duration_hours > parked_hours_for_charging and needs_charging:
            st.warning(
                f"The estimated charging time ({format_duration(charging_duration_hours)}) "
                f"may exceed your available plugged-in time ({format_duration(parked_hours_for_charging)}). "
                "You may want to start charging earlier or use a faster charger if possible."
            )
        else:
            st.success(
                "Your available plugged-in time looks sufficient for the recommended charge."
            )

    if not needs_charging:
        st.success(
            "Based on your inputs, no additional charging is strictly required to cover tomorrow’s trip "
            "while maintaining a healthy SOC buffer."
        )
    else:
        st.info(
            "These recommendations are conservative and designed to balance trip readiness with long-term battery health."
        )


def main() -> None:
    st.set_page_config(
        page_title="EV Charging Recommendation Assistant",
        layout="wide",
        page_icon=None,
    )

    # Load model and feature engineer
    model, feature_engineer, feature_columns, training_data = load_trained_model_and_features()

    st.title("EV Charging Recommendation Assistant")
    st.subheader(
        "How much should you charge today to comfortably cover tomorrow's trip while preserving battery health?"
    )

    if model is not None:
        st.markdown(
            """
This tool provides **charging recommendations powered by an ML model trained on EV charging patterns**.  
It combines machine learning cost predictions with rule-based charging strategies for optimal results.
            """
        )
    else:
        st.markdown(
            """
This tool provides **rule-based charging recommendations** using simple energy and battery-health heuristics.  
*Note: The trained ML model is not available. Train it by running main.py first.*
            """
        )

    recommendation_tab, experiment_tab = st.tabs(
        ["Charging Recommendation", "Experiment Results"]
    )

    with recommendation_tab:
        render_recommendation_tab(
            model=model,
            feature_engineer=feature_engineer,
            feature_columns=feature_columns,
            training_data=training_data,
        )

    with experiment_tab:
        model_name = type(model).__name__ if model is not None else "Model not loaded"
        render_results_tab(model_name=model_name)


if __name__ == "__main__":
    main()
