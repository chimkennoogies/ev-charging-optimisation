#!/usr/bin/env python3
"""Test ML prediction function"""
import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "outputs"))

from src.utils import load_model, get_project_root
from src.feature_engineering import FeatureEngineer
from datetime import datetime
import pickle

# Load model and feature engineer
model = load_model('trained_model.pkl')
fe = FeatureEngineer()

# Load feature columns
models_dir = get_project_root() / 'models'
feature_cols_path = models_dir / 'feature_columns.pkl'
with open(feature_cols_path, 'rb') as f:
    feature_columns = pickle.load(f)

# Load processed training data
training_data = pd.read_csv(get_project_root() / "data" / "processed" / "processed_data.csv")
if len(training_data) > 500:
    training_data = training_data.sample(n=500, random_state=42)

print(f"Testing ML prediction with {len(feature_columns)} features...\n")
print(f"Training data shape: {training_data.shape}\n")

# Create test input
now = datetime.now()
distance_km = 50
temperature_c = 20
current_soc = 40
target_soc = 60
battery_capacity_kwh = 60
charger_type = "Level 2 (AC, ~7.2 kW)"
vehicle_model = "Tesla Model 3"
energy_to_charge = (target_soc - current_soc) * battery_capacity_kwh / 100
estimated_charging_hours = energy_to_charge / 7.2

session_data = pd.DataFrame({
    'Distance Driven (since last charge) (km)': [distance_km],
    'Temperature (°C)': [temperature_c],
    'State of Charge (Start %)': [current_soc],
    'State of Charge (End %)': [min(target_soc, 100)],
    'Battery Capacity (kWh)': [battery_capacity_kwh],
    'Energy Consumed (kWh)': [energy_to_charge],
    'Charging Rate (kW)': [energy_to_charge / max(estimated_charging_hours, 0.1)],
    'Charging Duration (hours)': [estimated_charging_hours],
    'Vehicle Model': [vehicle_model],
    'Charger Type': [charger_type],
    'Vehicle Age (years)': [5.0],
    'User Type': ['Commuter'],
    'Charging Station Location': ['New York'],
    'Time of Day': ['Evening'],
    'User ID': ['User_999'],  # String User ID to match training data
    'Charging Station ID': ['Station_999'],  # String ID to match training data
    'day_name': [now.strftime('%A')],
    'month': [now.month],
    'day_of_month': [now.day],
    'hour': [now.hour],
    'Charging Cost (USD)': [0.0],
})

print("Input DataFrame shape:", session_data.shape)

# Apply feature engineering with training data
session_features = fe.engineer_features(
    session_data.copy(),
    train_df=training_data,
    encode_categorical=True,
    original_train_df=training_data.copy()
)

print("After feature engineering shape:", session_features.shape)

# Prepare for prediction
for feat in feature_columns:
    if feat not in session_features.columns:
        session_features[feat] = 0.0

X = session_features[feature_columns].fillna(0)

print("Final feature matrix shape:", X.shape)
print("Expected features:", len(feature_columns))
print("Features match:", X.shape[1] == len(feature_columns))

# Make prediction
try:
    predicted_cost = model.predict(X)[0]
    print(f"\n✓ Prediction successful!")
    print(f"  Predicted charging cost: ${predicted_cost:.2f}")
except Exception as e:
    print(f"\n✗ Prediction failed: {e}")
