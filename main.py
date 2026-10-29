"""
Main entry point for EV Charging Optimization project.

This script orchestrates the complete ML pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training
4. Model evaluation
5. Optimization recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the directory that contains the `src` package to the import path.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "outputs"))

from src.utils import (
    load_data, save_processed_data, save_model, load_model,
    save_metrics, save_figure, get_project_root
)
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer, create_user_features
from src.train_model import ModelTrainer
from src.health_metrics import compute_health_metrics, apply_charger_type
from src.optimization import ChargingOptimizer


def main():
    """Main execution function."""
    print("=" * 60)
    print("EV Charging Pattern Optimization - ML Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    try:
        df = load_data("ev_charging_data.csv")
        print(f"Loaded {len(df)} charging sessions")
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure ev_charging_data.csv is in data/raw/ directory")
        return
    
    # Step 2: Preprocess data
    print("\n[Step 2] Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df, treat_outliers=True, outlier_method='cap')
    
    # Save processed data
    save_processed_data(df_processed, "processed_data.csv")
    
    # Step 3: Time-based train-test split
    print("\n[Step 3] Splitting data...")
    train_df, test_df = preprocessor.time_based_split(df_processed, test_size=0.2)
    
    # Step 3.5: Create SOC features first (needed for user features)
    print("\n[Step 3.5] Creating SOC features for user aggregation...")
    feature_engineer_temp = FeatureEngineer()
    train_df_with_soc = feature_engineer_temp.create_soc_features(train_df.copy())
    
    # Step 3.6: Create user-level aggregated features from TRAIN DATA ONLY
    print("\n[Step 3.6] Creating user-level aggregated features...")
    user_features = create_user_features(train_df_with_soc)
    
    # Get actual user feature columns from the created features
    user_feature_cols = [col for col in user_features.columns if col != 'User ID']
    
    # Merge user features into train and test datasets
    print("Merging user features into train and test datasets...")
    train_df = train_df.merge(user_features, on="User ID", how="left")
    test_df = test_df.merge(user_features, on="User ID", how="left")
    
    # Handle missing user features (for unseen users in test data)
    print("Handling missing user features for unseen users...")
    # Only fill columns that exist
    for col in user_feature_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(0)
    
    print(f"User features created: {user_feature_cols}")
    
    print(f"User features merged. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Step 4: Feature engineering
    print("\n[Step 4] Engineering features...")
    feature_engineer = FeatureEngineer()
    
    # Engineer features for training set (user features already exist)
    train_df_features = feature_engineer.engineer_features(
        train_df.copy(), 
        train_df=None,  # For training, compute stats from train_df itself
        encode_categorical=True,
        original_train_df=train_df.copy()  # Store original for encoding
    )
    
    # Engineer features for test set (user features already exist)
    test_df_features = feature_engineer.engineer_features(
        test_df.copy(),
        train_df=train_df_features,  # Use feature-engineered training data for reference
        encode_categorical=True,
        original_train_df=train_df.copy()  # Use original training data for encoding
    )
    
    # Get feature columns
    feature_cols = feature_engineer.get_feature_columns(
        train_df_features, 
        target_col='Charging Cost (USD)'
    )
    
    # Safety check: Verify all feature columns exist in both datasets
    print("\n[Step 4.5] Validating feature columns...")
    missing_in_train = set(feature_cols) - set(train_df_features.columns)
    missing_in_test = set(feature_cols) - set(test_df_features.columns)
    
    if missing_in_train:
        raise ValueError(f"Missing feature columns in training set: {missing_in_train}")
    if missing_in_test:
        raise ValueError(f"Missing feature columns in test set: {missing_in_test}")
    
    print(f"✓ All {len(feature_cols)} feature columns present in both datasets")
    
    # Prepare features and target
    X_train = train_df_features[feature_cols].fillna(0)
    y_train = train_df_features['Charging Cost (USD)']
    
    X_test = test_df_features[feature_cols].fillna(0)
    y_test = test_df_features['Charging Cost (USD)']
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 5: Train models
    print("\n[Step 5] Training models...")
    trainer = ModelTrainer()
    models = trainer.train_all_models(
        X_train, y_train,
        X_val=None,  # Using test set for final evaluation
        y_val=None,
        tune_hyperparameters=False  # Set to True for hyperparameter tuning
    )
    
    # Step 6: Evaluate models using battery-health metrics
    print("\n[Step 6] Evaluating models (battery-health metrics)...")

    best_model = trainer.best_model

    # Health metrics for baseline (no optimization)
    baseline_metrics = compute_health_metrics(test_df)

    model_comparison = []

    # Evaluate "Our Model" using the best model with optimization
    optimizer = ChargingOptimizer(best_model, feature_cols, feature_engineer)
    optimized_rows = []
    for idx in range(len(test_df_features)):
        session_features = test_df_features.iloc[[idx]].copy()
        session_raw = test_df.iloc[[idx]].copy()

        try:
            charger_opt = optimizer.optimize_charger_type(session_features)
            optimal_charger = charger_opt.get("optimal_charger") or session_raw["Charger Type"].iloc[0]
        except Exception:
            optimal_charger = session_raw["Charger Type"].iloc[0]

        session_raw = apply_charger_type(session_raw, optimal_charger)
        optimized_rows.append(session_raw)

    optimized_df = pd.concat(optimized_rows, ignore_index=True)
    our_health_metrics = compute_health_metrics(optimized_df)
    model_comparison.append(
        {
            "model": "Our Model",
            "optimized": True,
            **our_health_metrics,
        }
    )

    # Evaluate each trained baseline model by applying its optimization recommendations
    name_map = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    for model_name, model in trainer.models.items():
        if model is best_model:
            continue
        optimizer = ChargingOptimizer(model, feature_cols, feature_engineer)
        optimized_rows = []

        for idx in range(len(test_df_features)):
            session_features = test_df_features.iloc[[idx]].copy()
            session_raw = test_df.iloc[[idx]].copy()

            try:
                charger_opt = optimizer.optimize_charger_type(session_features)
                optimal_charger = charger_opt.get("optimal_charger") or session_raw["Charger Type"].iloc[0]
            except Exception:
                optimal_charger = session_raw["Charger Type"].iloc[0]

            session_raw = apply_charger_type(session_raw, optimal_charger)
            optimized_rows.append(session_raw)

        optimized_df = pd.concat(optimized_rows, ignore_index=True)
        health_metrics = compute_health_metrics(optimized_df)

        model_comparison.append(
            {
                "model": name_map.get(model_name, model_name.replace("_", " ").title()),
                "optimized": True,
                **health_metrics,
            }
        )

    # Add baseline to comparison
    model_comparison.append(
        {
            "model": "No Optimization Baseline",
            "optimized": False,
            **baseline_metrics,
        }
    )

    # Save comparison metrics
    comparison_payload = {"models": model_comparison}
    save_metrics(comparison_payload, "model_comparison_metrics.json")

    # Save "test_metrics.json" for UI (use our best model metrics)
    our_model_metrics = next(
        (m for m in model_comparison if m["model"] == "Our Model"), model_comparison[0]
    )
    save_metrics(our_model_metrics, "test_metrics.json")
    
    # Step 7: Save model
    print("\n[Step 7] Saving model...")
    save_model(best_model, "trained_model.pkl")
    
    # Save feature columns for use in app
    import pickle
    models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "feature_columns.pkl", 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Feature columns saved to {models_dir / 'feature_columns.pkl'}")
    
    # Step 8: Optimization examples
    print("\n[Step 8] Generating optimization recommendations...")
    optimizer = ChargingOptimizer(best_model, feature_cols, feature_engineer)
    
    # Example: Optimize a sample session
    sample_session = test_df_features.iloc[[0]].copy()
    recommendations = optimizer.generate_recommendations(sample_session)
    
    print("\nSample Optimization Recommendations:")
    print(f"Current Predicted Cost: ${recommendations['current_predicted_cost']:.2f}")
    print(f"Optimal Predicted Cost: ${recommendations['optimal_predicted_cost']:.2f}")
    print(f"Potential Savings: ${recommendations['potential_savings']:.2f} ({recommendations['savings_percentage']:.1f}%)")
    
    if recommendations['time_optimization']['optimal_hour'] is not None:
        print(f"\nOptimal Charging Hour: {recommendations['time_optimization']['optimal_hour']}:00")
    
    if recommendations['charger_optimization']['optimal_charger'] is not None:
        print(f"Optimal Charger Type: {recommendations['charger_optimization']['optimal_charger']}")
    
    # Identify inefficient sessions
    print("\nIdentifying inefficient charging sessions...")
    inefficient_sessions = optimizer.identify_inefficient_sessions(test_df_features)
    
    print("\n" + "=" * 60)
    print("Pipeline execution complete!")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  - Model: models/trained_model.pkl")
    print(f"  - Metrics: outputs/metrics.json")
    print(f"  - Figures: outputs/figures/")
    print(f"  - Processed data: data/processed/processed_data.csv")


if __name__ == "__main__":
    main()
