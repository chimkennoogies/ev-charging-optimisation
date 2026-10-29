"""
Model training module for EV charging optimization.

Trains:
- Random Forest Regressor
- XGBoost Regressor
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Class for training regression models."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                            n_estimators: int = 100, max_depth: int = 20,
                            min_samples_split: int = 5, random_state: int = 42,
                            n_jobs: int = -1) -> RandomForestRegressor:
        """
        Train a Random Forest Regressor.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            random_state: Random seed
            n_jobs: Number of parallel jobs
            
        Returns:
            Trained Random Forest model
        """
        print("Training Random Forest Regressor...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        print("Random Forest training complete.")
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     n_estimators: int = 100, max_depth: int = 6,
                     learning_rate: float = 0.1, random_state: int = 42,
                     n_jobs: int = -1) -> XGBRegressor:
        """
        Train an XGBoost Regressor.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            random_state: Random seed
            n_jobs: Number of parallel jobs
            
        Returns:
            Trained XGBoost model
        """
        print("Training XGBoost Regressor...")
        
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        print("XGBoost training complete.")
        return model
    
    def hyperparameter_tuning(self, model_type: str, X_train: pd.DataFrame,
                              y_train: pd.Series, cv: int = 5) -> dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_type: 'random_forest' or 'xgboost'
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters
        """
        print(f"Performing hyperparameter tuning for {model_type}...")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            base_model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[model_type] = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series,
                      set_name: str = "dataset") -> dict:
        """
        Evaluate a model and return metrics.
        
        Args:
            model: Trained model
            X: Features
            y: True target values
            set_name: Name of the dataset (for printing)
            
        Returns:
          p  Dictionary of evaluation metrics
        """
        y_pred = model.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"\n{set_name} Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return metrics
    
    def select_best_model(self, X_val: pd.DataFrame, y_val: pd.Series) -> str:
        """
        Select the best model based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Name of the best model
        """
        print("\nComparing models on validation set...")
        
        best_score = float('inf')
        best_name = None
        
        for name, model in self.models.items():
            metrics = self.evaluate_model(model, X_val, y_val, set_name=name)
            rmse = metrics['RMSE']
            
            if rmse < best_score:
                best_score = rmse
                best_name = name
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {best_name} (RMSE: {best_score:.4f})")
        return best_name
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame = None, y_val: pd.Series = None,
                         tune_hyperparameters: bool = False) -> dict:
        """
        Train all models and select the best one.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with all trained models
        """
        print("=" * 60)
        print("Training Models")
        print("=" * 60)
        
        # Train Random Forest
        if tune_hyperparameters:
            self.hyperparameter_tuning('random_forest', X_train, y_train)
        else:
            self.train_random_forest(X_train, y_train)
        
        # Train XGBoost
        if tune_hyperparameters:
            self.hyperparameter_tuning('xgboost', X_train, y_train)
        else:
            self.train_xgboost(X_train, y_train)
        
        # Select best model if validation set provided
        if X_val is not None and y_val is not None:
            self.select_best_model(X_val, y_val)
        else:
            # Default to XGBoost if no validation set
            self.best_model = self.models['xgboost']
            self.best_model_name = 'xgboost'
            print("\nNo validation set provided. Using XGBoost as default.")
        
        return self.models
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the best model or specified model.
        
        Args:
            model_name: Name of model ('random_forest' or 'xgboost'). If None, uses best model.
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models[model_name]
        
        if model is None:
            raise ValueError("No model available. Train a model first.")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            raise ValueError(f"Model {model_name} does not support feature importance.")
        
        # Create DataFrame
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
