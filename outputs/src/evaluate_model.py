"""
Model evaluation module for EV charging optimization.

Computes:
- RMSE, MAE, R² score
- Feature importance analysis
- Prediction visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """Class for evaluating trained models."""
    
    def __init__(self):
        self.metrics = {}
        
    def compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                       set_name: str = "test") -> dict:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            set_name: Name of the dataset
            
        Returns:
            Dictionary of metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        median_ae = np.median(np.abs(y_true - y_pred))
        
        metrics = {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'Median_AE': float(median_ae)
        }
        
        self.metrics[set_name] = metrics
        
        print(f"\n{set_name.upper()} Set Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Median Absolute Error: {median_ae:.4f}")
        
        return metrics
    
    def plot_predictions_vs_actual(self, y_true: pd.Series, y_pred: np.ndarray,
                                   set_name: str = "test", save_path: Optional[str] = None):
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            set_name: Name of the dataset
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Charging Cost (USD)', fontsize=12)
        axes[0].set_ylabel('Predicted Charging Cost (USD)', fontsize=12)
        axes[0].set_title(f'Predicted vs Actual - {set_name.title()} Set', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Charging Cost (USD)', fontsize=12)
        axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        axes[1].set_title(f'Residual Plot - {set_name.title()} Set', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Always save as PNG
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            print(f"\n✓ Plot saved: {save_path}")
        else:
            # Generate default save path if not provided
            from pathlib import Path
            default_path = Path("outputs/figures/predictions_vs_actual.png")
            default_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(default_path, dpi=300, bbox_inches='tight', format='png')
            print(f"\n✓ Plot saved: {default_path}")
        
        # Print explanation
        print("\n" + "="*70)
        print("PLOT EXPLANATION: Predictions vs Actual")
        print("="*70)
        print("Left Panel - Scatter Plot:")
        print("  • Shows how well model predictions match actual charging costs")
        print("  • Each point represents one charging session")
        print("  • Red dashed line = perfect predictions (actual = predicted)")
        print("  • Points close to the line = accurate predictions")
        print("  • Points far from the line = prediction errors")
        print("\nRight Panel - Residual Plot:")
        print("  • Shows prediction errors (residuals = actual - predicted)")
        print("  • Red dashed line at y=0 = no error")
        print("  • Points above zero = model underestimates cost")
        print("  • Points below zero = model overestimates cost")
        print("  • Random scatter around zero = good model")
        print("  • Patterns/curves = model bias (needs improvement)")
        print("="*70)
        
        plt.close()  # Close figure to free memory
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            save_path: Path to save the figure
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature Name', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features for Charging Cost Prediction', fontsize=14)
        plt.tight_layout()
        
        # Always save as PNG
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            print(f"\n✓ Plot saved: {save_path}")
        else:
            # Generate default save path if not provided
            from pathlib import Path
            default_path = Path("outputs/figures/feature_importance.png")
            default_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(default_path, dpi=300, bbox_inches='tight', format='png')
            print(f"\n✓ Plot saved: {default_path}")
        
        # Print explanation
        print("\n" + "="*70)
        print("PLOT EXPLANATION: Feature Importance")
        print("="*70)
        print("This plot shows which features are most important for predicting charging costs.")
        print("\nInterpretation:")
        print("  • Higher importance = feature has more influence on predictions")
        print("  • Top features drive the model's decision-making")
        print("  • Features at the top are the strongest predictors")
        print("\nTop 5 Features:")
        for idx, row in top_features.head(5).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        print("\nInsights:")
        print("  • Focus optimization efforts on high-importance features")
        print("  • Low-importance features may be candidates for removal")
        print("  • Helps understand what factors most affect charging costs")
        print("="*70)
        
        plt.close()  # Close figure to free memory
    
    def plot_error_distribution(self, y_true: pd.Series, y_pred: np.ndarray,
                               set_name: str = "test", save_path: Optional[str] = None):
        """
        Plot distribution of prediction errors.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            set_name: Name of the dataset
            save_path: Path to save the figure
        """
        errors = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        axes[0].axvline(x=errors.mean(), color='g', linestyle='--', lw=2, label=f'Mean Error: ${errors.mean():.2f}')
        axes[0].set_xlabel('Prediction Error (USD)', fontsize=12)
        axes[0].set_ylabel('Frequency (Number of Sessions)', fontsize=12)
        axes[0].set_title(f'Error Distribution Histogram - {set_name.title()} Set', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        bp = axes[1].boxplot(errors, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
        axes[1].set_ylabel('Prediction Error (USD)', fontsize=12)
        axes[1].set_title(f'Error Box Plot - {set_name.title()} Set', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Always save as PNG
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            print(f"\n✓ Plot saved: {save_path}")
        else:
            # Generate default save path if not provided
            from pathlib import Path
            default_path = Path("outputs/figures/error_distribution.png")
            default_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(default_path, dpi=300, bbox_inches='tight', format='png')
            print(f"\n✓ Plot saved: {default_path}")
        
        # Print explanation
        print("\n" + "="*70)
        print("PLOT EXPLANATION: Error Distribution")
        print("="*70)
        print("This plot shows how prediction errors are distributed across test sessions.")
        print("\nLeft Panel - Histogram:")
        print("  • Shows the frequency distribution of prediction errors")
        print("  • X-axis: Error amount in USD (positive = overestimate, negative = underestimate)")
        print("  • Y-axis: Number of sessions with that error")
        print("  • Red line at 0 = perfect prediction")
        print("  • Green line = mean error (bias)")
        print("  • Narrow, centered distribution = accurate model")
        print("  • Wide distribution = high variance in predictions")
        print("\nRight Panel - Box Plot:")
        print("  • Shows error distribution summary statistics")
        print("  • Box = interquartile range (middle 50% of errors)")
        print("  • Line in box = median error")
        print("  • Whiskers = range of typical errors")
        print("  • Dots = outliers (unusual prediction errors)")
        print("\nKey Statistics:")
        print(f"  • Mean Error: ${errors.mean():.2f} ({'overestimate' if errors.mean() > 0 else 'underestimate' if errors.mean() < 0 else 'unbiased'})")
        print(f"  • Median Error: ${errors.median():.2f}")
        print(f"  • Std Deviation: ${errors.std():.2f}")
        print(f"  • Min Error: ${errors.min():.2f}")
        print(f"  • Max Error: ${errors.max():.2f}")
        print("="*70)
        
        plt.close()  # Close figure to free memory
    
    def generate_evaluation_report(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                  importance_df: pd.DataFrame, save_dir: Optional[str] = None):
        """
        Generate comprehensive evaluation report with all plots and metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            importance_df: DataFrame with feature importance
            save_dir: Directory to save plots and metrics
        """
        print("=" * 60)
        print("Model Evaluation Report")
        print("=" * 60)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred, set_name="test")
        
        # Generate plots
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            self.plot_predictions_vs_actual(
                y_test, y_pred, 
                save_path=os.path.join(save_dir, 'predictions_vs_actual.png')
            )
            
            self.plot_feature_importance(
                importance_df,
                save_path=os.path.join(save_dir, 'feature_importance.png')
            )
            
            self.plot_error_distribution(
                y_test, y_pred,
                save_path=os.path.join(save_dir, 'error_distribution.png')
            )
        else:
            self.plot_predictions_vs_actual(y_test, y_pred)
            self.plot_feature_importance(importance_df)
            self.plot_error_distribution(y_test, y_pred)
        
        return metrics
