"""
Optimization module for EV charging recommendations.

Uses trained model to:
- Simulate alternative charging scenarios
- Recommend optimal charging time and charger type
- Identify inefficient charging behavior
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ChargingOptimizer:
    """Class for optimizing EV charging patterns using trained model."""
    
    def __init__(self, model, feature_columns: List[str], feature_engineer):
        """
        Initialize optimizer with trained model.
        
        Args:
            model: Trained regression model
            feature_columns: List of feature column names
            feature_engineer: FeatureEngineer instance for feature creation
        """
        self.model = model
        self.feature_columns = feature_columns
        self.feature_engineer = feature_engineer
        
    def predict_cost(self, session_data: pd.DataFrame) -> np.ndarray:
        """
        Predict charging cost for given session data.
        
        Args:
            session_data: DataFrame with session features
            
        Returns:
            Array of predicted costs
        """
        # Ensure all required features are present
        session_features = self._prepare_features(session_data)
        return self.model.predict(session_features)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with features ready for model
        """
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                # Fill missing columns with median/default values
                if 'target_encoded' in col or 'avg' in col:
                    df[col] = 0.0  # Default for encoded/aggregated features
                else:
                    df[col] = 0.0
        
        return df[self.feature_columns]
    
    def optimize_charging_time(self, session_data: pd.DataFrame,
                              hour_range: Tuple[int, int] = (0, 23)) -> Dict:
        """
        Find optimal charging time for a given session.
        
        Args:
            session_data: Single session DataFrame (one row)
            hour_range: Range of hours to test (start, end)
            
        Returns:
            Dictionary with optimal time and predicted cost
        """
        if len(session_data) != 1:
            raise ValueError("session_data must contain exactly one row")
        
        session = session_data.iloc[0].copy()
        best_hour = None
        best_cost = float('inf')
        results = []
        
        # Test different hours
        for hour in range(hour_range[0], hour_range[1] + 1):
            test_session = session.copy()
            test_session['hour'] = hour
            
            # Update derived features
            test_session['is_peak_hours'] = int(
                ((hour >= 6) & (hour < 10)) | ((hour >= 17) & (hour < 21))
            )
            test_session['is_off_peak'] = int((hour >= 22) | (hour < 6))
            test_session['is_weekend'] = session.get('is_weekend', 0)
            
            # Create DataFrame for prediction
            test_df = pd.DataFrame([test_session])
            
            # Engineer features
            test_df = self.feature_engineer.create_time_features(test_df)
            
            # Predict cost
            try:
                cost = self.predict_cost(test_df)[0]
                results.append({'hour': hour, 'predicted_cost': cost})
                
                if cost < best_cost:
                    best_cost = cost
                    best_hour = hour
            except Exception as e:
                print(f"Error predicting for hour {hour}: {e}")
                continue
        
        return {
            'optimal_hour': best_hour,
            'optimal_cost': best_cost,
            'current_hour': session.get('hour', None),
            'current_cost': self.predict_cost(session_data)[0] if 'hour' in session_data.columns else None,
            'all_hours': results
        }
    
    def optimize_charger_type(self, session_data: pd.DataFrame,
                              charger_types: Optional[List[str]] = None) -> Dict:
        """
        Find optimal charger type for a given session.
        
        Args:
            session_data: Single session DataFrame (one row)
            charger_types: List of charger types to test. If None, uses unique values from data.
            
        Returns:
            Dictionary with optimal charger type and predicted cost
        """
        if len(session_data) != 1:
            raise ValueError("session_data must contain exactly one row")
        
        session = session_data.iloc[0].copy()
        current_charger = session.get('Charger Type', None)
        
        if charger_types is None:
            # Get unique charger types from original data
            charger_types = ['Level 1', 'Level 2', 'DC Fast']  # Common types
        
        best_charger = None
        best_cost = float('inf')
        results = []
        
        # Test different charger types
        for charger_type in charger_types:
            test_session = session.copy()
            test_session['Charger Type'] = charger_type
            
            # Update charging rate based on charger type (simplified)
            charger_rate_map = {
                'Level 1': 1.4,  # kW
                'Level 2': 7.2,  # kW
                'DC Fast': 50.0  # kW
            }
            
            if charger_type in charger_rate_map:
                test_session['Charging Rate (kW)'] = charger_rate_map[charger_type]
            
            # Create DataFrame for prediction
            test_df = pd.DataFrame([test_session])
            
            # Engineer features
            test_df = self.feature_engineer.create_efficiency_features(test_df)
            test_df = self.feature_engineer.create_interaction_features(test_df)
            
            # Predict cost
            try:
                cost = self.predict_cost(test_df)[0]
                results.append({
                    'charger_type': charger_type,
                    'predicted_cost': cost,
                    'charging_rate': charger_rate_map.get(charger_type, None)
                })
                
                if cost < best_cost:
                    best_cost = cost
                    best_charger = charger_type
            except Exception as e:
                print(f"Error predicting for charger {charger_type}: {e}")
                continue
        
        return {
            'optimal_charger': best_charger,
            'optimal_cost': best_cost,
            'current_charger': current_charger,
            'current_cost': self.predict_cost(session_data)[0],
            'all_chargers': results
        }
    
    def identify_inefficient_sessions(self, sessions_df: pd.DataFrame,
                                     cost_threshold_percentile: float = 75) -> pd.DataFrame:
        """
        Identify sessions with inefficient charging patterns.
        
        Args:
            sessions_df: DataFrame with multiple sessions
            cost_threshold_percentile: Percentile threshold for identifying high-cost sessions
            
        Returns:
            DataFrame with inefficient sessions flagged
        """
        # Predict costs for all sessions
        predicted_costs = self.predict_cost(sessions_df)
        
        # Calculate threshold
        threshold = np.percentile(predicted_costs, cost_threshold_percentile)
        
        # Flag inefficient sessions
        sessions_df = sessions_df.copy()
        sessions_df['predicted_cost'] = predicted_costs
        sessions_df['is_inefficient'] = predicted_costs > threshold
        
        # Calculate potential savings
        median_cost = np.median(predicted_costs)
        sessions_df['potential_savings'] = np.maximum(
            0, predicted_costs - median_cost
        )
        
        inefficient = sessions_df[sessions_df['is_inefficient']].copy()
        
        print(f"Identified {len(inefficient)} inefficient sessions ({len(inefficient)/len(sessions_df)*100:.1f}%)")
        print(f"Average potential savings: ${inefficient['potential_savings'].mean():.2f}")
        
        return inefficient.sort_values('potential_savings', ascending=False)
    
    def generate_recommendations(self, session_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive charging recommendations for a session.
        
        Args:
            session_data: Single session DataFrame (one row)
            
        Returns:
            Dictionary with recommendations
        """
        if len(session_data) != 1:
            raise ValueError("session_data must contain exactly one row")
        
        recommendations = {}
        
        # Optimize charging time
        time_optimization = self.optimize_charging_time(session_data)
        recommendations['time_optimization'] = time_optimization
        
        # Optimize charger type
        charger_optimization = self.optimize_charger_type(session_data)
        recommendations['charger_optimization'] = charger_optimization
        
        # Calculate potential savings
        current_cost = self.predict_cost(session_data)[0]
        optimal_cost = min(
            time_optimization.get('optimal_cost', current_cost),
            charger_optimization.get('optimal_cost', current_cost)
        )
        
        recommendations['current_predicted_cost'] = current_cost
        recommendations['optimal_predicted_cost'] = optimal_cost
        recommendations['potential_savings'] = current_cost - optimal_cost
        recommendations['savings_percentage'] = (
            (current_cost - optimal_cost) / current_cost * 100
            if current_cost > 0 else 0
        )
        
        return recommendations
    
    def simulate_scenarios(self, base_session: pd.DataFrame,
                          scenarios: List[Dict]) -> pd.DataFrame:
        """
        Simulate multiple charging scenarios.
        
        Args:
            base_session: Base session DataFrame (one row)
            scenarios: List of dictionaries with scenario modifications
                       e.g., [{'hour': 2, 'Charger Type': 'Level 2'}, ...]
        
        Returns:
            DataFrame with scenario results
        """
        if len(base_session) != 1:
            raise ValueError("base_session must contain exactly one row")
        
        base = base_session.iloc[0].copy()
        results = []
        
        for i, scenario in enumerate(scenarios):
            test_session = base.copy()
            
            # Apply scenario modifications
            for key, value in scenario.items():
                test_session[key] = value
            
            # Create DataFrame
            test_df = pd.DataFrame([test_session])
            
            # Engineer features
            test_df = self.feature_engineer.create_time_features(test_df)
            test_df = self.feature_engineer.create_efficiency_features(test_df)
            
            # Predict cost
            try:
                cost = self.predict_cost(test_df)[0]
                results.append({
                    'scenario_id': i + 1,
                    'predicted_cost': cost,
                    **scenario
                })
            except Exception as e:
                print(f"Error in scenario {i+1}: {e}")
                continue
        
        return pd.DataFrame(results).sort_values('predicted_cost')
