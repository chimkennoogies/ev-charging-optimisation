"""
Feature engineering module for EV charging optimization.

Creates:
- SOC-based features
- Efficiency metrics
- Cost per kWh
- Time-based features
- User-level aggregated features
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


def create_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates historical user-level aggregated features from training data.
    
    This function should be called ONLY on training data to prevent data leakage.
    The resulting features are then merged into both train and test sets.
    
    Args:
        df: Training DataFrame (must have User ID and required columns)
        
    Returns:
        DataFrame with User ID and user-level aggregated features:
        - user_avg_cost: Average charging cost per user
        - user_cost_std: Standard deviation of charging cost
        - user_charging_frequency: Number of charging sessions per user
        - user_avg_energy: Average energy consumed per user
        - user_energy_std: Standard deviation of energy consumed
        - user_avg_duration: Average charging duration per user
        - user_duration_std: Standard deviation of charging duration
        - user_avg_soc_gain: Average SOC gain per user (if available)
    """
    df = df.copy()
    
    # Required columns for user feature creation
    required_cols = [
        'User ID',
        'Charging Cost (USD)',
        'Energy Consumed (kWh)',
        'Charging Duration (hours)'
    ]
    
    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for user features: {missing_cols}")
    
    # Build aggregation dictionary
    agg_dict = {
        'Charging Cost (USD)': ['mean', 'std', 'count'],
        'Energy Consumed (kWh)': ['mean', 'std'],
        'Charging Duration (hours)': ['mean', 'std']
    }
    
    # Add soc_gain if it exists (created during feature engineering)
    if 'soc_gain' in df.columns:
        agg_dict['soc_gain'] = ['mean']
    elif 'State of Charge (Start %)' in df.columns and 'State of Charge (End %)' in df.columns:
        # Calculate soc_gain if not already present
        df['soc_gain'] = df['State of Charge (End %)'] - df['State of Charge (Start %)']
        agg_dict['soc_gain'] = ['mean']
    # If soc_gain is not available, we'll just skip it (user_avg_soc_gain will be missing)
    
    # Compute user-level statistics
    user_features = df.groupby('User ID').agg(agg_dict).reset_index()
    
    # Flatten column names
    user_features.columns = ['User ID'] + [
        '_'.join(col).strip() if isinstance(col, tuple) and len(col) > 1 and col[1] else str(col)
        for col in user_features.columns[1:]
    ]
    
    # Rename columns to match expected names
    rename_dict = {
        'Charging Cost (USD)_mean': 'user_avg_cost',
        'Charging Cost (USD)_std': 'user_cost_std',
        'Charging Cost (USD)_count': 'user_charging_frequency',
        'Energy Consumed (kWh)_mean': 'user_avg_energy',
        'Energy Consumed (kWh)_std': 'user_energy_std',
        'Charging Duration (hours)_mean': 'user_avg_duration',
        'Charging Duration (hours)_std': 'user_duration_std',
    }
    
    if 'soc_gain_mean' in user_features.columns:
        rename_dict['soc_gain_mean'] = 'user_avg_soc_gain'
    
    user_features = user_features.rename(columns=rename_dict)
    
    # Fill missing values (shouldn't happen, but safety check)
    user_feature_cols = [col for col in user_features.columns if col != 'User ID']
    for col in user_feature_cols:
        if user_features[col].isnull().any():
            # Fill with median, or 0 if all are NaN
            fill_value = user_features[col].median() if not user_features[col].isnull().all() else 0
            user_features[col] = user_features[col].fillna(fill_value)
    
    return user_features


class FeatureEngineer:
    """Class for engineering features from EV charging data."""
    
    def __init__(self):
        self.user_stats = None  # Will store user-level statistics for encoding
        self.original_train_df = None  # Store original training data for encoding
        
    def create_soc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create State of Charge (SOC) related features.
        
        Features:
        - SOC gain (End SOC - Start SOC)
        - SOC gain percentage
        - Initial SOC level category
        - Target SOC level category
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with SOC features added
        """
        df = df.copy()
        
        # SOC gain
        if 'State of Charge (Start %)' in df.columns and 'State of Charge (End %)' in df.columns:
            df['soc_gain'] = df['State of Charge (End %)'] - df['State of Charge (Start %)']
            df['soc_gain_pct'] = df['soc_gain'] / (100 - df['State of Charge (Start %)'] + 1e-6)
            
            # SOC level categories
            df['initial_soc_level'] = pd.cut(
                df['State of Charge (Start %)'],
                bins=[0, 20, 50, 80, 100],
                labels=['Very Low', 'Low', 'Medium', 'High']
            )
            
            df['target_soc_level'] = pd.cut(
                df['State of Charge (End %)'],
                bins=[0, 50, 80, 90, 100],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        return df
    
    def create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create charging efficiency metrics.
        
        Features:
        - Energy per SOC gain
        - Charging rate efficiency
        - Battery utilization
        - Theoretical vs actual charging time
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with efficiency features added
        """
        df = df.copy()
        
        # Energy per SOC gain
        if 'soc_gain' in df.columns and 'Energy Consumed (kWh)' in df.columns:
            df['energy_per_soc_gain'] = df['Energy Consumed (kWh)'] / (df['soc_gain'] + 1e-6)
        
        # Charging rate efficiency (actual vs theoretical)
        if 'Charging Rate (kW)' in df.columns and 'Battery Capacity (kWh)' in df.columns:
            # Theoretical max rate based on battery capacity
            df['theoretical_max_rate'] = df['Battery Capacity (kWh)'] * 0.5  # Simplified assumption
            df['rate_efficiency'] = df['Charging Rate (kW)'] / (df['theoretical_max_rate'] + 1e-6)
        
        # Battery utilization
        if 'Battery Capacity (kWh)' in df.columns and 'Energy Consumed (kWh)' in df.columns:
            df['battery_utilization'] = df['Energy Consumed (kWh)'] / (df['Battery Capacity (kWh)'] + 1e-6)
        
        # Charging duration efficiency
        if 'Charging Duration (hours)' in df.columns and 'soc_gain' in df.columns:
            df['soc_gain_per_hour'] = df['soc_gain'] / (df['Charging Duration (hours)'] + 1e-6)
        
        return df
    
    def create_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cost-related features.
        
        Features:
        - Cost per kWh
        - Cost per SOC gain
        - Cost per hour
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cost features added
        """
        df = df.copy()
        
        if 'Charging Cost (USD)' in df.columns:
            # Cost per kWh
            if 'Energy Consumed (kWh)' in df.columns:
                df['cost_per_kwh'] = df['Charging Cost (USD)'] / (df['Energy Consumed (kWh)'] + 1e-6)
            
            # Cost per SOC gain
            if 'soc_gain' in df.columns:
                df['cost_per_soc_gain'] = df['Charging Cost (USD)'] / (df['soc_gain'] + 1e-6)
            
            # Cost per hour
            if 'Charging Duration (hours)' in df.columns:
                df['cost_per_hour'] = df['Charging Cost (USD)'] / (df['Charging Duration (hours)'] + 1e-6)
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Features:
        - Hour of day
        - Day of week (numeric and categorical)
        - Is weekend
        - Is peak hours (e.g., 6-10 AM, 5-9 PM)
        - Month
        - Season
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features added
        """
        df = df.copy()
        
        if 'Charging Start Time' in df.columns:
            # Extract temporal features
            df['hour'] = df['Charging Start Time'].dt.hour
            df['day_of_week'] = df['Charging Start Time'].dt.dayofweek
            df['day_name'] = df['Charging Start Time'].dt.day_name()
            df['month'] = df['Charging Start Time'].dt.month
            df['day_of_month'] = df['Charging Start Time'].dt.day
            
            # Weekend flag
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Peak hours (morning: 6-10, evening: 17-21)
            df['is_peak_hours'] = (
                ((df['hour'] >= 6) & (df['hour'] < 10)) |
                ((df['hour'] >= 17) & (df['hour'] < 21))
            ).astype(int)
            
            # Off-peak hours (typically cheaper)
            df['is_off_peak'] = (
                (df['hour'] >= 22) | (df['hour'] < 6)
            ).astype(int)
            
            # Season
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        
        return df
    
    def create_user_features(self, df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create user-level aggregated features to prevent data leakage.
        
        NOTE: This method is kept for backward compatibility but user features
        should be created using the standalone create_user_features() function
        and merged before calling engineer_features().
        
        If user features already exist in df, this method does nothing.
        
        Args:
            df: Input DataFrame
            train_df: Training DataFrame for computing statistics (prevents leakage)
            
        Returns:
            DataFrame with user features added
        """
        df = df.copy()
        
        # Check if user features already exist
        user_feature_cols = [
            'user_avg_cost', 'user_cost_std', 'user_charging_frequency',
            'user_avg_energy', 'user_energy_std', 'user_avg_duration',
            'user_duration_std', 'user_avg_soc_gain'
        ]
        
        existing_user_features = [col for col in user_feature_cols if col in df.columns]
        if len(existing_user_features) == len(user_feature_cols):
            # User features already exist, skip creation
            return df
        
        # If user features don't exist, create them (backward compatibility)
        # Determine which dataframe to use for computing statistics
        if train_df is not None:
            # Use training data only to prevent leakage
            stats_df = train_df.copy()
        else:
            # For training data, compute from current dataframe
            stats_df = df.copy()
        
        # Compute user-level statistics
        if 'User ID' in stats_df.columns:
            # Build aggregation dictionary dynamically based on available columns
            agg_dict = {
                'Charging Cost (USD)': ['mean', 'std', 'count'],
                'Energy Consumed (kWh)': ['mean', 'std'],
                'Charging Duration (hours)': ['mean', 'std']
            }
            
            # Add soc_gain if it exists
            if 'soc_gain' in stats_df.columns:
                agg_dict['soc_gain'] = ['mean']
            
            user_stats = stats_df.groupby('User ID').agg(agg_dict).reset_index()
            
            # Flatten column names
            user_stats.columns = ['User ID'] + [
                '_'.join(col).strip() if isinstance(col, tuple) and len(col) > 1 and col[1] else str(col)
                for col in user_stats.columns[1:]
            ]
            
            # Rename columns for clarity
            rename_dict = {
                'Charging Cost (USD)_mean': 'user_avg_cost',
                'Charging Cost (USD)_std': 'user_cost_std',
                'Charging Cost (USD)_count': 'user_charging_frequency',
                'Energy Consumed (kWh)_mean': 'user_avg_energy',
                'Energy Consumed (kWh)_std': 'user_energy_std',
                'Charging Duration (hours)_mean': 'user_avg_duration',
                'Charging Duration (hours)_std': 'user_duration_std',
            }
            
            if 'soc_gain_mean' in user_stats.columns:
                rename_dict['soc_gain_mean'] = 'user_avg_soc_gain'
            
            user_stats = user_stats.rename(columns=rename_dict)
            
            # Fill missing values for new users
            for col in user_stats.columns:
                if col != 'User ID' and col in user_stats.columns:
                    user_stats[col] = user_stats[col].fillna(user_stats[col].median())
            
            # Merge with main dataframe
            df = df.merge(user_stats, on='User ID', how='left')
            
            # Fill NaN values for users not in training set
            for col in user_stats.columns:
                if col != 'User ID' and col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            
            # Store for later use
            self.user_stats = user_stats
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Features:
        - Battery capacity × Charging rate
        - Temperature × Charging efficiency
        - Vehicle age × Charging cost
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()
        
        # Battery capacity × Charging rate
        if 'Battery Capacity (kWh)' in df.columns and 'Charging Rate (kW)' in df.columns:
            df['capacity_rate_interaction'] = df['Battery Capacity (kWh)'] * df['Charging Rate (kW)']
        
        # Temperature × Charging efficiency (if temperature available)
        if 'Temperature (°C)' in df.columns and 'rate_efficiency' in df.columns:
            df['temp_efficiency_interaction'] = df['Temperature (°C)'] * df['rate_efficiency']
        
        # Vehicle age × Cost (older vehicles might have different charging patterns)
        if 'Vehicle Age (years)' in df.columns and 'Charging Cost (USD)' in df.columns:
            df['age_cost_interaction'] = df['Vehicle Age (years)'] * df['Charging Cost (USD)']
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   train_df: Optional[pd.DataFrame] = None,
                                   method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical features.
        
        Strategy:
        - Low cardinality (< 10): One-hot encoding
        - High cardinality: Target encoding (using training data only)
        
        Args:
            df: Input DataFrame
            train_df: Training DataFrame for target encoding (prevents leakage)
            method: Encoding method ('onehot' or 'target')
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude datetime columns
        categorical_cols = [col for col in categorical_cols 
                          if not pd.api.types.is_datetime64_any_dtype(df[col])]
        
        # Separate low and high cardinality
        low_cardinality = []
        high_cardinality = []
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            if n_unique < 10:
                low_cardinality.append(col)
            else:
                high_cardinality.append(col)
        
        # One-hot encode low cardinality features
        if method == 'onehot' or len(high_cardinality) == 0:
            for col in low_cardinality:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        # Target encoding for high cardinality (using training data)
        if train_df is not None and len(high_cardinality) > 0:
            target_col = 'Charging Cost (USD)'
            for col in high_cardinality:
                # Check if column exists in both df and train_df, and target column exists in train_df
                if col not in df.columns:
                    continue
                if col not in train_df.columns:
                    # Column doesn't exist in training data, skip encoding
                    continue
                if target_col not in train_df.columns:
                    # Target column doesn't exist, skip encoding
                    continue
                
                try:
                    # Compute target mean from training data
                    target_mean = train_df.groupby(col)[target_col].mean()
                    # Map to current dataframe
                    df[f'{col}_target_encoded'] = df[col].map(target_mean)
                    # Fill missing values with overall mean
                    df[f'{col}_target_encoded'] = df[f'{col}_target_encoded'].fillna(
                        train_df[target_col].mean()
                    )
                    df = df.drop(columns=[col])
                except (KeyError, ValueError) as e:
                    # Column doesn't exist or other error, skip encoding
                    continue
        elif len(high_cardinality) > 0:
            # For training data, compute target encoding
            target_col = 'Charging Cost (USD)'
            for col in high_cardinality:
                target_mean = df.groupby(col)[target_col].mean()
                df[f'{col}_target_encoded'] = df[col].map(target_mean)
                df[f'{col}_target_encoded'] = df[f'{col}_target_encoded'].fillna(
                    df[target_col].mean()
                )
                df = df.drop(columns=[col])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None,
                         encode_categorical: bool = True, original_train_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            train_df: Training DataFrame (for preventing data leakage) - can be feature-engineered
            encode_categorical: Whether to encode categorical features
            original_train_df: Original training DataFrame before feature engineering (for encoding)
            
        Returns:
            DataFrame with all engineered features
        """
        print("Starting feature engineering...")
        
        # Store original training data for encoding if provided
        if original_train_df is not None:
            self.original_train_df = original_train_df.copy()
        
        # SOC features
        df = self.create_soc_features(df)
        
        # Efficiency features
        df = self.create_efficiency_features(df)
        
        # Cost features
        df = self.create_cost_features(df)
        
        # Time features
        df = self.create_time_features(df)
        
        # User features (with leakage prevention)
        df = self.create_user_features(df, train_df=train_df)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        # Encode categorical features
        if encode_categorical:
            # Use original_train_df for encoding if available, otherwise use train_df
            encoding_train_df = self.original_train_df if self.original_train_df is not None else train_df
            df = self.encode_categorical_features(df, train_df=encoding_train_df)
        
        print(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, target_col: str = 'Charging Cost (USD)') -> list:
        """
        Get list of feature columns (excluding target and ID columns).
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            List of feature column names
        """
        exclude_cols = [
            target_col,
            'User ID',
            'Charging Station ID',
            'Charging Start Time',
            'Charging End Time',
            'Time of Day',  # Already extracted hour
            'Day of Week',  # Already extracted day_of_week
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any remaining datetime columns
        feature_cols = [col for col in feature_cols 
                       if not pd.api.types.is_datetime64_any_dtype(df[col])]
        
        return feature_cols
