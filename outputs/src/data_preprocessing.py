"""
Data preprocessing module for EV charging optimization.

Handles:
- Missing value imputation
- Outlier detection and treatment
- Timestamp parsing
- Data validation
- Time-based train-test split
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Class for preprocessing EV charging data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.outlier_threshold = 3.0  # Z-score threshold for outliers
        
    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse timestamp columns and extract temporal features.
        
        Args:
            df: DataFrame with 'Charging Start Time' and 'Charging End Time'
            
        Returns:
            DataFrame with parsed timestamps
        """
        df = df.copy()
        
        # Parse timestamps
        df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'])
        df['Charging End Time'] = pd.to_datetime(df['Charging End Time'])
        
        # Validate timestamps (end should be after start)
        invalid_mask = df['Charging End Time'] <= df['Charging Start Time']
        if invalid_mask.sum() > 0:
            print(f"Warning: Found {invalid_mask.sum()} sessions with invalid timestamps. Removing them.")
            df = df[~invalid_mask].reset_index(drop=True)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Strategy:
        - Numerical: median imputation
        - Categorical: mode imputation
        - Timestamps: forward fill or drop
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("Missing values found:")
            print(missing[missing > 0])
        
        # Numerical columns: median imputation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns: mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers. If None, checks all numerical columns.
            
        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID columns and target variables
            exclude_cols = ['User ID', 'Charging Station ID']
            columns = [col for col in columns if col not in exclude_cols]
        
        outlier_flags = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_flags[f'{col}_outlier'] = z_scores > self.outlier_threshold
        
        return outlier_flags
    
    def treat_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """
        Treat outliers in numerical columns.
        
        Args:
            df: Input DataFrame
            method: 'cap' (winsorize) or 'remove' (drop rows)
            
        Returns:
            DataFrame with outliers treated
        """
        df = df.copy()
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['User ID', 'Charging Station ID']
        columns = [col for col in numerical_cols if col not in exclude_cols]
        
        if method == 'cap':
            # Cap outliers at 1st and 99th percentiles
            for col in columns:
                if col in df.columns:
                    lower_bound = df[col].quantile(0.01)
                    upper_bound = df[col].quantile(0.99)
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            # Remove rows with outliers
            outlier_flags = self.detect_outliers(df, columns)
            outlier_rows = outlier_flags.any(axis=1)
            df = df[~outlier_rows].reset_index(drop=True)
            print(f"Removed {outlier_rows.sum()} rows with outliers")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data validation rules.
        
        Rules:
        - SOC should be between 0 and 100
        - Energy consumed should be positive
        - Charging duration should be positive
        - End SOC should be >= Start SOC
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Validate SOC values
        soc_cols = [col for col in df.columns if 'State of Charge' in col or 'SOC' in col]
        for col in soc_cols:
            df = df[(df[col] >= 0) & (df[col] <= 100)]
        
        # Validate energy consumed
        if 'Energy Consumed (kWh)' in df.columns:
            df = df[df['Energy Consumed (kWh)'] > 0]
        
        # Validate charging duration
        if 'Charging Duration (hours)' in df.columns:
            df = df[df['Charging Duration (hours)'] > 0]
        
        # Validate SOC progression
        if 'State of Charge (Start %)' in df.columns and 'State of Charge (End %)' in df.columns:
            df = df[df['State of Charge (End %)'] >= df['State of Charge (Start %)']]
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows that failed validation")
        
        return df.reset_index(drop=True)
    
    def time_based_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                        time_col: str = 'Charging Start Time') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform time-based train-test split to prevent data leakage.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data to use for testing
            time_col: Column name for time-based splitting
            
        Returns:
            Tuple of (train_df, test_df)
        """
        df = df.copy()
        df = df.sort_values(by=time_col).reset_index(drop=True)
        
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        print(f"Train date range: {train_df[time_col].min()} to {train_df[time_col].max()}")
        print(f"Test date range: {test_df[time_col].min()} to {test_df[time_col].max()}")
        
        return train_df, test_df
    
    def preprocess(self, df: pd.DataFrame, treat_outliers: bool = True, 
                   outlier_method: str = 'cap') -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw input DataFrame
            treat_outliers: Whether to treat outliers
            outlier_method: Method for outlier treatment ('cap' or 'remove')
            
        Returns:
            Preprocessed DataFrame
        """
        print("Starting data preprocessing...")
        
        # Parse timestamps
        df = self.parse_timestamps(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Validate data
        df = self.validate_data(df)
        
        # Treat outliers
        if treat_outliers:
            df = self.treat_outliers(df, method=outlier_method)
        
        print(f"Preprocessing complete. Final dataset shape: {df.shape}")
        return df
