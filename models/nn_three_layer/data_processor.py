import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import print_section, split_features_by_group
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, df, target_col='success'):
        """
        Initialize data processor.
        
        Args:
            df: pandas DataFrame containing the dataset
            target_col: Name of the target column
        """
        self.df = df.copy()
        self.target_col = target_col
        self.feature_groups = split_features_by_group(df, target_col)
        self.scaler = StandardScaler()
        print(f"DataProcessor initialized with {len(df)} records and {len(df.columns)} columns")
    
    def filter_features(self, include_groups=None, exclude_groups=None):
        """
        Filter features based on groups to include or exclude.
        
        Args:
            include_groups: List of feature groups to include, None means all
            exclude_groups: List of feature groups to exclude
            
        Returns:
            List of selected feature names
        """
        if include_groups is None:
            # Include all feature groups
            selected_features = set()
            for group, features in self.feature_groups.items():
                if group != 'all':  # Skip the 'all' group to avoid duplication
                    selected_features.update(features)
        else:
            # Include only specified groups
            selected_features = set()
            for group in include_groups:
                if group in self.feature_groups:
                    selected_features.update(self.feature_groups[group])
                else:
                    print(f"Warning: Group '{group}' not found in feature groups")
        
        if exclude_groups:
            for group in exclude_groups:
                if group in self.feature_groups:
                    selected_features.difference_update(self.feature_groups[group])
                else:
                    print(f"Warning: Group '{group}' not found in feature groups")
        

        if self.target_col in selected_features:
            selected_features.remove(self.target_col)
        
        return list(selected_features)
    
    def split_time_based(self, time_col, train_years, val_years, test_years):
        """
        Split data based on time periods.
        
        Args:
            time_col: Column name containing the year
            train_years: List of years for training set
            val_years: List of years for validation set
            test_years: List of years for test set
            
        Returns:
            dict: Dictionary containing train, val, test splits and features
        """
        print_section("Time-Based Data Split")

        train_mask = self.df[time_col].isin(train_years)
        val_mask = self.df[time_col].isin(val_years)
        test_mask = self.df[time_col].isin(test_years)

        train_df = self.df[train_mask].copy()
        val_df = self.df[val_mask].copy()
        test_df = self.df[test_mask].copy()

        print(f"Train set: {len(train_df)} records ({len(train_df)/len(self.df)*100:.2f}%) from years {train_years}")
        print(f"Validation set: {len(val_df)} records ({len(val_df)/len(self.df)*100:.2f}%) from years {val_years}")
        print(f"Test set: {len(test_df)} records ({len(test_df)/len(self.df)*100:.2f}%) from years {test_years}")
        

        train_success_rate = train_df[self.target_col].mean() * 100
        val_success_rate = val_df[self.target_col].mean() * 100
        test_success_rate = test_df[self.target_col].mean() * 100
        
        print(f"Success rate in train set: {train_success_rate:.2f}%")
        print(f"Success rate in validation set: {val_success_rate:.2f}%")
        print(f"Success rate in test set: {test_success_rate:.2f}%")
        

        result = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'success_rates': {
                'train': train_success_rate,
                'val': val_success_rate,
                'test': test_success_rate
            }
        }
        
        return result
    
    def prepare_data_splits(self, data_splits, features, normalize=True):
        """
        Prepare X and y splits from data splits, with optional normalization.
        
        Args:
            data_splits: Dictionary containing train, val, test DataFrames
            features: List of feature names to include
            normalize: Whether to normalize features using StandardScaler
            
        Returns:
            dict: Dictionary containing X and y for train, val, test
        """
        print(f"Preparing data with {len(features)} features")
        
        result = {}
        
        for split_name in ['train', 'val', 'test']:
            if split_name in data_splits:
                df = data_splits[split_name]
                X = df[features].values
                y = df[self.target_col].values
                
                # Store original DataFrames for feature importance analysis
                result[f'X_{split_name}_df'] = df[features]
                
                if normalize and split_name == 'train':
                    self.scaler.fit(X)
                    print(f"Fitted scaler on {split_name} data")
                
                if normalize:
                    X = self.scaler.transform(X)
                    print(f"Normalized {split_name} data")
                
                result[f'X_{split_name}'] = X
                result[f'y_{split_name}'] = y

        result['feature_names'] = features

        if 'y_train' in result:
            n_samples = len(result['y_train'])
            n_positive = np.sum(result['y_train'])
            n_negative = n_samples - n_positive
            
            # Calculate balanced class weights
            weight_positive = n_samples / (2 * n_positive) if n_positive > 0 else 1.0
            weight_negative = n_samples / (2 * n_negative) if n_negative > 0 else 1.0
            
            class_weights = {
                0: weight_negative,
                1: weight_positive
            }
            
            print(f"Class weights: {class_weights}")
            result['class_weights'] = class_weights
        
        return result
    
    def preprocess_time_periods(self, early_years, mid_years, recent_years):
        """
        Create multiple time period models to address time drift.
        
        Args:
            early_years: List of early years for first model
            mid_years: List of middle years for second model
            recent_years: List of recent years for third model
            
        Returns:
            dict: Dictionary of time period datasets
        """
        print_section("Multi-Period Data Preparation")
        
        time_periods = {
            'early': self.df[self.df['founded_year'].isin(early_years)].copy(),
            'mid': self.df[self.df['founded_year'].isin(mid_years)].copy(),
            'recent': self.df[self.df['founded_year'].isin(recent_years)].copy()
        }
        
        for period, df in time_periods.items():
            success_rate = df[self.target_col].mean() * 100
            print(f"{period.capitalize()} period ({early_years if period == 'early' else mid_years if period == 'mid' else recent_years}): {len(df)} records, Success rate: {success_rate:.2f}%")
        
        return time_periods