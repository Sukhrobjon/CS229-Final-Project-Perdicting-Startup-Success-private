import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
        
        # Exclude specified groups
        if exclude_groups:
            for group in exclude_groups:
                if group in self.feature_groups:
                    selected_features.difference_update(self.feature_groups[group])
                else:
                    print(f"Warning: Group '{group}' not found in feature groups")
        
        # Ensure target column is not in the selected features
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
        
        # Create splits
        train_mask = self.df[time_col].isin(train_years)
        val_mask = self.df[time_col].isin(val_years)
        test_mask = self.df[time_col].isin(test_years)
        
        # Extract splits
        train_df = self.df[train_mask].copy()
        val_df = self.df[val_mask].copy()
        test_df = self.df[test_mask].copy()
        
        # Print split info
        print(f"Train set: {len(train_df)} records ({len(train_df)/len(self.df)*100:.2f}%) from years {train_years}")
        print(f"Validation set: {len(val_df)} records ({len(val_df)/len(self.df)*100:.2f}%) from years {val_years}")
        print(f"Test set: {len(test_df)} records ({len(test_df)/len(self.df)*100:.2f}%) from years {test_years}")
        
        # Print success rates
        train_success_rate = train_df[self.target_col].mean() * 100
        val_success_rate = val_df[self.target_col].mean() * 100
        test_success_rate = test_df[self.target_col].mean() * 100
        
        print(f"Success rate in train set: {train_success_rate:.2f}%")
        print(f"Success rate in validation set: {val_success_rate:.2f}%")
        print(f"Success rate in test set: {test_success_rate:.2f}%")
        
        # Return data splits
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
    
    def prepare_data_splits(self, data_splits, features):
        """
        Prepare X and y splits from data splits.
        
        Args:
            data_splits: Dictionary containing train, val, test DataFrames
            features: List of feature names to include
            
        Returns:
            dict: Dictionary containing X and y for train, val, test
        """
        print(f"Preparing data with {len(features)} features")
        
        result = {}
        for split_name in ['train', 'val', 'test']:
            if split_name in data_splits:
                df = data_splits[split_name]
                result[f'X_{split_name}'] = df[features]
                result[f'y_{split_name}'] = df[self.target_col]
        
        return result
    

    def add_time_interaction_features(self, features_to_interact=None):
        """
        Add interaction features between time and other features.
        
        Args:
            features_to_interact: List of features to create interactions with,
                                None means use top team and sector features
        
        Returns:
            pandas.DataFrame: DataFrame with added interaction features
        """
        print_section("Adding Time Interaction Features")
        
        df = self.df.copy()
        
        # If no features specified, use the most important team and sector features
        if features_to_interact is None:
            team_features = ['founderCount', 'has_previous_startups_total', 
                            'has_executive_experience']
            topic_features = [f'topic_prob_{i}' for i in range(15)][:5]  # Top 5 topics
            features_to_interact = team_features + topic_features
        
        print(f"Creating time interactions for {len(features_to_interact)} features")
        
        # Create time period feature for clear interpretability
        if 'time_period' not in df.columns:
            conditions = [
                df['founded_year'].isin([2012, 2013, 2014]),
                df['founded_year'].isin([2015, 2016, 2017, 2018]),
                df['founded_year'].isin([2019, 2020, 2021, 2022])
            ]
            choices = [0, 1, 2]  # 0=early, 1=mid, 2=recent
            df['time_period'] = np.select(conditions, choices, default=1)
        
        # Create interaction features
        for feature in features_to_interact:
            df[f'{feature}_x_period'] = df[feature] * df['time_period']
            print(f"  Created: {feature}_x_period")
        
        self.df = df
        print(f"Added {len(features_to_interact)} time interaction features")
        
        return df
    

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