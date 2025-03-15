# feature_selection.py

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from utils import ensure_dir

class FeatureSelector:
    def __init__(self, X_train, y_train):
        """
        Initialize feature selector.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = list(X_train.columns)
        print(f"FeatureSelector initialized with {len(self.feature_names)} features")
    
    def select_features_rf(self, threshold='median'):
        """
        Select features using Random Forest importance.
        
        Args:
            threshold: Threshold for feature selection
            
        Returns:
            list: Selected feature names
        """
        print(f"Selecting features using Random Forest with threshold: {threshold}")
        
        # Train a Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        
        # Select features
        selector = SelectFromModel(rf, threshold=threshold)
        selector.fit(self.X_train, self.y_train)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) 
                             if selected_mask[i]]
        
        print(f"Selected {len(selected_features)} out of {len(self.feature_names)} features")
        
        # Store feature importances
        self.feature_importances = dict(zip(self.feature_names, rf.feature_importances_))
        
        return selected_features
    
    def plot_feature_importance(self, top_n=20, output_path=None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            output_path: Path to save the plot
            
        Returns:
            None
        """
        if not hasattr(self, 'feature_importances'):
            print("Feature importances not calculated yet. Run select_features_rf first.")
            return
        
        # Sort features by importance
        sorted_importances = sorted(self.feature_importances.items(), 
                                   key=lambda x: x[1], reverse=True)
        top_features = sorted_importances[:top_n]
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=df_plot)
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {output_path}")
        
        plt.close()
        
        return df_plot
    
    def get_top_features(self, top_n=20):
        """
        Get top n important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            list: Top feature names
        """
        if not hasattr(self, 'feature_importances'):
            print("Feature importances not calculated yet. Run select_features_rf first.")
            return []
        
        # Sort features by importance
        sorted_importances = sorted(self.feature_importances.items(), 
                                   key=lambda x: x[1], reverse=True)
        top_features = [item[0] for item in sorted_importances[:top_n]]
        
        return top_features