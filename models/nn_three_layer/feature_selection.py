import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from utils import ensure_dir
import shap
from sklearn.inspection import permutation_importance
from functools import partial

class FeatureSelector:
    def __init__(self, X_train, y_train, feature_names):
        """
        Initialize feature selector.
        
        Args:
            X_train: Training features (numpy array)
            y_train: Training target (numpy array)
            feature_names: List of feature names
        """
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
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
        
        if isinstance(self.X_train, np.ndarray):
            X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        else:
            X_train_df = self.X_train
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_df, self.y_train)
        selector = SelectFromModel(rf, threshold=threshold)
        selector.fit(X_train_df, self.y_train)
        
        selected_mask = selector.get_support()
        selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) 
                             if selected_mask[i]]
        
        print(f"Selected {len(selected_features)} out of {len(self.feature_names)} features")
        self.feature_importances = dict(zip(self.feature_names, rf.feature_importances_))
        
        return selected_features
    
    def permutation_importance(self, model, X_val, y_val, n_iter=10):
        """
        Calculate permutation importance for neural network model.
        
        Args:
            model: Trained TensorFlow model
            X_val: Validation features
            y_val: Validation target
            n_iter: Number of iterations for permutation importance
            
        Returns:
            dict: Feature importance scores
        """
        print(f"Calculating permutation importance with {n_iter} iterations...")
        
        # Create a scorer function for the TensorFlow model
        def tf_score(X):
            return model.predict(X).flatten()
        
        # Convert to DataFrame if numpy array
        if isinstance(X_val, np.ndarray):
            X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
        else:
            X_val_df = X_val
        
        # Using scikit-learn's permutation_importance instead of ELI5
        perm_importance = permutation_importance(
            estimator=lambda X: tf_score(X),
            X=X_val_df.values,
            y=y_val,
            n_repeats=n_iter,
            random_state=42,
            n_jobs=-1
        )
        
        importances = perm_importance.importances_mean
        importance_std = perm_importance.importances_std
        
        self.feature_importances = dict(zip(self.feature_names, importances))
        self.feature_importance_std = dict(zip(self.feature_names, importance_std))
        
        print("Permutation importance calculation completed")
        return self.feature_importances
    
    def shap_importance(self, model, X_val, background_samples=100):
        """
        Calculate SHAP importance for neural network model.
        
        Args:
            model: Trained TensorFlow model
            X_val: Validation features
            background_samples: Number of background samples for SHAP
            
        Returns:
            dict: Feature importance scores
        """
        print(f"Calculating SHAP importance with {background_samples} background samples...")
        
        # Convert to DataFrame if numpy array
        if isinstance(X_val, np.ndarray):
            X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
        else:
            X_val_df = X_val
        
        # Sample background data for SHAP explainer
        if len(X_val_df) > background_samples:
            background_data = X_val_df.sample(background_samples, random_state=42)
        else:
            background_data = X_val_df
        
        # Create SHAP explainer based on model type
        if isinstance(model, tf.keras.Model):
            # For TensorFlow models
            explainer = shap.DeepExplainer(model, background_data.values)
            shap_values = explainer.shap_values(X_val_df.values)
            
            # For multi-class models, shap_values is a list of arrays, one per class
            if isinstance(shap_values, list):
                # Take absolute mean across all classes
                importances = np.abs(np.array(shap_values)).mean(axis=(0, 1))
            else:
                importances = np.abs(shap_values).mean(axis=0)
        else:
            # For other model types (e.g., XGBoost, RandomForest)
            explainer = shap.Explainer(model, background_data)
            shap_values = explainer(X_val_df)
            importances = np.abs(shap_values.values).mean(axis=0)
        
        self.feature_importances = dict(zip(self.feature_names, importances))
        # SHAP doesn't provide std values directly, but we can create an empty dict for compatibility
        self.feature_importance_std = dict(zip(self.feature_names, [0]*len(self.feature_names)))
        
        print("SHAP importance calculation completed")
        return self.feature_importances
    
    def plot_feature_importance(self, top_n=20, output_path=None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            output_path: Path to save the plot
            
        Returns:
            pandas.DataFrame: Feature importance data
        """
        if not hasattr(self, 'feature_importances'):
            print("Feature importances not calculated yet. Run select_features_rf, permutation_importance, or shap_importance first.")
            return
        
        sorted_importances = sorted(self.feature_importances.items(), 
                                   key=lambda x: x[1], reverse=True)
        top_features = sorted_importances[:top_n]
        
        df_plot = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
        
        if hasattr(self, 'feature_importance_std'):
            std_values = [self.feature_importance_std.get(feature, 0) for feature, _ in top_features]
            df_plot['Std'] = std_values
        
        plt.figure(figsize=(10, 8))
        
        if 'Std' in df_plot.columns and np.sum(df_plot['Std']) > 0:
            sns.barplot(x='Importance', y='Feature', data=df_plot, 
                      xerr=df_plot['Std'], palette='viridis')
        else:
            sns.barplot(x='Importance', y='Feature', data=df_plot, palette='viridis')
        
        plt.title(f'Top {len(df_plot)} Feature Importances')
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
            print("Feature importances not calculated yet. Run select_features_rf, permutation_importance, or shap_importance first.")
            return []
        
        sorted_importances = sorted(self.feature_importances.items(), 
                                   key=lambda x: x[1], reverse=True)
        top_features = [item[0] for item in sorted_importances[:top_n]]
        
        return top_features