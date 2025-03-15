import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
import json

class ShapAnalyzer:
    def __init__(self, model_path, feature_names, output_dir):
        """
        Initialize SHAP analyzer.
        
        Args:
            model_path: Path to the saved XGBoost model
            feature_names: List of feature names
            output_dir: Directory to save SHAP visualizations
        """
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze(self, X, sample_size=1000):
        """
        Run SHAP analysis on input data.
        
        Args:
            X: DataFrame with features
            sample_size: Number of samples to use for SHAP analysis
        """
        # Create output directory for SHAP plots
        shap_dir = self.output_dir / "shap_analysis"
        shap_dir.mkdir(exist_ok=True)
        
        # Sample data if needed (SHAP can be computationally expensive)
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
            
        # Create DMatrix for XGBoost
        dmatrix = xgb.DMatrix(X_sample, feature_names=self.feature_names)
        
        # Initialize SHAP explainer
        print("Initializing SHAP explainer...")
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # Create and save SHAP plots
        self._create_summary_plot(explainer, shap_values, X_sample, shap_dir)
        self._create_dependence_plots(explainer, shap_values, X_sample, shap_dir)
        self._create_force_plots(explainer, shap_values, X_sample, shap_dir)
        self._save_shap_values(shap_values, X_sample, shap_dir)
        
        return shap_values
    
    def _create_summary_plot(self, explainer, shap_values, X, output_dir):
        """Create and save SHAP summary plot."""
        print("Creating summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_dependence_plots(self, explainer, shap_values, X, output_dir):
        """Create and save SHAP dependence plots for top features."""
        print("Creating dependence plots for top features...")
        # Get mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(self.feature_names, mean_abs_shap)), 
                                        columns=['Feature', 'SHAP_Importance'])
        feature_importance = feature_importance.sort_values('SHAP_Importance', ascending=False)
        
        # Create dependence plots for top 10 features
        for feature in feature_importance['Feature'][:10]:
            plt.figure(figsize=(10, 6))
            i = self.feature_names.index(feature)
            shap.dependence_plot(i, shap_values, X, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_dependence_{feature}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_force_plots(self, explainer, shap_values, X, output_dir):
        """Create and save SHAP force plots for sample instances."""
        print("Creating force plots for sample instances...")
        # Get examples of high, medium and low predictions
        pred = self.model.predict(xgb.DMatrix(X))
        high_idx = np.argmax(pred)
        low_idx = np.argmin(pred)
        mid_idx = np.argsort(pred)[len(pred)//2]
        
        for label, idx in [("high", high_idx), ("medium", mid_idx), ("low", low_idx)]:
            plt.figure(figsize=(16, 4))
            shap.force_plot(explainer.expected_value, shap_values[idx], X.iloc[idx], 
                          feature_names=self.feature_names, matplotlib=True, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_force_plot_{label}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_shap_values(self, shap_values, X, output_dir):
        """Save SHAP values to CSV for further analysis."""
        print("Saving SHAP values to CSV...")
        shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
        shap_df.to_csv(output_dir / "shap_values.csv", index=False)
        
        # Save feature importance based on SHAP
        mean_abs_shap = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(self.feature_names, mean_abs_shap)), 
                                        columns=['Feature', 'SHAP_Importance'])
        feature_importance = feature_importance.sort_values('SHAP_Importance', ascending=False)
        feature_importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)