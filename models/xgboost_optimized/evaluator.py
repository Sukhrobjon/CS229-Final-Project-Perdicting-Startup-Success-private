"""
Model evaluation and visualization tools
"""

# evaluator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    roc_auc_score, precision_score, recall_score, 
    f1_score, accuracy_score, classification_report
)
import xgboost as xgb
from utils import print_section, ensure_dir

class ModelEvaluator:
    def __init__(self, model, feature_names):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        print(f"ModelEvaluator initialized with {len(feature_names)} features")
    
    def evaluate(self, X, y, split_name="test"):
        """
        Evaluate model on given data.
        
        Args:
            X: Features
            y: Target
            split_name: Name of the split (train, val, test)
            
        Returns:
            dict: Evaluation metrics
        """
        print_section(f"Model Evaluation ({split_name.capitalize()} Set)")
        
        # Convert to DMatrix for prediction
        dmatrix = xgb.DMatrix(X)
        
        # Make predictions
        y_pred_proba = self.model.predict(dmatrix)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Print metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        return metrics
    
    def plot_confusion_matrix(self, metrics, output_path=None):
        """
        Plot confusion matrix.
        
        Args:
            metrics: Metrics dictionary containing confusion matrix
            output_path: Path to save the plot
            
        Returns:
            None
        """
        cm = metrics['confusion_matrix']
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {output_path}")
        
        plt.close()
    
    def plot_roc_curve(self, X, y, output_path=None):
        """
        Plot ROC curve.
        
        Args:
            X: Features
            y: Target
            output_path: Path to save the plot
            
        Returns:
            None
        """
        # Convert to DMatrix for prediction
        dmatrix = xgb.DMatrix(X)
        
        # Make predictions
        y_pred_proba = self.model.predict(dmatrix)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve plot saved to {output_path}")
        
        plt.close()
    
    # Add to evaluator.py

    def optimize_threshold_for_recent_data(self, X_recent, y_recent, output_path=None):
        """
        Find optimal threshold specifically for recent data.
        
        Args:
            X_recent: Recent period features
            y_recent: Recent period target
            output_path: Path to save visualization
            
        Returns:
            float: Optimal threshold
        """
        print_section("Optimizing Threshold for Recent Data")
        
        # Convert to DMatrix
        dmatrix = xgb.DMatrix(X_recent)
        
        # Make predictions
        y_pred_proba = self.model.predict(dmatrix)
        
        # Calculate precision-recall for different thresholds
        precision, recall, thresholds = precision_recall_curve(y_recent, y_pred_proba)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        
        print(f"Optimal threshold for recent data: {best_threshold:.4f}")
        print(f"F1 score at optimal threshold: {f1_scores[best_idx]:.4f}")
        print(f"Precision at optimal threshold: {precision[best_idx]:.4f}")
        print(f"Recall at optimal threshold: {recall[best_idx]:.4f}")
        
        # Create visualization of different thresholds
        if output_path:
            plt.figure(figsize=(10, 6))
            
            # Plot F1, precision and recall curves
            threshold_df = pd.DataFrame({
                'Threshold': thresholds,
                'Precision': precision[:-1],
                'Recall': recall[:-1],
                'F1 Score': f1_scores
            })
            
            # Plot metrics vs threshold
            plt.plot(threshold_df['Threshold'], threshold_df['F1 Score'], 
                    label='F1 Score', linewidth=2)
            plt.plot(threshold_df['Threshold'], threshold_df['Precision'], 
                    label='Precision', linewidth=2)
            plt.plot(threshold_df['Threshold'], threshold_df['Recall'], 
                    label='Recall', linewidth=2)
            
            # Mark optimal threshold
            plt.axvline(x=best_threshold, color='r', linestyle='--', 
                    label=f'Optimal threshold: {best_threshold:.3f}')
            
            plt.title('Metrics vs. Classification Threshold (Recent Data)')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return best_threshold

    def plot_precision_recall_curve(self, X, y, output_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            X: Features
            y: Target
            output_path: Path to save the plot
            
        Returns:
            None
        """
        # Convert to DMatrix for prediction
        dmatrix = xgb.DMatrix(X)
        
        # Make predictions
        y_pred_proba = self.model.predict(dmatrix)
        
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        
        # Calculate average precision
        avg_precision = np.mean(precision)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label=f'Avg. Precision = {avg_precision:.3f}')
        plt.axhline(y=y.mean(), color='r', linestyle='--', label=f'Baseline (y = {y.mean():.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve plot saved to {output_path}")
        
        plt.close()
    
    def plot_feature_importance(self, top_n=20, output_path=None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            output_path: Path to save the plot
            
        Returns:
            pandas.DataFrame: Feature importance data
        """
        # Get feature importance from model
        importance_scores = self.model.get_score(importance_type='gain')
        
        # Convert to DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': list(importance_scores.keys()),
            'Importance': list(importance_scores.values())
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Select top N features
        if len(importance_df) > top_n:
            importance_df = importance_df.head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {len(importance_df)} Feature Importances')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {output_path}")
        
        plt.close()
        
        return importance_df
    
    def find_optimal_threshold(self, X, y):
        """
        Find the optimal classification threshold.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            float: Optimal threshold
        """
        print_section("Finding Optimal Classification Threshold")
        
        # Convert to DMatrix for prediction
        dmatrix = xgb.DMatrix(X)
        
        # Make predictions
        y_pred_proba = self.model.predict(dmatrix)
        
        # Calculate precision and recall for different thresholds
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"Optimal threshold: {best_threshold:.4f}")
        print(f"F1 score at optimal threshold: {best_f1:.4f}")
        print(f"Precision at optimal threshold: {precision[best_idx]:.4f}")
        print(f"Recall at optimal threshold: {recall[best_idx]:.4f}")
        
        return best_threshold
    
    def evaluate_time_periods(self, time_periods_data, features, output_dir=None):
        """
        Evaluate model performance across different time periods.
        
        Args:
            time_periods_data: Dictionary of DataFrames for different time periods
            features: List of features to use
            output_dir: Directory to save outputs
            
        Returns:
            dict: Evaluation results for each period
        """
        print_section("Time Period Evaluation")
        
        results = {}
        
        for period, df in time_periods_data.items():
            print(f"\n{period.upper()} PERIOD:")
            
            # Extract features and target
            X = df[features]
            y = df['success']
            
            # Evaluate
            metrics = self.evaluate(X, y, split_name=period)
            results[period] = metrics
        
        # Plot comparison
        if output_dir:
            self._plot_time_period_comparison(results, output_dir)
        
        return results
    
    def _plot_time_period_comparison(self, results, output_dir):
        """
        Plot comparison of metrics across time periods.
        
        Args:
            results: Dictionary of evaluation results for each period
            output_dir: Directory to save outputs
            
        Returns:
            None
        """
        # Extract metrics for plotting
        periods = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Create DataFrame for plotting
        data = []
        for period in periods:
            for metric in metrics:
                data.append({
                    'Period': period,
                    'Metric': metric,
                    'Value': results[period][metric]
                })
        df_plot = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Metric', y='Value', hue='Period', data=df_plot)
        plt.title('Performance Metrics Across Time Periods')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = f"{output_dir}/time_period_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Time period comparison plot saved to {output_path}")
        
        plt.close()