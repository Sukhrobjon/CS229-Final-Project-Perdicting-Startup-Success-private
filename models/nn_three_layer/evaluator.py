import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    roc_auc_score, precision_score, recall_score, 
    f1_score, accuracy_score, classification_report
)
import tensorflow as tf
from utils import print_section, ensure_dir

class ModelEvaluator:
    def __init__(self, model, feature_names):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained neural network model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        print(f"ModelEvaluator initialized with {len(feature_names)} features")
    
    def evaluate(self, X, y, split_name="test"):
        """
        Evaluate model on given data.
        
        Args:
            X: Features (numpy array)
            y: Target (numpy array)
            split_name: Name of the split (train, val, test)
            
        Returns:
            dict: Evaluation metrics
        """
        print_section(f"Model Evaluation ({split_name.capitalize()} Set)")
        

        y_pred_proba = self.model.predict(X).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }

        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(cm)
        

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
            X: Features (numpy array)
            y: Target (numpy array)
            output_path: Path to save the plot
            
        Returns:
            None
        """
        y_pred_proba = self.model.predict(X).flatten()
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = roc_auc_score(y, y_pred_proba)
        

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
    
    def plot_precision_recall_curve(self, X, y, output_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            X: Features (numpy array)
            y: Target (numpy array)
            output_path: Path to save the plot
            
        Returns:
            None
        """

        y_pred_proba = self.model.predict(X).flatten()
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        avg_precision = np.mean(precision)

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
    
    def plot_training_history(self, history, output_path=None):
        """
        Plot training history.
        
        Args:
            history: Training history
            output_path: Path to save the plot
            
        Returns:
            None
        """
        # Convert history to dictionary if it's a History object
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot loss
        ax1.plot(history_dict['loss'], label='Training Loss')
        ax1.plot(history_dict['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot metrics
        metrics = ['accuracy', 'auc', 'precision', 'recall']
        for metric in metrics:
            if metric in history_dict and f'val_{metric}' in history_dict:
                ax2.plot(history_dict[metric], label=f'Training {metric.capitalize()}')
                ax2.plot(history_dict[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        
        ax2.set_title('Metrics')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {output_path}")
        
        plt.close()
    
    def find_optimal_threshold(self, X, y):
        """
        Find the optimal classification threshold.
        
        Args:
            X: Features (numpy array)
            y: Target (numpy array)
            
        Returns:
            float: Optimal threshold
        """
        print_section("Finding Optimal Classification Threshold")
        y_pred_proba = self.model.predict(X).flatten()
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
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
            
            X = df[features].values
            y = df['success'].values
            

            metrics = self.evaluate(X, y, split_name=period)
            results[period] = metrics
        
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

        periods = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        

        data = []
        for period in periods:
            for metric in metrics:
                data.append({
                    'Period': period,
                    'Metric': metric,
                    'Value': results[period][metric]
                })
        df_plot = pd.DataFrame(data)
        

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Metric', y='Value', hue='Period', data=df_plot)
        plt.title('Performance Metrics Across Time Periods')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = f"{output_dir}/time_period_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Time period comparison plot saved to {output_path}")
        
        plt.close()