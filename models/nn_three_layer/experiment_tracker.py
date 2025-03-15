import json
import os
from datetime import datetime
import tensorflow as tf
from utils import ensure_dir

class ExperimentTracker:
    def __init__(self, experiment_name, output_dir="experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment outputs
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        
        self.experiment_dir = os.path.join(output_dir, self.experiment_id)
        ensure_dir(self.experiment_dir)
        
        self.experiment_data = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "name": experiment_name,
            "parameters": {},
            "metrics": {},
            "feature_groups": [],
            "notes": [],
            "model_summary": "",
            "training_history": {}
        }
        
        print(f"Initialized experiment: {self.experiment_id}")
        print(f"Outputs will be saved to: {self.experiment_dir}")
    
    def log_parameters(self, parameters):
        """Log hyperparameters or other settings."""
        self.experiment_data["parameters"].update(parameters)
        print(f"Logged parameters: {parameters}")
    
    def log_metrics(self, metrics, stage="test"):
        """Log evaluation metrics."""
        if stage not in self.experiment_data["metrics"]:
            self.experiment_data["metrics"][stage] = {}
        
        self.experiment_data["metrics"][stage].update(metrics)
        
        # Print metrics in a readable format
        print(f"\n{stage.upper()} METRICS:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    
    def log_feature_groups(self, included_groups, excluded_groups=None):
        """Log feature groups used in the experiment."""
        feature_info = {
            "included": included_groups,
            "excluded": excluded_groups or []
        }
        self.experiment_data["feature_groups"] = feature_info
        
        print(f"Using feature groups: {', '.join(included_groups)}")
        if excluded_groups:
            print(f"Excluded feature groups: {', '.join(excluded_groups)}")
    
    def add_note(self, note):
        """Add a note to the experiment."""
        self.experiment_data["notes"].append(note)
        print(f"Note: {note}")
    
    def save_experiment(self):
        """Save experiment data to JSON file."""
        experiment_path = os.path.join(self.experiment_dir, "experiment.json")
        
        # Convert numpy values to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                                np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj
        
        serializable_data = json.loads(
            json.dumps(self.experiment_data, default=convert_to_serializable)
        )
        
        with open(experiment_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"Experiment data saved to {experiment_path}")
    
    def get_output_path(self, filename):
        """Get path for saving outputs within experiment directory."""
        return os.path.join(self.experiment_dir, filename)
    
    def log_confusion_matrix(self, cm, stage="test"):
        """Log confusion matrix values."""
        self.experiment_data["metrics"][stage]["confusion_matrix"] = cm.tolist()
    
    def log_model_summary(self, model):
        """Log model summary as string."""
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        model_summary = "\n".join(summary_list)
        
        self.experiment_data["model_summary"] = model_summary
        
        # Save model diagram
        tf.keras.utils.plot_model(
            model, 
            to_file=self.get_output_path("model_architecture.png"),
            show_shapes=True,
            show_layer_names=True
        )
    
    def log_training_history(self, history):
        """Log training history."""
        if hasattr(history, 'history'):
            history = history.history
        
        self.experiment_data["training_history"] = history