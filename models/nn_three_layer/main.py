import os
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from utils import ensure_dir, print_section, generate_timestamp, set_gpu_memory_growth
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from feature_selection import FeatureSelector
from experiment_tracker import ExperimentTracker

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Startup Success Prediction (Neural Network)')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='output_nn', help='Directory to save outputs')
    

    parser.add_argument('--include_features', type=str, nargs='+', default=None, 
                        choices=['team_competency', 'econ_vc', 'temporal', 'location', 'lda_topics'],
                        help='Feature groups to include')
    parser.add_argument('--exclude_features', type=str, nargs='+', default=None,
                        choices=['team_competency', 'econ_vc', 'temporal', 'location', 'lda_topics'],
                        help='Feature groups to exclude')
    
 
    parser.add_argument('--train_years', type=int, nargs='+', default=[2012, 2013, 2014, 2015, 2016, 2017, 2018],
                        help='Years for training set')
    parser.add_argument('--val_years', type=int, nargs='+', default=[2019, 2020],
                        help='Years for validation set')
    parser.add_argument('--test_years', type=int, nargs='+', default=[2021, 2022],
                        help='Years for test set')
    
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--hyperparameter_tuning', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--max_trials', type=int, default=30, help='Maximum number of hyperparameter tuning trials')
    parser.add_argument('--k_fold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--run_cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--experiment_name', type=str, default='startup_success_nn',
                        help='Name of the experiment')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.use_gpu:
        set_gpu_memory_growth()

    ensure_dir(args.output_dir)

    tracker = ExperimentTracker(args.experiment_name, args.output_dir)
    
    tracker.log_parameters(vars(args))
    
    print_section("Loading Data")
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    

    processor = DataProcessor(df)

    tracker.log_feature_groups(
        included_groups=args.include_features or list(processor.feature_groups.keys()),
        excluded_groups=args.exclude_features
    )
    
    features = processor.filter_features(args.include_features, args.exclude_features)
    
    data_splits = processor.split_time_based(
        'founded_year', 
        args.train_years, 
        args.val_years, 
        args.test_years
    )
    
    data = processor.prepare_data_splits(data_splits, features, normalize=True)
    trainer = ModelTrainer(use_gpu=args.use_gpu)

    model_dir = os.path.join(tracker.experiment_dir, 'model')
    ensure_dir(model_dir)
    
    if args.hyperparameter_tuning:
        tuner_dir = os.path.join(tracker.experiment_dir, 'tuner')
        ensure_dir(tuner_dir)
        
        best_params = trainer.hyperparameter_tuning(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            max_trials=args.max_trials,
            output_dir=tuner_dir,
            class_weights=data.get('class_weights')
        )
        tracker.log_parameters({"best_params": best_params})
    else:
        best_params = None
    

    model, history = trainer.train_model(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        params=best_params,
        output_dir=model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        class_weights=data.get('class_weights')
    )
    

    tracker.log_model_summary(model)
    

    tracker.log_training_history(history.history)
    
    if args.run_cv:
        cv_results = trainer.kfold_cv(
            np.concatenate([data['X_train'], data['X_val']]),
            np.concatenate([data['y_train'], data['y_val']]),
            best_params,
            k=args.k_fold,
            batch_size=args.batch_size,
            epochs=args.epochs,
            class_weights=data.get('class_weights')
        )
        

        tracker.log_metrics(cv_results['mean_metrics'], stage='cv')
    

    evaluator = ModelEvaluator(model, data['feature_names'])
    
    if 'X_train_df' in data and 'X_val_df' in data:
        feature_selector = FeatureSelector(
            np.concatenate([data['X_train'], data['X_val']]),
            np.concatenate([data['y_train'], data['y_val']]),
            data['feature_names']
        )
        
        feature_selector.permutation_importance(
            model, data['X_val'], data['y_val'], n_iter=10
        )
    
    train_metrics = evaluator.evaluate(data['X_train'], data['y_train'], split_name='train')
    tracker.log_metrics(train_metrics, stage='train')
    tracker.log_confusion_matrix(train_metrics['confusion_matrix'], stage='train')
    
    val_metrics = evaluator.evaluate(data['X_val'], data['y_val'], split_name='val')
    tracker.log_metrics(val_metrics, stage='val')
    tracker.log_confusion_matrix(val_metrics['confusion_matrix'], stage='val')
    
    test_metrics = evaluator.evaluate(data['X_test'], data['y_test'], split_name='test')
    tracker.log_metrics(test_metrics, stage='test')
    tracker.log_confusion_matrix(test_metrics['confusion_matrix'], stage='test')
    
    plot_dir = os.path.join(tracker.experiment_dir, 'plots')
    ensure_dir(plot_dir)
    evaluator.plot_training_history(history, f"{plot_dir}/training_history.png")
    
    evaluator.plot_confusion_matrix(train_metrics, f"{plot_dir}/train_confusion_matrix.png")
    evaluator.plot_confusion_matrix(val_metrics, f"{plot_dir}/val_confusion_matrix.png")
    evaluator.plot_confusion_matrix(test_metrics, f"{plot_dir}/test_confusion_matrix.png")
    
    evaluator.plot_roc_curve(data['X_train'], data['y_train'], f"{plot_dir}/train_roc_curve.png")
    evaluator.plot_roc_curve(data['X_val'], data['y_val'], f"{plot_dir}/val_roc_curve.png")
    evaluator.plot_roc_curve(data['X_test'], data['y_test'], f"{plot_dir}/test_roc_curve.png")
    
    evaluator.plot_precision_recall_curve(data['X_train'], data['y_train'], f"{plot_dir}/train_pr_curve.png")
    evaluator.plot_precision_recall_curve(data['X_val'], data['y_val'], f"{plot_dir}/val_pr_curve.png")
    evaluator.plot_precision_recall_curve(data['X_test'], data['y_test'], f"{plot_dir}/test_pr_curve.png")
    
    if hasattr(feature_selector, 'feature_importances'):
        importance_df = feature_selector.plot_feature_importance(top_n=20, output_path=f"{plot_dir}/feature_importance.png")
        tracker.log_parameters({"feature_importance": importance_df.to_dict('records')})
    
    optimal_threshold = evaluator.find_optimal_threshold(data['X_val'], data['y_val'])
    tracker.log_parameters({"optimal_threshold": optimal_threshold})
    
    if len(set(args.train_years + args.val_years + args.test_years)) >= 6:
        all_years = sorted(set(args.train_years + args.val_years + args.test_years))
        n_years = len(all_years)
        
        early_years = all_years[:n_years//3]
        mid_years = all_years[n_years//3:2*n_years//3]
        recent_years = all_years[2*n_years//3:]
        time_periods = processor.preprocess_time_periods(early_years, mid_years, recent_years)
        
        time_period_data = {}
        for period, df in time_periods.items():
            X = df[features].values
            # Apply normalization using the same scaler
            X = processor.scaler.transform(X)
            time_period_data[period] = X
        

        time_period_results = evaluator.evaluate_time_periods(time_periods, features, plot_dir)
    
        for period, metrics in time_period_results.items():
            tracker.log_metrics(metrics, stage=f"period_{period}")
    
    # Save model
    model_path = os.path.join(model_dir, "final_model.h5")
    trainer.save_model(model_path)
    
    tracker.save_experiment()
    
    print_section("Experiment Completed")
    print(f"Results saved to {tracker.experiment_dir}")

if __name__ == "__main__":
    main()