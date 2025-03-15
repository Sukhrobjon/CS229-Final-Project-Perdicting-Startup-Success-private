import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import ensure_dir, print_section, generate_timestamp
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from feature_selection import FeatureSelector
from experiment_tracker import ExperimentTracker
from shap_analyzer import ShapAnalyzer
import config



# Add this function to main.py
def train_time_specific_models(df, features, output_dir):
    """Train separate models for different time periods."""
    print_section("Training Time-Specific Models")
    
    # Define time periods
    time_periods = {
        'early': [2012, 2013, 2014, 2015],
        'mid': [2016, 2017, 2018],
        'recent': [2019, 2020, 2021, 2022]
    }
    
    # Create output directory
    models_dir = os.path.join(output_dir, 'time_specific_models')
    ensure_dir(models_dir)
    
    results = {}
    
    # Train model for each time period
    for period_name, years in time_periods.items():
        print(f"\n{period_name.upper()} PERIOD MODEL (Years: {years})")
        
        # Filter data for this period
        period_df = df[df['founded_year'].isin(years)]
        
        # Split into train/val/test (80/10/10)
        train_val_test = np.random.choice(
            ['train', 'val', 'test'], 
            size=len(period_df), 
            p=[0.8, 0.1, 0.1]
        )
        
        train_df = period_df[train_val_test == 'train']
        val_df = period_df[train_val_test == 'val']
        test_df = period_df[train_val_test == 'test']
        
        # Prepare data
        X_train, y_train = train_df[features], train_df['success']
        X_val, y_val = val_df[features], val_df['success']
        X_test, y_test = test_df[features], test_df['success']
        
        # Train model
        trainer = ModelTrainer(use_gpu=True)
        best_params, model, _ = trainer.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, max_evals=30
        )
        
        # Evaluate model
        evaluator = ModelEvaluator(model, features)
        metrics = evaluator.evaluate(X_test, y_test)
        
        # Save model and results
        model_path = os.path.join(models_dir, f"{period_name}_model.json")
        trainer.save_model(model_path)
        
        results[period_name] = {
            'metrics': metrics,
            'params': best_params
        }
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Startup Success Prediction')
    
    parser.add_argument('--train_time_specific', action='store_true', 
                   help='Train separate models for different time periods')


    parser.add_argument('--run_shap_analysis', action='store_true',
                   help='Run SHAP analysis for model explainability')

    parser.add_argument('--add_time_interactions', action='store_true',
                   help='Add time interaction features')
    # Data parameters
    parser.add_argument('--data_path', type=str, default=config.DATA_PATH)
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR)
    
    # Feature selection parameters
    parser.add_argument('--include_features', type=str, nargs='+', default=config.INCLUDE_FEATURES, 
                        choices=['team_competency', 'econ_vc', 'temporal', 'location', 'lda_topics'])
    parser.add_argument('--exclude_features', type=str, nargs='+', default=config.EXCLUDE_FEATURES,
                        choices=['team_competency', 'econ_vc', 'temporal', 'location', 'lda_topics'])
    
    # Time splitting parameters
    parser.add_argument('--train_years', type=int, nargs='+', default=config.TRAIN_YEARS)
    parser.add_argument('--val_years', type=int, nargs='+', default=config.VAL_YEARS)
    parser.add_argument('--test_years', type=int, nargs='+', default=config.TEST_YEARS)
    
    # Model parameters
    parser.add_argument('--use_gpu', action='store_true', default=config.USE_GPU)
    parser.add_argument('--n_trials', type=int, default=config.N_TRIALS)
    parser.add_argument('--k_fold', type=int, default=config.K_FOLD)
    parser.add_argument('--run_cv', action='store_true', default=config.RUN_CV)
    
    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default=config.EXPERIMENT_NAME)
    
    # Predefined experiment configuration
    parser.add_argument('--config_name', type=str, choices=list(config.EXPERIMENT_CONFIGS.keys()))
    
    args = parser.parse_args()
    
    # Override with predefined config if specified
    if args.config_name and args.config_name in config.EXPERIMENT_CONFIGS:
        config_dict = config.EXPERIMENT_CONFIGS[args.config_name]
        for key, value in config_dict.items():
            setattr(args, key.lower(), value)
    
    return args


def plot_feature_experiment_comparison(results, output_dir):
    """Plot comparison of different feature group experiments."""
    # Extract metrics for plotting
    experiments = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Create DataFrame for plotting
    data = []
    for experiment in experiments:
        for metric in metrics:
            data.append({
                'Experiment': experiment,
                'Metric': metric,
                'Value': results[experiment]['test_metrics'][metric],
                'Feature Count': results[experiment]['feature_count']
            })
    df_plot = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Experiment', y='Value', hue='Metric', data=df_plot)
    plt.title('Performance Metrics Across Feature Group Experiments')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "feature_experiment_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    tracker = ExperimentTracker(args.experiment_name, args.output_dir)
 
    tracker.log_parameters(vars(args))
    
    print_section("Loading Data")
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    processor = DataProcessor(df)
    
    # Log feature groups
    tracker.log_feature_groups(
        included_groups=args.include_features or list(processor.feature_groups.keys()),
        excluded_groups=args.exclude_features
    )
    
    features = processor.filter_features(args.include_features, args.exclude_features)
    
    if args.add_time_interactions:
        df = processor.add_time_interaction_features()

    # Split data based on time
    data_splits = processor.split_time_based(
        'founded_year', 
        args.train_years, 
        args.val_years, 
        args.test_years
    )
    
    # Prepare data
    data = processor.prepare_data_splits(data_splits, features)
    
    # Initialize model trainer
    trainer = ModelTrainer(use_gpu=args.use_gpu)
    
    # Optimize hyperparameters
    best_params, best_model, best_metrics = trainer.optimize_hyperparameters(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        max_evals=args.n_trials
    )
    
    # Log optimization results
    tracker.log_metrics(best_metrics, stage='optimization')
    tracker.log_parameters({"best_params": best_params})
    
    # Train final model
    model = trainer.train_final_model(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        params=best_params
    )
    
    # Run cross-validation
    if args.run_cv:
        cv_results = trainer.kfold_cv(
            pd.concat([data['X_train'], data['X_val']]),
            pd.concat([data['y_train'], data['y_val']]),
            best_params,
            k=args.k_fold
        )

        tracker.log_metrics(cv_results['mean_metrics'], stage='cv')
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, features)
    
    # Evaluate on training set
    train_metrics = evaluator.evaluate(data['X_train'], data['y_train'], split_name='train')
    tracker.log_metrics(train_metrics, stage='train')
    tracker.log_confusion_matrix(train_metrics['confusion_matrix'], stage='train')
    
    # Evaluate on validation set
    val_metrics = evaluator.evaluate(data['X_val'], data['y_val'], split_name='val')
    tracker.log_metrics(val_metrics, stage='val')
    tracker.log_confusion_matrix(val_metrics['confusion_matrix'], stage='val')
    
    # Evaluate on test set
    test_metrics = evaluator.evaluate(data['X_test'], data['y_test'], split_name='test')
    tracker.log_metrics(test_metrics, stage='test')
    tracker.log_confusion_matrix(test_metrics['confusion_matrix'], stage='test')
    
    # Save plots
    plot_dir = os.path.join(tracker.experiment_dir, 'plots')
    ensure_dir(plot_dir)
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrix(train_metrics, f"{plot_dir}/train_confusion_matrix.png")
    evaluator.plot_confusion_matrix(val_metrics, f"{plot_dir}/val_confusion_matrix.png")
    evaluator.plot_confusion_matrix(test_metrics, f"{plot_dir}/test_confusion_matrix.png")
    
    # Plot ROC curves
    evaluator.plot_roc_curve(data['X_train'], data['y_train'], f"{plot_dir}/train_roc_curve.png")
    evaluator.plot_roc_curve(data['X_val'], data['y_val'], f"{plot_dir}/val_roc_curve.png")
    evaluator.plot_roc_curve(data['X_test'], data['y_test'], f"{plot_dir}/test_roc_curve.png")
    
    # Plot precision-recall curves
    evaluator.plot_precision_recall_curve(data['X_train'], data['y_train'], f"{plot_dir}/train_pr_curve.png")
    evaluator.plot_precision_recall_curve(data['X_val'], data['y_val'], f"{plot_dir}/val_pr_curve.png")
    evaluator.plot_precision_recall_curve(data['X_test'], data['y_test'], f"{plot_dir}/test_pr_curve.png")
    
    # Plot feature importance
    importance_df = evaluator.plot_feature_importance(top_n=20, output_path=f"{plot_dir}/feature_importance.png")
    
    # Log feature importance
    tracker.log_feature_importance(importance_df.to_dict('records'))
    
    # Find optimal threshold
    optimal_threshold = evaluator.find_optimal_threshold(data['X_val'], data['y_val'])
    tracker.log_parameters({"optimal_threshold": optimal_threshold})
    
    # Evaluate time periods if we have multiple years
    if len(set(args.train_years + args.val_years + args.test_years)) >= 6:
        # Define early, mid, and recent periods
        all_years = sorted(set(args.train_years + args.val_years + args.test_years))
        n_years = len(all_years)
        
        early_years = all_years[:n_years//3]
        mid_years = all_years[n_years//3:2*n_years//3]
        recent_years = all_years[2*n_years//3:]
        
        # Process time periods
        time_periods = processor.preprocess_time_periods(early_years, mid_years, recent_years)
        
        # Evaluate across time periods
        time_period_results = evaluator.evaluate_time_periods(time_periods, features, plot_dir)
        
        # Log time period results
        for period, metrics in time_period_results.items():
            tracker.log_metrics(metrics, stage=f"period_{period}")
    
    # if args.run_shap_analysis:        
    #     print_section("Running SHAP Analysis")
        
    #     # Get feature names (excluding target variable)
    #     feature_names = [col for col in data['X_train'].columns]
    #     # Initialize SHAP analyzer
    #     shap_analyzer = ShapAnalyzer(
    #         model_path=os.path.join(tracker.experiment_dir, "model.json"),
    #         feature_names=feature_names,
    #         output_dir=tracker.experiment_dir
    #     )
        
    #     # Run SHAP analysis on validation data
    #     shap_values = shap_analyzer.analyze(data['X_val'])
        
    #     # If time-specific models were trained, analyze them too
    #     if args.train_time_specific:
    #         models_dir = os.path.join(tracker.experiment_dir, 'time_specific_models')
    #         for period in ['early', 'mid', 'recent']:
    #             print(f"\nSHAP Analysis for {period.upper()} period model")
    #             period_analyzer = ShapAnalyzer(
    #                 model_path=os.path.join(models_dir, f"{period}_model.json"),
    #                 feature_names=feature_names,
    #                 output_dir=os.path.join(models_dir, f"{period}_shap")
    #             )
    #             # Use the corresponding time period data
    #             period_data = time_periods_data[period]
    #             X_period = period_data[features]
    #             period_analyzer.analyze(X_period)




    # Save model
    model_path = os.path.join(tracker.experiment_dir, "model.json")
    trainer.save_model(model_path)
    
    if args.train_time_specific:
        time_specific_results = train_time_specific_models(
        df, features, tracker.experiment_dir
    )
        
    
    if args.run_shap_analysis:
        
        print_section("Running SHAP Analysis")
        
        # Get feature names (excluding target variable)
        feature_names = [col for col in data['X_train'].columns]
        
        # Initialize SHAP analyzer
        shap_analyzer = ShapAnalyzer(
            model_path=os.path.join(tracker.experiment_dir, "model.json"),
            feature_names=feature_names,
            output_dir=tracker.experiment_dir
        )
        
        # Run SHAP analysis on validation data
        shap_values = shap_analyzer.analyze(data['X_val'])
        
        if args.train_time_specific:
            models_dir = os.path.join(tracker.experiment_dir, 'time_specific_models')
            for period in ['early', 'mid', 'recent']:
                print(f"\nSHAP Analysis for {period.upper()} period model")
                period_analyzer = ShapAnalyzer(
                    model_path=os.path.join(models_dir, f"{period}_model.json"),
                    feature_names=feature_names,
                    output_dir=os.path.join(models_dir, f"{period}_shap")
                )
                
                period_data = time_periods[period]
                X_period = period_data[features]
                period_analyzer.analyze(X_period)
        
        tracker.log_parameters({"time_specific_results": time_specific_results})
        # Save experiment data
        tracker.save_experiment()
        
        print_section("Experiment Completed")
        print(f"Results saved to {tracker.experiment_dir}")


def run_feature_selection_experiment(df, tracker):
    """Run experiments with different feature groups to find most generalizable ones."""
    print_section("Feature Selection Experiment")
    
    # Define feature groups
    feature_groups = {
        'all': None,
        'no_econ': ['econ_vc'],  # Exclude economic indicators
        'team_only': ['temporal', 'location', 'econ_vc', 'lda_topics'],  # Only team features
        'team_topics': ['temporal', 'location', 'econ_vc'],  # Team + topics
        'core': ['econ_vc', 'temporal']  # Team + topics + location
    }
    
    results = {}
    
    for name, exclude in feature_groups.items():
        print(f"\nRunning experiment: {name}")
        
        # Initialize processor
        processor = DataProcessor(df)
        
        # Filter features
        if name == 'all':
            features = processor.filter_features()
        else:
            features = processor.filter_features(exclude_groups=exclude)
        
        # Split data
        data_splits = processor.split_time_based(
            'founded_year', 
            args.train_years, 
            args.val_years, 
            args.test_years
        )
        
        # Prepare data
        data = processor.prepare_data_splits(data_splits, features)
        
        # Train and evaluate model
        trainer = ModelTrainer(use_gpu=args.use_gpu)
        best_params, model, _ = trainer.optimize_hyperparameters(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            max_evals=30
        )
        
        # Evaluate on test set (recent data)
        evaluator = ModelEvaluator(model, features)
        metrics = evaluator.evaluate(data['X_test'], data['y_test'])
        
        # Store results
        results[name] = {
            'test_metrics': metrics,
            'feature_count': len(features)
        }
        
        # Save model
        model_path = os.path.join(tracker.experiment_dir, f"model_{name}.json")
        trainer.save_model(model_path)
    
    # Log comparative results
    tracker.log_parameters({"feature_experiments": results})
    
    # Create comparison visualization
    plot_feature_experiment_comparison(results, tracker.experiment_dir)
    
    return results

if __name__ == "__main__":
    main()