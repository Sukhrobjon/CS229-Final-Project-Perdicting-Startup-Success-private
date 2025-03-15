# model_trainer.py

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
import time
from utils import print_section, save_object

class ModelTrainer:
    def __init__(self, use_gpu=False):
        """
        Initialize model trainer.
        
        Args:
            use_gpu: Whether to use GPU for training
        """
        self.use_gpu = use_gpu
        self.best_params = None
        self.best_model = None
        print(f"ModelTrainer initialized. GPU: {'Enabled' if use_gpu else 'Disabled'}")
    
    def define_parameter_space(self):
        """
        Define hyperparameter search space for Bayesian optimization.
        
        Returns:
            dict: Hyperparameter search space
        """
        space = {
            'max_depth': scope.int(hp.quniform('max_depth', 3, 12, 1)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),
            'gamma': hp.uniform('gamma', 0, 1),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-10), np.log(1)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-10), np.log(1)),
            'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 10)
        }
        return space
    
    def objective(self, params, X_train, y_train, X_val, y_val):
        """
        Objective function for hyperparameter optimization.
        
        Args:
            params: Hyperparameters to evaluate
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            dict: Results including loss and status
        """
        # Ensure max_depth is an integer
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        # Set fixed parameters
        fixed_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'verbosity': 0,
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu'
        }
        
        # Combine parameters
        full_params = {**params, **fixed_params}
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model with early stopping
        evals_result = {}
        model = xgb.train(
            full_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False
        )
        
        # Get best score
        best_score = model.best_score
        best_iteration = model.best_iteration
        
        # Calculate additional metrics
        y_pred_proba = model.predict(dval)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Return result (negative AUC for minimization)
        return {
            'loss': -auc,
            'status': STATUS_OK,
            'model': model,
            'params': params,
            'best_iteration': best_iteration,
            'metrics': {
                'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, max_evals=50):
        """
        Perform Bayesian hyperparameter optimization.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            max_evals: Maximum number of evaluations
            
        Returns:
            dict: Best parameters
        """
        print_section("Bayesian Hyperparameter Optimization")
        print(f"Running optimization with {max_evals} trials")
        start_time = time.time()
        
        # Define the parameter space
        space = self.define_parameter_space()
        
        # Initialize trials object to store results
        trials = Trials()
        
        # Define objective function
        def objective_wrapper(params):
            return self.objective(params, X_train, y_train, X_val, y_val)
        
        # Run optimization
        best = fmin(
            fn=objective_wrapper,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=1
        )
        
        # Get all results
        results = [trial['result'] for trial in trials.trials]
        
        # Find best result (with lowest loss)
        best_result = min(results, key=lambda x: x['loss'])
        self.best_params = best_result['params']
        
        # Store best model
        self.best_model = best_result['model']
        
        # Print optimization results
        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best AUC: {-best_result['loss']:.4f}")
        print(f"Best iteration: {best_result['best_iteration']}")
        print("Best hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Print best metrics
        print("\nValidation metrics with best parameters:")
        for metric, value in best_result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        return self.best_params, self.best_model, best_result['metrics']
    

    

    def train_final_model(self, X_train, y_train, X_val, y_val, params=None):
        """
        Train final model with best parameters.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            params: Model parameters, if None use self.best_params
            
        Returns:
            xgb.Booster: Trained model
        """
        print_section("Training Final Model")
        
        # Use best parameters if not provided
        if params is None:
            if self.best_params is None:
                raise ValueError("No parameters provided and no best_params found. Run optimize_hyperparameters first.")
            params = self.best_params
        
        # Print parameters
        print("Model parameters:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        # Ensure integer parameters
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        # Set fixed parameters
        fixed_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'verbosity': 0,
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu'
        }
        
        # Combine parameters
        full_params = {**params, **fixed_params}
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model with early stopping
        start_time = time.time()
        evals_result = {}
        model = xgb.train(
            full_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=100
        )
        
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best validation AUC: {model.best_score:.4f}")
        
        # Store the model
        self.best_model = model
        
        return model
    
    def kfold_cv(self, X, y, params, k=5, stratified=True):
        """
        Perform k-fold cross-validation.
        
        Args:
            X, y: Data for cross-validation
            params: Model parameters
            k: Number of folds
            stratified: Use stratified k-fold if True
            
        Returns:
            dict: Cross-validation results
        """
        print_section(f"{k}-Fold Cross-Validation")
        
        # Initialize k-fold
        if stratified:
            kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            print(f"Using Stratified {k}-Fold CV")
        else:
            kfold = KFold(n_splits=k, shuffle=True, random_state=42)
            print(f"Using {k}-Fold CV")
        
        # Ensure integer parameters
        params = params.copy()
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        # Set fixed parameters
        fixed_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'verbosity': 0,
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu'
        }
        
        # Combine parameters
        full_params = {**params, **fixed_params}
        
        # Initialize results storage
        cv_results = {
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        # Perform k-fold CV
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\nFold {fold+1}/{k}:")
            
            # Split data
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create DMatrix objects
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
            
            # Train model
            evals_result = {}
            model = xgb.train(
                full_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                evals_result=evals_result,
                verbose_eval=False
            )
            
            # Evaluate model
            y_pred_proba = model.predict(dval)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            fold_metrics = {
                'auc': roc_auc_score(y_val_fold, y_pred_proba),
                'accuracy': accuracy_score(y_val_fold, y_pred),
                'precision': precision_score(y_val_fold, y_pred),
                'recall': recall_score(y_val_fold, y_pred),
                'f1': f1_score(y_val_fold, y_pred)
            }
            
            # Print fold metrics
            print(f"  AUC: {fold_metrics['auc']:.4f}")
            print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
            print(f"  Precision: {fold_metrics['precision']:.4f}")
            print(f"  Recall: {fold_metrics['recall']:.4f}")
            print(f"  F1: {fold_metrics['f1']:.4f}")
            
            # Store fold results
            cv_results['fold_metrics'].append(fold_metrics)
        
        # Calculate mean and std for metrics
        metrics_df = pd.DataFrame(cv_results['fold_metrics'])
        cv_results['mean_metrics'] = metrics_df.mean().to_dict()
        cv_results['std_metrics'] = metrics_df.std().to_dict()
        
        # Print overall results
        print("\nCross-Validation Results:")
        for metric in cv_results['mean_metrics']:
            mean = cv_results['mean_metrics'][metric]
            std = cv_results['std_metrics'][metric]
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
        
        return cv_results
    
    def save_model(self, model_path):
        """
        Save trained model to file.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            None
        """
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        self.best_model.save_model(model_path)
        print(f"Model saved to {model_path}")
    

    def train_time_ensemble(self, time_models, X, time_column):
        """
        Train an ensemble model that uses different models based on time period.
        
        Args:
            time_models: Dictionary of models for different time periods
            X: Features DataFrame
            time_column: Column name with year or time period
            
        Returns:
            callable: Ensemble prediction function
        """
        print_section("Creating Time-Based Ensemble Model")
        
        # Define which model to use for which years
        time_mapping = {
            'early': [2012, 2013, 2014],
            'mid': [2015, 2016, 2017, 2018],
            'recent': [2019, 2020, 2021, 2022]
        }
        
        def ensemble_predict(X_pred):
            """Make predictions using the appropriate time-specific model."""
            # Create DMatrix for prediction
            dmatrix = xgb.DMatrix(X_pred)
            
            # Get time period for each row
            if 'time_period' in X_pred.columns:
                # If we have a specific time_period column
                time_periods = X_pred['time_period'].values
            else:
                # Map years to period names
                years = X_pred[time_column].values
                time_periods = np.array(['recent'] * len(years), dtype=object)
                for period, year_list in time_mapping.items():
                    mask = np.isin(years, year_list)
                    time_periods[mask] = period
            
            # Make predictions using appropriate model for each row
            predictions = np.zeros(len(X_pred))
            for period in ['early', 'mid', 'recent']:
                mask = time_periods == period
                if np.any(mask):
                    predictions[mask] = time_models[period].predict(dmatrix.slice(np.where(mask)[0]))
            
            return predictions
        
        print("Ensemble model created successfully")
        return ensemble_predict


    def get_feature_importance(self, feature_names):
        """
        Get feature importance from trained model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            dict: Feature importance dictionary
        """
        if self.best_model is None:
            raise ValueError("No model available. Train a model first.")
        
        # Get feature importance scores
        importance_scores = self.best_model.get_score(importance_type='gain')
        
        # Create full feature importance dictionary with zeros for missing features
        feature_importance = {feature: 0 for feature in feature_names}
        
        # Update with actual importance scores
        for feature, score in importance_scores.items():
            if feature in feature_importance:
                feature_importance[feature] = score
        
        return feature_importance