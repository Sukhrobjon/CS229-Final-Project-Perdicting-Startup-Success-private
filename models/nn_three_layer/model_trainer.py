import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input,
    Activation, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score
)
from sklearn.model_selection import KFold, StratifiedKFold
import kerastuner as kt
from utils import print_section, ensure_dir
import time
import os

class ModelTrainer:
    def __init__(self, use_gpu=False):
        """
        Initialize model trainer.
        
        Args:
            use_gpu: Whether to use GPU for training
        """
        self.use_gpu = use_gpu
        self.best_model = None
        self.best_params = None
        
        # Set up GPU configuration
        if self.use_gpu:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU acceleration enabled. Found {len(gpus)} GPUs.")
                except RuntimeError as e:
                    print(f"Error setting GPU memory growth: {e}")
            else:
                print("No GPUs found. Using CPU despite use_gpu=True.")
        else:
            print("GPU acceleration disabled. Using CPU.")
    
    def build_model(self, input_dim, params=None):
        """
        Build neural network model.
        
        Args:
            input_dim: Input dimension (feature count)
            params: Model parameters, if None use defaults
            
        Returns:
            tf.keras.Model: Neural network model
        """
        # Default parameters
        default_params = {
            'units_1': 128,
            'units_2': 64,
            'units_3': 32,
            'dropout_1': 0.3,
            'dropout_2': 0.3,
            'dropout_3': 0.3,
            'learning_rate': 0.001,
            'activation': 'relu',
            'batch_norm': True
        }
        
        # Use provided parameters if available
        if params is None:
            params = default_params
        else:
            # Fill missing parameters with defaults
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
        # Define activation function
        if params['activation'] == 'leaky_relu':
            activation = LeakyReLU(alpha=0.1)
        else:
            activation = params['activation']
        
        # Define model architecture
        inputs = Input(shape=(input_dim,), name='input')
        
        # First hidden layer
        x = Dense(params['units_1'], name='dense_1')(inputs)
        if params['batch_norm']:
            x = BatchNormalization(name='bn_1')(x)
        if params['activation'] == 'leaky_relu':
            x = LeakyReLU(alpha=0.1, name='leaky_relu_1')(x)
        else:
            x = Activation(activation, name='activation_1')(x)
        x = Dropout(params['dropout_1'], name='dropout_1')(x)
        
        # Second hidden layer
        x = Dense(params['units_2'], name='dense_2')(x)
        if params['batch_norm']:
            x = BatchNormalization(name='bn_2')(x)
        if params['activation'] == 'leaky_relu':
            x = LeakyReLU(alpha=0.1, name='leaky_relu_2')(x)
        else:
            x = Activation(activation, name='activation_2')(x)
        x = Dropout(params['dropout_2'], name='dropout_2')(x)
        
        # Third hidden layer
        x = Dense(params['units_3'], name='dense_3')(x)
        if params['batch_norm']:
            x = BatchNormalization(name='bn_3')(x)
        if params['activation'] == 'leaky_relu':
            x = LeakyReLU(alpha=0.1, name='leaky_relu_3')(x)
        else:
            x = Activation(activation, name='activation_3')(x)
        x = Dropout(params['dropout_3'], name='dropout_3')(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                AUC(name='auc'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        )
        
        return model
    
    def create_callbacks(self, output_dir, patience=20):
        """
        Create training callbacks.
        
        Args:
            output_dir: Directory to save outputs
            patience: Patience for early stopping
            
        Returns:
            list: Callbacks
        """
        callbacks = [
            # Early stopping on validation loss
            EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=patience,
                verbose=1,
                restore_best_weights=True
            ),
            
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=os.path.join(output_dir, 'best_model.h5'),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                verbose=1,
                min_lr=1e-6
            )
        ]
        
        return callbacks
    
    def train_model(self, X_train, y_train, X_val, y_val, params=None, output_dir='output', 
                  batch_size=64, epochs=100, class_weights=None):
        """
        Train neural network model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            params: Model parameters
            output_dir: Directory to save outputs
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            class_weights: Class weights for imbalanced data
            
        Returns:
            tuple: (trained model, training history)
        """
        print_section("Training Neural Network Model")
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Get input dimension
        input_dim = X_train.shape[1]
        
        # Build model
        model = self.build_model(input_dim, params)
        
        # Print model summary
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks(output_dir)
        
        # Train model
        print(f"Training model with batch_size={batch_size}, epochs={epochs}")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2
        )
        
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        # Store best model
        self.best_model = model
        
        return model, history
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, 
                             max_trials=30, output_dir='tuner', class_weights=None):
        """
        Perform hyperparameter tuning using KerasTuner.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            max_trials: Maximum number of trials
            output_dir: Directory to save tuner outputs
            class_weights: Class weights for imbalanced data
            
        Returns:
            dict: Best hyperparameters
        """
        print_section(f"Hyperparameter Tuning (Max Trials: {max_trials})")
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        input_dim = X_train.shape[1]
        
        # Define the model-building function
        def build_model_tuner(hp):
            # Define search space
            units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32)
            units_2 = hp.Int('units_2', min_value=16, max_value=128, step=16)
            units_3 = hp.Int('units_3', min_value=8, max_value=64, step=8)
            
            dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
            dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
            dropout_3 = hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)
            
            learning_rate = hp.Choice('learning_rate', 
                                     values=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
            
            activation = hp.Choice('activation', 
                                  values=['relu', 'leaky_relu', 'elu', 'tanh'])
            
            batch_norm = hp.Boolean('batch_norm')
            
            # Build model using hyperparameters
            params = {
                'units_1': units_1,
                'units_2': units_2,
                'units_3': units_3,
                'dropout_1': dropout_1,
                'dropout_2': dropout_2,
                'dropout_3': dropout_3,
                'learning_rate': learning_rate,
                'activation': activation,
                'batch_norm': batch_norm
            }
            
            return self.build_model(input_dim, params)
        
        # Create tuner
        tuner = kt.Hyperband(
            build_model_tuner,
            objective=kt.Objective('val_auc', direction='max'),
            max_epochs=100,
            factor=3,
            directory=output_dir,
            project_name='startup_success_nn'
        )
        
        # Create early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=15,
            restore_best_weights=True
        )
        
        # Execute hyperparameter search
        print("Starting hyperparameter search...")
        start_time = time.time()
        
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose=1
        )
        
        tuning_time = time.time() - start_time
        print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters(1)[0]
        
        # Convert to dict
        self.best_params = {
            'units_1': best_hp.get('units_1'),
            'units_2': best_hp.get('units_2'),
            'units_3': best_hp.get('units_3'),
            'dropout_1': best_hp.get('dropout_1'),
            'dropout_2': best_hp.get('dropout_2'),
            'dropout_3': best_hp.get('dropout_3'),
            'learning_rate': best_hp.get('learning_rate'),
            'activation': best_hp.get('activation'),
            'batch_norm': best_hp.get('batch_norm')
        }
        
        # Print best hyperparameters
        print("Best hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Build best model
        best_model = tuner.hypermodel.build(best_hp)
        self.best_model = best_model
        
        return self.best_params
    
    def kfold_cv(self, X, y, params, k=5, stratified=True, batch_size=64, epochs=100, class_weights=None):
        """
        Perform k-fold cross-validation.
        
        Args:
            X, y: Data for cross-validation
            params: Model parameters
            k: Number of folds
            stratified: Use stratified k-fold if True
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            class_weights: Class weights for imbalanced data
            
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
        
        # Initialize results storage
        cv_results = {
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        # Get input dimension
        input_dim = X.shape[1]
        
        # Perform k-fold CV
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\nFold {fold+1}/{k}:")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Build model
            model = self.build_model(input_dim, params)
            
            # Create early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=15,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                class_weight=class_weights,
                verbose=0
            )
            
            # Get best epoch's validation metrics
            best_epoch = np.argmax(history.history['val_auc'])
            
            # Evaluate model
            y_pred_proba = model.predict(X_val_fold).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            fold_metrics = {
                'auc': roc_auc_score(y_val_fold, y_pred_proba),
                'accuracy': accuracy_score(y_val_fold, y_pred),
                'precision': precision_score(y_val_fold, y_pred),
                'recall': recall_score(y_val_fold, y_pred),
                'f1': f1_score(y_val_fold, y_pred),
                'best_epoch': best_epoch + 1  # +1 because epochs are 0-indexed
            }
            
            # Print fold metrics
            print(f"  AUC: {fold_metrics['auc']:.4f}")
            print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
            print(f"  Precision: {fold_metrics['precision']:.4f}")
            print(f"  Recall: {fold_metrics['recall']:.4f}")
            print(f"  F1: {fold_metrics['f1']:.4f}")
            print(f"  Best epoch: {fold_metrics['best_epoch']}")
            
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
        
        self.best_model.save(model_path)
        print(f"Model saved to {model_path}")
        
   
        model_json = self.best_model.to_json()
        json_path = model_path.replace('.h5', '.json')
        with open(json_path, 'w') as f:
            f.write(model_json)
        print(f"Model architecture saved to {json_path}")
    
    def load_model(self, model_path):
        """
        Load model from file.
        
        Args:
            model_path: Path to load the model from
            
        Returns:
            tf.keras.Model: Loaded model
        """
        self.best_model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return self.best_model