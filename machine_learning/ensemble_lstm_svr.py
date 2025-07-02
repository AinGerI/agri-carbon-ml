#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM-SVR Ensemble Model for Agricultural Time Series Prediction

This module provides a comprehensive ensemble approach that combines:
- LSTM neural networks for capturing temporal dependencies
- Support Vector Regression for robust prediction using LSTM outputs as features
- Advanced data preprocessing and feature engineering
- Comprehensive evaluation and visualization capabilities
- GUI interface for easy usage

The ensemble works in two stages:
1. Train LSTM model on time series data to capture temporal patterns
2. Use LSTM predictions as features for SVR to make final predictions

Combines functionality from:
- ensemble_lstm_gasvm.py
- LSTM.py
- svr.py

Author: Thesis Research Project
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

# TensorFlow/Keras imports with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM functionality will be limited.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib for Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class LSTMSVREnsemble:
    """
    LSTM-SVR Ensemble Model for Time Series Prediction.
    
    Combines LSTM neural networks and Support Vector Regression in a two-stage
    ensemble approach for improved time series forecasting accuracy.
    """
    
    def __init__(self, output_folder=None):
        """Initialize ensemble model with default parameters."""
        # Output folder setup
        if output_folder is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = f"LSTM_SVR_Ensemble_{timestamp}"
        else:
            self.output_folder = output_folder
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        # LSTM parameters
        self.lstm_params = {
            'sequence_length': 3,  # Number of time steps to look back
            'lstm_units': [50, 50],  # LSTM layer units
            'dropout_rate': 0.2,  # Dropout rate for regularization
            'learning_rate': 0.001,  # Learning rate for optimizer
            'epochs': 50,  # Maximum training epochs
            'batch_size': 32,  # Batch size for training
            'validation_split': 0.2,  # Fraction of data for validation
            'patience': 10,  # Early stopping patience
            'train_split': 0.7,  # Training data proportion
            'val_split': 0.15  # Validation data proportion (rest is test)
        }
        
        # SVR parameters
        self.svr_params = {
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['rbf', 'linear']
            },
            'cv_folds': 5,
            'scoring': 'neg_mean_squared_error'
        }
        
        # Model components
        self.lstm_model = None
        self.svr_model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.use_feature_scaling = True
        self.use_target_scaling = False
        
        # Data storage
        self.original_data = None
        self.processed_data = None
        self.feature_columns = None
        self.target_column = None
        
        # Results storage
        self.lstm_results = None
        self.svr_results = None
        self.ensemble_results = None
        self.evaluation_results = None
    
    def load_data(self, file_path, sheet_name=None, target_column=None):
        """
        Load and preprocess data from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet to load
            target_column: Name of target column (uses last column if None)
            
        Returns:
            bool: Success status
        """
        try:
            # Read Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            print(f"Loaded data with shape: {df.shape}")
            
            # Identify columns
            if target_column and target_column in df.columns:
                self.target_column = target_column
                self.feature_columns = [col for col in df.columns if col != target_column]
            else:
                # Assume first column is time/index, last column is target
                self.feature_columns = df.columns[1:-1].tolist()
                self.target_column = df.columns[-1]
            
            print(f"Feature columns: {self.feature_columns}")
            print(f"Target column: {self.target_column}")
            
            # Convert to numeric and handle missing values
            numeric_columns = self.feature_columns + [self.target_column]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing values
            df = df.dropna(subset=numeric_columns)
            
            if df.empty:
                raise ValueError("No valid data remaining after preprocessing")
            
            self.original_data = df
            print(f"After preprocessing: {df.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_sequences(self, data, sequence_length):
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input data (features + target)
            sequence_length: Length of input sequences
            
        Returns:
            tuple: (X, y) sequences
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values.astype('float32')
        else:
            data_array = data.astype('float32')
        
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        
        if len(data_array) <= sequence_length:
            print(f"Warning: Data length ({len(data_array)}) <= sequence length ({sequence_length})")
            return np.array([]), np.array([])
        
        X, y = [], []
        for i in range(len(data_array) - sequence_length):
            X.append(data_array[i:i + sequence_length, :-1])  # All features
            y.append(data_array[i + sequence_length, -1])     # Target value
        
        return np.array(X), np.array(y)
    
    def train_lstm_stage(self, progress_callback=None):
        """
        Stage 1: Train LSTM model for temporal pattern learning.
        
        Args:
            progress_callback: Function to report progress
            
        Returns:
            dict: LSTM training results
        """
        if self.original_data is None:
            print("No data loaded. Please load data first.")
            return None
        
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot train LSTM model.")
            return None
        
        if progress_callback:
            progress_callback("Starting LSTM training...")
        
        print("\n===== Stage 1: Training LSTM Model =====")
        
        try:
            # Prepare data for LSTM
            features_and_target = self.feature_columns + [self.target_column]
            data_selected = self.original_data[features_and_target].copy()
            
            # Create sequences
            print("Creating time series sequences...")
            X, y = self.create_sequences(data_selected, self.lstm_params['sequence_length'])
            
            if len(X) == 0:
                raise ValueError(f"Cannot create sequences with data length {len(data_selected)} and sequence length {self.lstm_params['sequence_length']}")
            
            print(f"Created {len(X)} sequences")
            
            # Split data
            total_samples = len(X)
            train_split_idx = int(total_samples * self.lstm_params['train_split'])
            val_split_idx = train_split_idx + int(total_samples * self.lstm_params['val_split'])
            
            X_train = X[:train_split_idx]
            y_train = y[:train_split_idx]
            X_val = X[train_split_idx:val_split_idx]
            y_val = y[train_split_idx:val_split_idx]
            X_test = X[val_split_idx:]
            y_test = y[val_split_idx:]
            
            print(f"Training set: {len(X_train)} samples")
            print(f"Validation set: {len(X_val)} samples")
            print(f"Test set: {len(X_test)} samples")
            
            # Build LSTM model
            print("Building LSTM architecture...")
            num_features = X.shape[2]
            
            self.lstm_model = Sequential([
                LSTM(self.lstm_params['lstm_units'][0], return_sequences=True, 
                     input_shape=(self.lstm_params['sequence_length'], num_features)),
                Dropout(self.lstm_params['dropout_rate']),
                LSTM(self.lstm_params['lstm_units'][1]),
                Dropout(self.lstm_params['dropout_rate']),
                Dense(1)
            ])
            
            self.lstm_model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=self.lstm_params['learning_rate']),
                metrics=['mae']
            )
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.lstm_params['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            print("Training LSTM model...")
            if progress_callback:
                progress_callback("Training LSTM model...")
            
            history = self.lstm_model.fit(
                X_train, y_train,
                epochs=self.lstm_params['epochs'],
                batch_size=self.lstm_params['batch_size'],
                validation_data=(X_val, y_val) if len(X_val) > 0 else None,
                callbacks=callbacks,
                verbose=1
            )
            
            # Generate predictions
            print("Generating LSTM predictions...")
            train_pred = self.lstm_model.predict(X_train).flatten()
            val_pred = self.lstm_model.predict(X_val).flatten() if len(X_val) > 0 else np.array([])
            test_pred = self.lstm_model.predict(X_test).flatten() if len(X_test) > 0 else np.array([])
            
            # Calculate performance metrics
            train_metrics = self._calculate_metrics(y_train, train_pred)
            val_metrics = self._calculate_metrics(y_val, val_pred) if len(y_val) > 0 else None
            test_metrics = self._calculate_metrics(y_test, test_pred) if len(y_test) > 0 else None
            
            # Store LSTM results
            self.lstm_results = {
                'train': {'actual': y_train, 'predicted': train_pred, 'metrics': train_metrics},
                'validation': {'actual': y_val, 'predicted': val_pred, 'metrics': val_metrics},
                'test': {'actual': y_test, 'predicted': test_pred, 'metrics': test_metrics},
                'history': history.history,
                'sequence_data': {
                    'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                    'y_train': y_train, 'y_val': y_val, 'y_test': y_test
                }
            }
            
            # Print performance
            print("\nLSTM Model Performance:")
            print(f"Training: RMSE = {train_metrics['rmse']:.4f}, MAE = {train_metrics['mae']:.4f}, R² = {train_metrics['r2']:.4f}")
            if val_metrics:
                print(f"Validation: RMSE = {val_metrics['rmse']:.4f}, MAE = {val_metrics['mae']:.4f}, R² = {val_metrics['r2']:.4f}")
            if test_metrics:
                print(f"Test: RMSE = {test_metrics['rmse']:.4f}, MAE = {test_metrics['mae']:.4f}, R² = {test_metrics['r2']:.4f}")
            
            if progress_callback:
                progress_callback("LSTM training completed")
            
            return self.lstm_results
            
        except Exception as e:
            print(f"Error during LSTM training: {e}")
            return None
    
    def prepare_svr_data(self):
        """
        Prepare data for SVR training using LSTM predictions as features.
        
        Returns:
            dict: Prepared SVR training data
        """
        if self.lstm_results is None:
            print("LSTM model not trained. Please train LSTM first.")
            return None
        
        print("\n===== Preparing SVR Training Data =====")
        
        try:
            # Combine all LSTM predictions
            all_lstm_preds = np.concatenate([
                self.lstm_results['train']['predicted'],
                self.lstm_results['validation']['predicted'],
                self.lstm_results['test']['predicted']
            ])
            
            all_actual = np.concatenate([
                self.lstm_results['train']['actual'],
                self.lstm_results['validation']['actual'],
                self.lstm_results['test']['actual']
            ])
            
            # Get original features (skip first sequence_length rows)
            skip_rows = self.lstm_params['sequence_length']
            original_features = self.original_data[self.feature_columns].iloc[skip_rows:skip_rows + len(all_lstm_preds)].values
            
            # Combine original features with LSTM predictions
            lstm_feature = all_lstm_preds.reshape(-1, 1)
            combined_features = np.hstack([original_features, lstm_feature])
            
            # Create feature names
            svr_feature_names = self.feature_columns + ['LSTM_Prediction']
            
            # Add lagged target as feature
            lagged_target = np.zeros((len(all_actual), 1))
            lagged_target[1:] = all_actual[:-1].reshape(-1, 1)
            lagged_target[0] = all_actual[0]  # First value uses itself
            
            combined_features = np.hstack([combined_features, lagged_target])
            svr_feature_names.append('Target_Lag_1')
            
            # Split data using same proportions as LSTM
            total_samples = len(combined_features)
            train_size = len(self.lstm_results['train']['actual'])
            val_size = len(self.lstm_results['validation']['actual'])
            
            X_train = combined_features[:train_size]
            y_train = all_actual[:train_size]
            X_val = combined_features[train_size:train_size + val_size]
            y_val = all_actual[train_size:train_size + val_size]
            X_test = combined_features[train_size + val_size:]
            y_test = all_actual[train_size + val_size:]
            
            self.processed_data = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                'feature_names': svr_feature_names,
                'all_features': combined_features,
                'all_targets': all_actual
            }
            
            print(f"SVR data prepared: {total_samples} samples, {combined_features.shape[1]} features")
            print(f"Feature names: {svr_feature_names}")
            
            return self.processed_data
            
        except Exception as e:
            print(f"Error preparing SVR data: {e}")
            return None
    
    def train_svr_stage(self, progress_callback=None):
        """
        Stage 2: Train SVR model using LSTM predictions as features.
        
        Args:
            progress_callback: Function to report progress
            
        Returns:
            dict: SVR training results
        """
        if self.processed_data is None:
            if not self.prepare_svr_data():
                return None
        
        if progress_callback:
            progress_callback("Starting SVR training...")
        
        print("\n===== Stage 2: Training SVR Model =====")
        
        try:
            # Get training data
            X_train = self.processed_data['X_train']
            y_train = self.processed_data['y_train']
            X_val = self.processed_data['X_val']
            y_val = self.processed_data['y_val']
            X_test = self.processed_data['X_test']
            y_test = self.processed_data['y_test']
            
            # Combine training and validation for SVR optimization
            X_train_full = np.vstack([X_train, X_val]) if len(X_val) > 0 else X_train
            y_train_full = np.concatenate([y_train, y_val]) if len(y_val) > 0 else y_train
            
            print(f"SVR training data: {len(X_train_full)} samples, {X_train_full.shape[1]} features")
            
            # Create pipeline with optional scaling
            pipeline_steps = []
            
            if self.use_feature_scaling:
                pipeline_steps.append(('scaler', StandardScaler()))
            
            pipeline_steps.append(('svr', SVR()))
            
            # Create parameter grid with pipeline prefixes
            param_grid = {}
            for key, value in self.svr_params['param_grid'].items():
                param_grid[f'svr__{key}'] = value
            
            # Create pipeline
            pipeline = Pipeline(pipeline_steps)
            
            # Setup grid search with time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.svr_params['cv_folds'])
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=tscv,
                scoring=self.svr_params['scoring'],
                n_jobs=-1,
                verbose=1
            )
            
            # Train SVR
            print("Optimizing SVR parameters...")
            if progress_callback:
                progress_callback("Optimizing SVR parameters...")
            
            grid_search.fit(X_train_full, y_train_full)
            
            # Get best model
            self.svr_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"Best SVR parameters: {best_params}")
            print(f"Best cross-validation score: {best_score:.6f}")
            
            # Generate predictions
            train_pred = self.svr_model.predict(X_train)
            val_pred = self.svr_model.predict(X_val) if len(X_val) > 0 else np.array([])
            test_pred = self.svr_model.predict(X_test) if len(X_test) > 0 else np.array([])
            
            # Calculate performance metrics
            train_metrics = self._calculate_metrics(y_train, train_pred)
            val_metrics = self._calculate_metrics(y_val, val_pred) if len(y_val) > 0 else None
            test_metrics = self._calculate_metrics(y_test, test_pred) if len(y_test) > 0 else None
            
            # Store SVR results
            self.svr_results = {
                'train': {'actual': y_train, 'predicted': train_pred, 'metrics': train_metrics},
                'validation': {'actual': y_val, 'predicted': val_pred, 'metrics': val_metrics},
                'test': {'actual': y_test, 'predicted': test_pred, 'metrics': test_metrics},
                'best_params': best_params,
                'best_score': best_score,
                'feature_importance': self._get_feature_importance()
            }
            
            # Print performance
            print("\nSVR Model Performance:")
            print(f"Training: RMSE = {train_metrics['rmse']:.4f}, MAE = {train_metrics['mae']:.4f}, R² = {train_metrics['r2']:.4f}")
            if val_metrics:
                print(f"Validation: RMSE = {val_metrics['rmse']:.4f}, MAE = {val_metrics['mae']:.4f}, R² = {val_metrics['r2']:.4f}")
            if test_metrics:
                print(f"Test: RMSE = {test_metrics['rmse']:.4f}, MAE = {test_metrics['mae']:.4f}, R² = {test_metrics['r2']:.4f}")
            
            if progress_callback:
                progress_callback("SVR training completed")
            
            return self.svr_results
            
        except Exception as e:
            print(f"Error during SVR training: {e}")
            return None
    
    def train_ensemble(self, progress_callback=None):
        """
        Train complete ensemble model (LSTM + SVR).
        
        Args:
            progress_callback: Function to report progress
            
        Returns:
            dict: Complete ensemble results
        """
        print("Training LSTM-SVR Ensemble Model")
        print("=" * 50)
        
        # Stage 1: Train LSTM
        lstm_results = self.train_lstm_stage(progress_callback)
        if lstm_results is None:
            print("LSTM training failed. Cannot continue.")
            return None
        
        # Stage 2: Train SVR
        svr_results = self.train_svr_stage(progress_callback)
        if svr_results is None:
            print("SVR training failed.")
            return None
        
        # Compare ensemble vs individual models
        self._evaluate_ensemble_performance()
        
        if progress_callback:
            progress_callback("Ensemble training completed")
        
        return {
            'lstm_results': lstm_results,
            'svr_results': svr_results,
            'ensemble_evaluation': self.evaluation_results
        }
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics."""
        if len(y_true) == 0 or len(y_pred) == 0:
            return None
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (avoiding division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.inf
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def _get_feature_importance(self):
        """Get feature importance from SVR model."""
        if self.svr_model is None:
            return None
        
        try:
            # For linear kernel, we can get coefficients
            if hasattr(self.svr_model.named_steps['svr'], 'coef_'):
                importance = np.abs(self.svr_model.named_steps['svr'].coef_[0])
                return dict(zip(self.processed_data['feature_names'], importance))
            else:
                # For non-linear kernels, feature importance is not directly available
                return None
        except:
            return None
    
    def _evaluate_ensemble_performance(self):
        """Compare LSTM vs SVR vs ensemble performance."""
        if self.lstm_results is None or self.svr_results is None:
            return None
        
        print("\n" + "=" * 50)
        print("ENSEMBLE PERFORMANCE COMPARISON")
        print("=" * 50)
        
        # Compare test set performance
        lstm_test_metrics = self.lstm_results['test']['metrics']
        svr_test_metrics = self.svr_results['test']['metrics']
        
        if lstm_test_metrics and svr_test_metrics:
            print("\nTest Set Performance Comparison:")
            print(f"{'Metric':<10} {'LSTM':<10} {'SVR':<10} {'Improvement':<12}")
            print("-" * 45)
            
            metrics = ['rmse', 'mae', 'r2', 'mape']
            improvements = {}
            
            for metric in metrics:
                lstm_val = lstm_test_metrics[metric]
                svr_val = svr_test_metrics[metric]
                
                if metric == 'r2':
                    improvement = ((svr_val - lstm_val) / lstm_val * 100) if lstm_val != 0 else 0
                else:
                    improvement = ((lstm_val - svr_val) / lstm_val * 100) if lstm_val != 0 else 0
                
                improvements[metric] = improvement
                
                print(f"{metric.upper():<10} {lstm_val:<10.4f} {svr_val:<10.4f} {improvement:>+10.2f}%")
            
            # Determine best model
            if svr_test_metrics['rmse'] < lstm_test_metrics['rmse']:
                best_model = "SVR Ensemble"
            else:
                best_model = "LSTM"
            
            print(f"\nBest performing model: {best_model}")
            
            self.evaluation_results = {
                'lstm_metrics': lstm_test_metrics,
                'svr_metrics': svr_test_metrics,
                'improvements': improvements,
                'best_model': best_model
            }
        
        return self.evaluation_results
    
    def predict(self, X_new):
        """
        Make predictions with the ensemble model.
        
        Args:
            X_new: New input data
            
        Returns:
            np.array: Ensemble predictions
        """
        if self.svr_model is None:
            print("SVR model not trained. Please train the ensemble first.")
            return None
        
        try:
            # Prepare features for SVR (would need LSTM predictions)
            # This is a simplified version - in practice, you'd need to run
            # the full pipeline for new data
            predictions = self.svr_model.predict(X_new)
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def plot_results(self, save_plots=True):
        """Create comprehensive visualization plots."""
        if self.lstm_results is None or self.svr_results is None:
            print("No results to plot. Please train the ensemble first.")
            return
        
        try:
            # Plot 1: Model comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('LSTM-SVR Ensemble Model Results', fontsize=16)
            
            # LSTM training history
            if 'history' in self.lstm_results:
                history = self.lstm_results['history']
                axes[0, 0].plot(history['loss'], label='Training Loss')
                if 'val_loss' in history:
                    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
                axes[0, 0].set_title('LSTM Training History')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # LSTM predictions vs actual
            lstm_test = self.lstm_results['test']
            if lstm_test['actual'] is not None and len(lstm_test['actual']) > 0:
                axes[0, 1].scatter(lstm_test['actual'], lstm_test['predicted'], alpha=0.6)
                min_val = min(lstm_test['actual'].min(), lstm_test['predicted'].min())
                max_val = max(lstm_test['actual'].max(), lstm_test['predicted'].max())
                axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                axes[0, 1].set_title('LSTM: Predicted vs Actual')
                axes[0, 1].set_xlabel('Actual Values')
                axes[0, 1].set_ylabel('Predicted Values')
                axes[0, 1].grid(True, alpha=0.3)
            
            # SVR predictions vs actual
            svr_test = self.svr_results['test']
            if svr_test['actual'] is not None and len(svr_test['actual']) > 0:
                axes[1, 0].scatter(svr_test['actual'], svr_test['predicted'], alpha=0.6, color='orange')
                min_val = min(svr_test['actual'].min(), svr_test['predicted'].min())
                max_val = max(svr_test['actual'].max(), svr_test['predicted'].max())
                axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                axes[1, 0].set_title('SVR Ensemble: Predicted vs Actual')
                axes[1, 0].set_xlabel('Actual Values')
                axes[1, 0].set_ylabel('Predicted Values')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Performance comparison
            if self.evaluation_results:
                metrics = ['rmse', 'mae', 'r2']
                lstm_vals = [self.evaluation_results['lstm_metrics'][m] for m in metrics]
                svr_vals = [self.evaluation_results['svr_metrics'][m] for m in metrics]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, lstm_vals, width, label='LSTM', alpha=0.7)
                axes[1, 1].bar(x + width/2, svr_vals, width, label='SVR Ensemble', alpha=0.7)
                axes[1, 1].set_title('Performance Comparison')
                axes[1, 1].set_xlabel('Metrics')
                axes[1, 1].set_ylabel('Values')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels([m.upper() for m in metrics])
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(self.output_folder, 'ensemble_results.png'), 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
            # Plot 2: Time series comparison
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            # Combine all data for time series plot
            all_actual = np.concatenate([
                self.lstm_results['train']['actual'],
                self.lstm_results['validation']['actual'],
                self.lstm_results['test']['actual']
            ])
            
            all_lstm_pred = np.concatenate([
                self.lstm_results['train']['predicted'],
                self.lstm_results['validation']['predicted'],
                self.lstm_results['test']['predicted']
            ])
            
            all_svr_pred = np.concatenate([
                self.svr_results['train']['predicted'],
                self.svr_results['validation']['predicted'],
                self.svr_results['test']['predicted']
            ])
            
            time_index = range(len(all_actual))
            
            ax.plot(time_index, all_actual, label='Actual', linewidth=2, alpha=0.8)
            ax.plot(time_index, all_lstm_pred, label='LSTM Prediction', linewidth=2, alpha=0.7)
            ax.plot(time_index, all_svr_pred, label='SVR Ensemble Prediction', linewidth=2, alpha=0.7)
            
            # Mark train/validation/test boundaries
            train_end = len(self.lstm_results['train']['actual'])
            val_end = train_end + len(self.lstm_results['validation']['actual'])
            
            ax.axvline(x=train_end, color='gray', linestyle='--', alpha=0.5, label='Train/Val Split')
            ax.axvline(x=val_end, color='red', linestyle='--', alpha=0.5, label='Val/Test Split')
            
            ax.set_title(f'Time Series Predictions: {self.target_column}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_plots:
                plt.savefig(os.path.join(self.output_folder, 'time_series_comparison.png'), 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def save_model(self, filename=None):
        """Save the complete ensemble model."""
        if self.lstm_model is None or self.svr_model is None:
            print("No complete ensemble to save. Please train both models first.")
            return None
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lstm_svr_ensemble_{timestamp}"
        
        try:
            # Save ensemble components
            ensemble_data = {
                'lstm_model': self.lstm_model,
                'svr_model': self.svr_model,
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'lstm_params': self.lstm_params,
                'svr_params': self.svr_params,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'lstm_results': self.lstm_results,
                'svr_results': self.svr_results,
                'evaluation_results': self.evaluation_results,
                'processed_data': self.processed_data
            }
            
            model_path = os.path.join(self.output_folder, f"{filename}.joblib")
            joblib.dump(ensemble_data, model_path)
            
            print(f"Ensemble model saved to: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"Error saving ensemble model: {e}")
            return None
    
    def load_model(self, model_path):
        """Load a previously saved ensemble model."""
        try:
            ensemble_data = joblib.load(model_path)
            
            self.lstm_model = ensemble_data['lstm_model']
            self.svr_model = ensemble_data['svr_model']
            self.feature_scaler = ensemble_data['feature_scaler']
            self.target_scaler = ensemble_data['target_scaler']
            self.lstm_params = ensemble_data['lstm_params']
            self.svr_params = ensemble_data['svr_params']
            self.feature_columns = ensemble_data['feature_columns']
            self.target_column = ensemble_data['target_column']
            self.lstm_results = ensemble_data.get('lstm_results')
            self.svr_results = ensemble_data.get('svr_results')
            self.evaluation_results = ensemble_data.get('evaluation_results')
            self.processed_data = ensemble_data.get('processed_data')
            
            print(f"Ensemble model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading ensemble model: {e}")
            return False
    
    def export_results(self, filename=None):
        """Export all results to Excel file."""
        if self.lstm_results is None or self.svr_results is None:
            print("No results to export. Please train the ensemble first.")
            return None
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_results_{timestamp}.xlsx"
        
        try:
            excel_path = os.path.join(self.output_folder, filename)
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # LSTM results
                self._export_model_results(writer, 'LSTM', self.lstm_results)
                
                # SVR results
                self._export_model_results(writer, 'SVR_Ensemble', self.svr_results)
                
                # Comparison summary
                if self.evaluation_results:
                    comparison_df = pd.DataFrame({
                        'Model': ['LSTM', 'SVR_Ensemble'],
                        'RMSE': [self.evaluation_results['lstm_metrics']['rmse'], 
                                self.evaluation_results['svr_metrics']['rmse']],
                        'MAE': [self.evaluation_results['lstm_metrics']['mae'], 
                               self.evaluation_results['svr_metrics']['mae']],
                        'R2': [self.evaluation_results['lstm_metrics']['r2'], 
                              self.evaluation_results['svr_metrics']['r2']],
                        'MAPE': [self.evaluation_results['lstm_metrics']['mape'], 
                                self.evaluation_results['svr_metrics']['mape']]
                    })
                    comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
                
                # Parameters
                params_df = pd.DataFrame([
                    ['LSTM_sequence_length', self.lstm_params['sequence_length']],
                    ['LSTM_units', str(self.lstm_params['lstm_units'])],
                    ['LSTM_dropout_rate', self.lstm_params['dropout_rate']],
                    ['LSTM_learning_rate', self.lstm_params['learning_rate']],
                    ['SVR_best_params', str(self.svr_results.get('best_params', 'N/A'))],
                    ['SVR_best_score', self.svr_results.get('best_score', 'N/A')]
                ], columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            print(f"Results exported to: {excel_path}")
            return excel_path
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return None
    
    def _export_model_results(self, writer, model_name, results):
        """Export individual model results to Excel."""
        for dataset in ['train', 'validation', 'test']:
            if dataset in results and results[dataset]['actual'] is not None:
                data = results[dataset]
                if len(data['actual']) > 0:
                    df = pd.DataFrame({
                        'Actual': data['actual'],
                        'Predicted': data['predicted'],
                        'Residual': data['actual'] - data['predicted'],
                        'Absolute_Error': np.abs(data['actual'] - data['predicted'])
                    })
                    
                    if len(data['actual']) > 0:
                        df['Percentage_Error'] = ((data['actual'] - data['predicted']) / data['actual']) * 100
                    
                    sheet_name = f"{model_name}_{dataset.capitalize()}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)


class EnsembleLSTMSVRGUI:
    """GUI application for LSTM-SVR ensemble training."""
    
    def __init__(self, master):
        self.master = master
        master.title("LSTM-SVR 集成模型训练工具")
        master.geometry("900x700")
        
        # Initialize ensemble model
        self.ensemble = LSTMSVREnsemble()
        
        # State variables
        self.file_var = tk.StringVar()
        self.sheet_var = tk.StringVar()
        self.target_var = tk.StringVar()
        
        # Create GUI elements
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        main_frame = ttk.Frame(self.master, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="数据文件", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Excel文件:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(file_frame, text="浏览...", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(file_frame, text="工作表:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.sheet_var, width=25).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(file_frame, text="目标列:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.target_var, width=25).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        file_frame.columnconfigure(1, weight=1)
        
        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="模型参数", padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        # LSTM parameters
        lstm_frame = ttk.LabelFrame(param_frame, text="LSTM参数")
        lstm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(lstm_frame, text="序列长度:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.seq_len_var = tk.IntVar(value=3)
        ttk.Entry(lstm_frame, textvariable=self.seq_len_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(lstm_frame, text="LSTM单元:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.lstm_units_var = tk.StringVar(value="50,50")
        ttk.Entry(lstm_frame, textvariable=self.lstm_units_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(lstm_frame, text="学习率:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(lstm_frame, textvariable=self.lr_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(lstm_frame, text="训练轮数:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(lstm_frame, textvariable=self.epochs_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # SVR parameters
        svr_frame = ttk.LabelFrame(param_frame, text="SVR参数")
        svr_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(svr_frame, text="交叉验证折数:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.cv_folds_var = tk.IntVar(value=5)
        ttk.Entry(svr_frame, textvariable=self.cv_folds_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(svr_frame, text="特征缩放:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.feature_scaling_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(svr_frame, variable=self.feature_scaling_var).grid(row=1, column=1, padx=5, pady=2)
        
        # Control buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.train_btn = ttk.Button(btn_frame, text="训练集成模型", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.plot_btn = ttk.Button(btn_frame, text="显示结果图", command=self.show_plots, state=tk.DISABLED)
        self.plot_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(btn_frame, text="保存模型", command=self.save_model, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = ttk.Button(btn_frame, text="导出结果", command=self.export_results, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="训练日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def browse_file(self):
        """Browse for Excel file."""
        file_path = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        
        if file_path:
            self.file_var.set(file_path)
            self.log(f"已选择文件: {os.path.basename(file_path)}")
    
    def log(self, message):
        """Add message to log area."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.master.update_idletasks()
    
    def update_parameters(self):
        """Update ensemble parameters from GUI."""
        # Update LSTM parameters
        self.ensemble.lstm_params['sequence_length'] = self.seq_len_var.get()
        
        units_str = self.lstm_units_var.get()
        try:
            units = [int(x.strip()) for x in units_str.split(',')]
            self.ensemble.lstm_params['lstm_units'] = units
        except:
            self.log("警告: LSTM单元数格式错误，使用默认值 [50, 50]")
        
        self.ensemble.lstm_params['learning_rate'] = self.lr_var.get()
        self.ensemble.lstm_params['epochs'] = self.epochs_var.get()
        
        # Update SVR parameters
        self.ensemble.svr_params['cv_folds'] = self.cv_folds_var.get()
        self.ensemble.use_feature_scaling = self.feature_scaling_var.get()
    
    def start_training(self):
        """Start ensemble training process."""
        file_path = self.file_var.get()
        if not file_path:
            messagebox.showwarning("警告", "请先选择Excel文件")
            return
        
        # Update parameters
        self.update_parameters()
        
        # Disable controls
        self.train_btn.config(state=tk.DISABLED)
        self.progress.start()
        
        # Clear log
        self.log_text.delete('1.0', tk.END)
        
        # Start training thread
        def training_thread():
            try:
                def progress_callback(message):
                    self.master.after(0, lambda: self.log(message))
                
                # Load data
                sheet_name = self.sheet_var.get() or None
                target_column = self.target_var.get() or None
                
                progress_callback("加载数据...")
                if not self.ensemble.load_data(file_path, sheet_name, target_column):
                    raise ValueError("数据加载失败")
                
                # Train ensemble
                progress_callback("开始训练集成模型...")
                results = self.ensemble.train_ensemble(progress_callback)
                
                if results:
                    self.master.after(0, lambda: self.training_completed(True))
                else:
                    self.master.after(0, lambda: self.training_completed(False))
                    
            except Exception as e:
                error_msg = f"训练失败: {str(e)}"
                self.master.after(0, lambda: self.log(error_msg))
                self.master.after(0, lambda: self.training_completed(False))
        
        threading.Thread(target=training_thread, daemon=True).start()
    
    def training_completed(self, success):
        """Handle training completion."""
        self.progress.stop()
        self.train_btn.config(state=tk.NORMAL)
        
        if success:
            self.plot_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            messagebox.showinfo("成功", "集成模型训练完成!")
            self.log("=== 训练完成 ===")
        else:
            messagebox.showerror("失败", "集成模型训练失败，请查看日志")
    
    def show_plots(self):
        """Show result plots."""
        try:
            self.ensemble.plot_results()
        except Exception as e:
            messagebox.showerror("错误", f"显示图表失败: {e}")
    
    def save_model(self):
        """Save trained model."""
        try:
            save_path = filedialog.asksaveasfilename(
                title="保存模型",
                defaultextension=".joblib",
                filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
            )
            
            if save_path:
                # Remove extension as it will be added by save_model
                base_name = os.path.splitext(os.path.basename(save_path))[0]
                result_path = self.ensemble.save_model(base_name)
                
                if result_path:
                    messagebox.showinfo("成功", f"模型已保存到: {result_path}")
                else:
                    messagebox.showerror("失败", "模型保存失败")
        except Exception as e:
            messagebox.showerror("错误", f"保存模型失败: {e}")
    
    def export_results(self):
        """Export results to Excel."""
        try:
            save_path = filedialog.asksaveasfilename(
                title="导出结果",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if save_path:
                filename = os.path.basename(save_path)
                result_path = self.ensemble.export_results(filename)
                
                if result_path:
                    messagebox.showinfo("成功", f"结果已导出到: {result_path}")
                else:
                    messagebox.showerror("失败", "结果导出失败")
        except Exception as e:
            messagebox.showerror("错误", f"导出结果失败: {e}")


def create_gui():
    """Create and run the GUI application."""
    root = tk.Tk()
    app = EnsembleLSTMSVRGUI(root)
    root.mainloop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        file_path = sys.argv[1]
        sheet_name = sys.argv[2] if len(sys.argv) > 2 else None
        target_column = sys.argv[3] if len(sys.argv) > 3 else None
        
        ensemble = LSTMSVREnsemble()
        
        if ensemble.load_data(file_path, sheet_name, target_column):
            print("Data loaded successfully")
            
            results = ensemble.train_ensemble()
            if results:
                print("Ensemble training completed successfully")
                ensemble.plot_results()
                ensemble.save_model()
                ensemble.export_results()
            else:
                print("Ensemble training failed")
        else:
            print("Data loading failed")
    else:
        # GUI usage
        if not TENSORFLOW_AVAILABLE:
            print("Warning: TensorFlow not available. LSTM functionality will be limited.")
        create_gui()