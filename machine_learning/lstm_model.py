#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM Model for Agricultural Time Series Prediction

This module provides a comprehensive LSTM implementation for time series forecasting
with agricultural research data. Features include:
- Advanced time series data preprocessing
- Multi-layer LSTM architecture with dropout regularization
- Real-time training progress monitoring
- Comprehensive performance evaluation
- Visualization and reporting capabilities
- GUI interface for easy usage

Combines functionality from:
- LSTM.py

Author: Thesis Research Project
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import joblib

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

# Set matplotlib for Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class LSTMPredictor:
    """
    Advanced LSTM model for agricultural time series prediction.
    
    Supports multiple prediction horizons and provides comprehensive evaluation.
    """
    
    def __init__(self, output_folder=None):
        """Initialize LSTM predictor with default parameters."""
        # Output folder setup
        if output_folder is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = f"LSTM_Results_{timestamp}"
        else:
            self.output_folder = output_folder
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Model parameters with sensible defaults
        self.params = {
            'sequence_length': 3,  # Number of time steps to look back
            'lstm_units': [50, 50],  # LSTM layer units
            'dropout_rate': 0.2,  # Dropout rate for regularization
            'learning_rate': 0.001,  # Learning rate for optimizer
            'epochs': 100,  # Maximum training epochs
            'batch_size': 32,  # Batch size for training
            'validation_split': 0.2,  # Fraction of data for validation
            'patience': 15,  # Early stopping patience
            'predict_steps': 5  # Number of steps to predict ahead
        }
        
        # Model components
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.predictions = None
        self.evaluation_results = None
        
        # Data storage
        self.original_data = None
        self.processed_data = None
        self.feature_columns = None
        self.target_column = None
    
    def load_data(self, file_path, sheet_name=None, target_column=None):
        """
        Load and preprocess data from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet to load (if None, loads first sheet)
            target_column: Name of target column (if None, uses last column)
            
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
            
            # Identify feature and target columns
            if target_column and target_column in df.columns:
                self.target_column = target_column
                self.feature_columns = [col for col in df.columns if col != target_column]
            else:
                # Assume last column is target
                self.target_column = df.columns[-1]
                self.feature_columns = df.columns[:-1].tolist()
            
            # Store original data
            self.original_data = df.copy()
            
            # Convert to numeric and handle missing values
            numeric_columns = self.feature_columns + [self.target_column]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing values
            df = df.dropna(subset=numeric_columns)
            
            if df.empty:
                raise ValueError("No valid data remaining after preprocessing")
            
            print(f"After preprocessing: {df.shape}")
            print(f"Target column: {self.target_column}")
            print(f"Feature columns: {self.feature_columns}")
            
            self.processed_data = df
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_sequences(self, data, sequence_length, predict_steps=1):
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input data array
            sequence_length: Number of time steps in each sequence
            predict_steps: Number of steps to predict ahead
            
        Returns:
            tuple: (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data) - predict_steps + 1):
            # Input sequence
            X.append(data[i-sequence_length:i])
            # Target (can be single step or multiple steps)
            if predict_steps == 1:
                y.append(data[i])
            else:
                y.append(data[i:i+predict_steps])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape, output_shape=1):
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, features)
            output_shape: Shape of output (1 for single-step, n for multi-step)
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for LSTM model building")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.params['lstm_units'][0],
            return_sequences=len(self.params['lstm_units']) > 1,
            input_shape=input_shape
        ))
        model.add(Dropout(self.params['dropout_rate']))
        
        # Additional LSTM layers
        for i in range(1, len(self.params['lstm_units'])):
            return_seq = i < len(self.params['lstm_units']) - 1
            model.add(LSTM(
                self.params['lstm_units'][i],
                return_sequences=return_seq
            ))
            model.add(Dropout(self.params['dropout_rate']))
        
        # Output layer
        model.add(Dense(output_shape))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, progress_callback=None):
        """
        Train the LSTM model on loaded data.
        
        Args:
            progress_callback: Function to report training progress
            
        Returns:
            bool: Success status
        """
        if self.processed_data is None:
            print("No data loaded. Please load data first.")
            return False
        
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow is required for model training")
            return False
        
        try:
            # Prepare training data
            print("Preparing training data...")
            
            # Use only target column for univariate prediction
            # (Can be extended for multivariate prediction)
            target_data = self.processed_data[self.target_column].values.reshape(-1, 1)
            
            # Scale data
            scaled_data = self.scaler.fit_transform(target_data)
            
            # Create sequences
            X, y = self.create_sequences(
                scaled_data.flatten(),
                self.params['sequence_length'],
                self.params['predict_steps']
            )
            
            if len(X) == 0:
                raise ValueError("Insufficient data to create sequences")
            
            # Reshape for LSTM input
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            print(f"Training data shape: X={X.shape}, y={y.shape}")
            
            # Build model
            print("Building model...")
            input_shape = (X.shape[1], X.shape[2])
            output_shape = y.shape[1] if len(y.shape) > 1 else 1
            
            self.model = self.build_model(input_shape, output_shape)
            
            print("Model architecture:")
            self.model.summary()
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.params['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.params['patience']//2,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            print("Starting training...")
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, callback_func=None):
                    super().__init__()
                    self.callback_func = callback_func
                
                def on_epoch_end(self, epoch, logs=None):
                    if self.callback_func:
                        self.callback_func(epoch + 1, self.params['epochs'], logs)
            
            if progress_callback:
                callbacks.append(ProgressCallback(progress_callback))
            
            self.history = self.model.fit(
                X, y,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_split=self.params['validation_split'],
                callbacks=callbacks,
                verbose=1
            )
            
            print("Training completed successfully!")
            
            # Generate predictions
            self.predictions = self.model.predict(X)
            
            # Inverse transform predictions
            if self.params['predict_steps'] == 1:
                pred_original = self.scaler.inverse_transform(self.predictions.reshape(-1, 1))
                y_original = self.scaler.inverse_transform(y.reshape(-1, 1))
            else:
                pred_original = self.scaler.inverse_transform(self.predictions)
                y_original = self.scaler.inverse_transform(y)
            
            # Calculate evaluation metrics
            self.evaluation_results = self.evaluate_predictions(y_original, pred_original)
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def evaluate_predictions(self, y_true, y_pred):
        """
        Evaluate prediction performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            dict: Evaluation metrics
        """
        # Handle multi-step predictions by using last step
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_eval = y_true[:, -1]
            y_pred_eval = y_pred[:, -1]
        else:
            y_true_eval = y_true.flatten()
            y_pred_eval = y_pred.flatten()
        
        mse = mean_squared_error(y_true_eval, y_pred_eval)
        mae = mean_absolute_error(y_true_eval, y_pred_eval)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_eval, y_pred_eval)
        
        # Calculate MAPE (avoiding division by zero)
        mask = y_true_eval != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true_eval[mask] - y_pred_eval[mask]) / y_true_eval[mask])) * 100
        else:
            mape = np.inf
        
        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        print(f"\nModel Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return results
    
    def predict_future(self, steps=None):
        """
        Predict future values using the trained model.
        
        Args:
            steps: Number of future steps to predict
            
        Returns:
            np.array: Future predictions
        """
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None
        
        if steps is None:
            steps = self.params['predict_steps']
        
        try:
            # Get last sequence from training data
            target_data = self.processed_data[self.target_column].values.reshape(-1, 1)
            scaled_data = self.scaler.transform(target_data)
            
            # Get last sequence
            last_sequence = scaled_data[-self.params['sequence_length']:].flatten()
            
            future_predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # Reshape for prediction
                X_pred = current_sequence.reshape(1, self.params['sequence_length'], 1)
                
                # Predict next value
                next_pred = self.model.predict(X_pred, verbose=0)[0]
                
                if isinstance(next_pred, np.ndarray):
                    next_value = next_pred[0] if len(next_pred) > 0 else next_pred
                else:
                    next_value = next_pred
                
                future_predictions.append(next_value)
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], next_value)
            
            # Inverse transform predictions
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_original = self.scaler.inverse_transform(future_predictions)
            
            return future_original.flatten()
            
        except Exception as e:
            print(f"Error predicting future values: {e}")
            return None
    
    def plot_results(self, save_plots=True):
        """
        Create visualization plots for model results.
        
        Args:
            save_plots: Whether to save plots to files
        """
        if self.model is None or self.predictions is None:
            print("No results to plot. Please train the model first.")
            return
        
        try:
            # Prepare data for plotting
            target_data = self.processed_data[self.target_column].values
            
            # Get original scale predictions
            if self.params['predict_steps'] == 1:
                pred_original = self.scaler.inverse_transform(self.predictions.reshape(-1, 1)).flatten()
            else:
                pred_original = self.scaler.inverse_transform(self.predictions)[:, -1]
            
            # Align with original data (account for sequence length)
            actual_values = target_data[self.params['sequence_length']:self.params['sequence_length']+len(pred_original)]
            
            # Plot 1: Training history
            if self.history:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(self.history.history['loss'], label='Training Loss')
                plt.plot(self.history.history['val_loss'], label='Validation Loss')
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 3, 2)
                plt.plot(self.history.history['mae'], label='Training MAE')
                plt.plot(self.history.history['val_mae'], label='Validation MAE')
                plt.title('Model MAE')
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 3, 3)
                plt.plot(actual_values, label='Actual', alpha=0.7)
                plt.plot(pred_original, label='Predicted', alpha=0.7)
                plt.title('Predictions vs Actual')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(os.path.join(self.output_folder, 'training_results.png'), dpi=300, bbox_inches='tight')
                plt.show()
            
            # Plot 2: Detailed prediction comparison
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(actual_values, label='Actual', linewidth=2)
            plt.plot(pred_original, label='Predicted', linewidth=2, alpha=0.8)
            plt.title(f'LSTM Predictions vs Actual Values - {self.target_column}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            residuals = actual_values - pred_original
            plt.plot(residuals, color='red', alpha=0.7)
            plt.title('Prediction Residuals')
            plt.xlabel('Time Step')
            plt.ylabel('Residual')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(self.output_folder, 'prediction_comparison.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            # Plot 3: Future predictions if available
            future_pred = self.predict_future(10)  # Predict 10 steps ahead
            if future_pred is not None:
                plt.figure(figsize=(12, 6))
                
                # Plot historical data
                historical_x = range(len(target_data))
                plt.plot(historical_x, target_data, label='Historical Data', color='blue', linewidth=2)
                
                # Plot future predictions
                future_x = range(len(target_data), len(target_data) + len(future_pred))
                plt.plot(future_x, future_pred, label='Future Predictions', color='red', linewidth=2, linestyle='--')
                
                plt.axvline(x=len(target_data)-1, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')
                plt.title(f'Historical Data and Future Predictions - {self.target_column}')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if save_plots:
                    plt.savefig(os.path.join(self.output_folder, 'future_predictions.png'), dpi=300, bbox_inches='tight')
                plt.show()
        
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def save_model(self, filename=None):
        """
        Save the trained model and associated components.
        
        Args:
            filename: Base filename for saved files
            
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            print("No model to save. Please train the model first.")
            return None
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lstm_model_{timestamp}"
        
        try:
            # Save Keras model
            model_path = os.path.join(self.output_folder, f"{filename}.h5")
            self.model.save(model_path)
            
            # Save additional components
            components = {
                'scaler': self.scaler,
                'params': self.params,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'evaluation_results': self.evaluation_results
            }
            
            components_path = os.path.join(self.output_folder, f"{filename}_components.joblib")
            joblib.dump(components, components_path)
            
            print(f"Model saved to: {model_path}")
            print(f"Components saved to: {components_path}")
            
            return model_path
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
    
    def load_model(self, model_path, components_path):
        """
        Load a previously saved model.
        
        Args:
            model_path: Path to saved Keras model
            components_path: Path to saved components
            
        Returns:
            bool: Success status
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                print("TensorFlow is required to load Keras models")
                return False
            
            # Load Keras model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load components
            components = joblib.load(components_path)
            self.scaler = components['scaler']
            self.params = components['params']
            self.feature_columns = components['feature_columns']
            self.target_column = components['target_column']
            self.evaluation_results = components.get('evaluation_results')
            
            print(f"Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def export_results(self, filename=None):
        """
        Export prediction results to Excel file.
        
        Args:
            filename: Output filename
            
        Returns:
            str: Path to exported file
        """
        if self.predictions is None:
            print("No results to export. Please train the model first.")
            return None
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lstm_results_{timestamp}.xlsx"
        
        try:
            # Prepare results data
            target_data = self.processed_data[self.target_column].values
            
            if self.params['predict_steps'] == 1:
                pred_original = self.scaler.inverse_transform(self.predictions.reshape(-1, 1)).flatten()
            else:
                pred_original = self.scaler.inverse_transform(self.predictions)[:, -1]
            
            # Align data
            actual_values = target_data[self.params['sequence_length']:self.params['sequence_length']+len(pred_original)]
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Time_Step': range(len(actual_values)),
                'Actual': actual_values,
                'Predicted': pred_original,
                'Residual': actual_values - pred_original,
                'Absolute_Error': np.abs(actual_values - pred_original),
                'Percentage_Error': ((actual_values - pred_original) / actual_values) * 100
            })
            
            # Export to Excel with multiple sheets
            excel_path = os.path.join(self.output_folder, filename)
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Main results
                results_df.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Model parameters
                params_df = pd.DataFrame(list(self.params.items()), columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
                
                # Evaluation metrics
                if self.evaluation_results:
                    eval_df = pd.DataFrame(list(self.evaluation_results.items()), columns=['Metric', 'Value'])
                    eval_df.to_excel(writer, sheet_name='Evaluation', index=False)
                
                # Future predictions if available
                future_pred = self.predict_future(10)
                if future_pred is not None:
                    future_df = pd.DataFrame({
                        'Future_Step': range(1, len(future_pred) + 1),
                        'Predicted_Value': future_pred
                    })
                    future_df.to_excel(writer, sheet_name='Future_Predictions', index=False)
            
            print(f"Results exported to: {excel_path}")
            return excel_path
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return None


class LSTMPredictorGUI:
    """
    GUI application for LSTM time series prediction.
    """
    
    def __init__(self, master):
        self.master = master
        master.title("LSTM 时间序列预测工具")
        master.geometry("900x700")
        
        # Initialize predictor
        self.predictor = LSTMPredictor()
        
        # State variables
        self.file_path = tk.StringVar()
        self.sheet_names = []
        self.current_data = None
        
        # Create GUI elements
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Data tab
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="数据加载")
        self._create_data_tab()
        
        # Model tab
        self.model_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.model_frame, text="模型训练")
        self._create_model_tab()
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="结果分析")
        self._create_results_tab()
    
    def _create_data_tab(self):
        """Create data loading tab."""
        # File selection
        file_frame = ttk.LabelFrame(self.data_frame, text="数据文件", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Excel文件:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(file_frame, text="浏览...", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        file_frame.columnconfigure(1, weight=1)
        
        # Sheet selection
        sheet_frame = ttk.LabelFrame(self.data_frame, text="工作表选择", padding=10)
        sheet_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sheet_frame, text="选择工作表:").pack(anchor=tk.W)
        self.sheet_listbox = tk.Listbox(sheet_frame, height=6)
        self.sheet_listbox.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(sheet_frame, text="加载数据", command=self.load_data).pack(pady=5)
        
        # Data preview
        preview_frame = ttk.LabelFrame(self.data_frame, text="数据预览", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.data_text = scrolledtext.ScrolledText(preview_frame, height=15)
        self.data_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_model_tab(self):
        """Create model training tab."""
        # Parameters frame
        params_frame = ttk.LabelFrame(self.model_frame, text="模型参数", padding=10)
        params_frame.pack(fill=tk.X, pady=5)
        
        # Parameter controls
        params = [
            ('sequence_length', '序列长度:', 3),
            ('lstm_units', 'LSTM单元数:', '50,50'),
            ('dropout_rate', 'Dropout率:', 0.2),
            ('learning_rate', '学习率:', 0.001),
            ('epochs', '训练轮数:', 100),
            ('batch_size', '批大小:', 32),
            ('predict_steps', '预测步数:', 5)
        ]
        
        self.param_vars = {}
        for i, (key, label, default) in enumerate(params):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(params_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=str(default))
            self.param_vars[key] = var
            ttk.Entry(params_frame, textvariable=var, width=15).grid(row=row, column=col+1, padx=5, pady=2, sticky=tk.W)
        
        # Training controls
        train_frame = ttk.Frame(self.model_frame)
        train_frame.pack(fill=tk.X, pady=10)
        
        self.train_button = ttk.Button(train_frame, text="开始训练", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(train_frame, text="停止训练", state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="就绪")
        ttk.Label(self.model_frame, textvariable=self.progress_var).pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(self.model_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(self.model_frame, text="训练日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_results_tab(self):
        """Create results analysis tab."""
        # Results controls
        controls_frame = ttk.Frame(self.results_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="显示结果", command=self.show_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="导出结果", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="保存模型", command=self.save_model).pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_display_frame = ttk.LabelFrame(self.results_frame, text="结果显示", padding=10)
        results_display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_display_frame)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def browse_file(self):
        """Browse for Excel file."""
        filename = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)
            self.load_sheet_names()
    
    def load_sheet_names(self):
        """Load sheet names from Excel file."""
        try:
            xl = pd.ExcelFile(self.file_path.get())
            self.sheet_names = xl.sheet_names
            
            self.sheet_listbox.delete(0, tk.END)
            for sheet in self.sheet_names:
                self.sheet_listbox.insert(tk.END, sheet)
            
            if self.sheet_names:
                self.sheet_listbox.selection_set(0)
                
        except Exception as e:
            messagebox.showerror("错误", f"无法加载工作表: {e}")
    
    def load_data(self):
        """Load data from selected sheet."""
        try:
            selection = self.sheet_listbox.curselection()
            if not selection:
                messagebox.showwarning("警告", "请选择一个工作表")
                return
            
            sheet_name = self.sheet_names[selection[0]]
            
            if self.predictor.load_data(self.file_path.get(), sheet_name):
                # Display data preview
                preview_text = f"数据加载成功!\n\n"
                preview_text += f"工作表: {sheet_name}\n"
                preview_text += f"数据形状: {self.predictor.processed_data.shape}\n"
                preview_text += f"目标列: {self.predictor.target_column}\n"
                preview_text += f"特征列: {', '.join(self.predictor.feature_columns)}\n\n"
                preview_text += "数据预览:\n"
                preview_text += str(self.predictor.processed_data.head(10))
                
                self.data_text.delete('1.0', tk.END)
                self.data_text.insert('1.0', preview_text)
                
                messagebox.showinfo("成功", "数据加载成功!")
            else:
                messagebox.showerror("错误", "数据加载失败")
                
        except Exception as e:
            messagebox.showerror("错误", f"加载数据时出错: {e}")
    
    def start_training(self):
        """Start model training in a separate thread."""
        if self.predictor.processed_data is None:
            messagebox.showwarning("警告", "请先加载数据")
            return
        
        # Update parameters
        try:
            self.predictor.params['sequence_length'] = int(self.param_vars['sequence_length'].get())
            
            # Parse LSTM units
            units_str = self.param_vars['lstm_units'].get()
            if ',' in units_str:
                self.predictor.params['lstm_units'] = [int(x.strip()) for x in units_str.split(',')]
            else:
                self.predictor.params['lstm_units'] = [int(units_str)]
            
            self.predictor.params['dropout_rate'] = float(self.param_vars['dropout_rate'].get())
            self.predictor.params['learning_rate'] = float(self.param_vars['learning_rate'].get())
            self.predictor.params['epochs'] = int(self.param_vars['epochs'].get())
            self.predictor.params['batch_size'] = int(self.param_vars['batch_size'].get())
            self.predictor.params['predict_steps'] = int(self.param_vars['predict_steps'].get())
            
        except ValueError as e:
            messagebox.showerror("错误", f"参数设置错误: {e}")
            return
        
        # Disable training button
        self.train_button.config(state=tk.DISABLED)
        self.progress_bar['maximum'] = self.predictor.params['epochs']
        
        # Clear log
        self.log_text.delete('1.0', tk.END)
        
        # Define progress callback
        def progress_callback(epoch, total_epochs, logs):
            self.progress_var.set(f"训练进度: {epoch}/{total_epochs}")
            self.progress_bar['value'] = epoch
            
            if logs:
                log_msg = f"Epoch {epoch}/{total_epochs} - "
                log_msg += f"loss: {logs.get('loss', 0):.4f} - "
                log_msg += f"val_loss: {logs.get('val_loss', 0):.4f}\n"
                
                self.log_text.insert(tk.END, log_msg)
                self.log_text.see(tk.END)
            
            self.master.update_idletasks()
        
        # Start training thread
        def train_thread():
            try:
                success = self.predictor.train_model(progress_callback)
                
                # Re-enable training button
                self.master.after(0, lambda: self.train_button.config(state=tk.NORMAL))
                
                if success:
                    self.master.after(0, lambda: self.progress_var.set("训练完成!"))
                    self.master.after(0, lambda: messagebox.showinfo("成功", "模型训练完成!"))
                else:
                    self.master.after(0, lambda: self.progress_var.set("训练失败"))
                    self.master.after(0, lambda: messagebox.showerror("错误", "模型训练失败"))
                    
            except Exception as e:
                self.master.after(0, lambda: self.train_button.config(state=tk.NORMAL))
                self.master.after(0, lambda: messagebox.showerror("错误", f"训练过程中出错: {e}"))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def show_results(self):
        """Display training results."""
        if self.predictor.model is None:
            messagebox.showwarning("警告", "请先训练模型")
            return
        
        try:
            # Display evaluation results
            results_text = "模型评估结果:\n\n"
            
            if self.predictor.evaluation_results:
                for metric, value in self.predictor.evaluation_results.items():
                    results_text += f"{metric.upper()}: {value:.4f}\n"
            
            results_text += "\n模型参数:\n\n"
            for param, value in self.predictor.params.items():
                results_text += f"{param}: {value}\n"
            
            self.results_text.delete('1.0', tk.END)
            self.results_text.insert('1.0', results_text)
            
            # Show plots
            self.predictor.plot_results(save_plots=True)
            
        except Exception as e:
            messagebox.showerror("错误", f"显示结果时出错: {e}")
    
    def export_results(self):
        """Export results to Excel file."""
        if self.predictor.predictions is None:
            messagebox.showwarning("警告", "没有结果可导出")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="保存结果",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if filename:
                result_path = self.predictor.export_results(os.path.basename(filename))
                if result_path:
                    messagebox.showinfo("成功", f"结果已导出到: {result_path}")
                
        except Exception as e:
            messagebox.showerror("错误", f"导出结果时出错: {e}")
    
    def save_model(self):
        """Save trained model."""
        if self.predictor.model is None:
            messagebox.showwarning("警告", "没有模型可保存")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="保存模型",
                defaultextension=".h5",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
            )
            
            if filename:
                base_name = os.path.splitext(os.path.basename(filename))[0]
                model_path = self.predictor.save_model(base_name)
                if model_path:
                    messagebox.showinfo("成功", f"模型已保存到: {model_path}")
                
        except Exception as e:
            messagebox.showerror("错误", f"保存模型时出错: {e}")


def create_gui():
    """Create and run the GUI application."""
    root = tk.Tk()
    app = LSTMPredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        file_path = sys.argv[1]
        sheet_name = sys.argv[2] if len(sys.argv) > 2 else None
        
        predictor = LSTMPredictor()
        
        if predictor.load_data(file_path, sheet_name):
            print("Data loaded successfully")
            
            if predictor.train_model():
                print("Model trained successfully")
                predictor.plot_results()
                predictor.export_results()
                predictor.save_model()
            else:
                print("Model training failed")
        else:
            print("Data loading failed")
    else:
        # GUI usage
        if not TENSORFLOW_AVAILABLE:
            print("Warning: TensorFlow not available. Some features may not work.")
        create_gui()
