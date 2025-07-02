#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genetic Algorithm Optimized SVR Model

This module provides SVR (Support Vector Regression) with genetic algorithm
optimization for hyperparameter tuning. Features include:
- Genetic algorithm-based hyperparameter optimization
- Time series cross-validation
- Comprehensive performance evaluation
- Multi-sheet Excel file processing
- Model persistence and visualization
- GUI interface for easy usage

Combines functionality from:
- svr.py

Author: Thesis Research Project
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
import warnings
import joblib
from datetime import datetime
import threading

# DEAP genetic algorithm library with fallback
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("Warning: DEAP library not available. Genetic algorithm optimization will not work.")

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib for Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class GeneticAlgorithmSVR:
    """
    SVR model with genetic algorithm optimization for hyperparameters.
    
    Uses genetic algorithms to find optimal SVR parameters including C, epsilon,
    gamma, and kernel type.
    """
    
    def __init__(self, output_folder=None):
        """Initialize GA-SVR with default parameters."""
        # Output folder setup
        if output_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = f"GA_SVR_Results_{timestamp}"
        else:
            self.output_folder = output_folder
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        # GA parameters
        self.ga_params = {
            'population_size': 50,
            'generations': 50,
            'cv_splits': 5,
            'crossover_prob': 0.7,
            'mutation_prob': 0.3,
            'tournament_size': 3
        }
        
        # SVR parameter bounds
        self.param_bounds = {
            'C': (0.1, 100.0),
            'epsilon': (0.001, 1.0),
            'gamma': (0.001, 1.0),
            'kernel_options': ['rbf', 'linear', 'poly', 'sigmoid']
        }
        
        # Model components
        self.best_model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.optimization_history = []
        self.evaluation_results = None
        
        # Data storage
        self.processed_data = None
        self.feature_columns = None
        self.target_column = None
    
    def load_data(self, file_path, sheet_name=None):
        """
        Load data from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet to load
            
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
            
            # Assume first column is time/index, last column is target
            if df.shape[1] < 2:
                raise ValueError("Data must have at least 2 columns (features and target)")
            
            # Identify columns
            self.feature_columns = df.columns[1:-1].tolist()  # Exclude first and last
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
            
            self.processed_data = df
            print(f"After preprocessing: {df.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_individual(self):
        """
        Create a random individual for the genetic algorithm.
        
        Returns:
            list: Individual with [C, epsilon, gamma, kernel_index]
        """
        C = random.uniform(*self.param_bounds['C'])
        epsilon = random.uniform(*self.param_bounds['epsilon'])
        gamma = random.uniform(*self.param_bounds['gamma'])
        kernel_idx = random.randint(0, len(self.param_bounds['kernel_options']) - 1)
        
        return [C, epsilon, gamma, kernel_idx]
    
    def evaluate_individual(self, individual, X, y):
        """
        Evaluate an individual's fitness using time series cross-validation.
        
        Args:
            individual: GA individual with SVR parameters
            X: Feature matrix
            y: Target values
            
        Returns:
            tuple: Fitness score (negative MSE)
        """
        # Decode individual
        C = max(0.1, individual[0])
        epsilon = max(0.001, individual[1])
        gamma = max(0.001, individual[2])
        kernel_idx = int(individual[3]) % len(self.param_bounds['kernel_options'])
        kernel = self.param_bounds['kernel_options'][kernel_idx]
        
        # Create SVR model
        model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.ga_params['cv_splits'])
        scores = []
        
        try:
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate MSE (negative for maximization)
                mse = mean_squared_error(y_test, y_pred)
                scores.append(-mse)
            
            # Return average score
            return (np.mean(scores),)
            
        except Exception as e:
            # Return very bad fitness for invalid parameters
            return (-1e6,)
    
    def custom_mutate(self, individual, mu=0, sigma=None, indpb=0.2):
        """
        Custom mutation operator that respects parameter bounds.
        
        Args:
            individual: Individual to mutate
            mu: Mean for Gaussian mutation
            sigma: Standard deviation for each parameter
            indpb: Probability of mutating each gene
            
        Returns:
            tuple: Mutated individual
        """
        if sigma is None:
            sigma = [10.0, 0.1, 0.1, 0.5]  # Different sigma for each parameter
        
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] += random.gauss(mu, sigma[i])
                
                # Enforce bounds
                if i == 0:  # C parameter
                    individual[i] = max(self.param_bounds['C'][0], 
                                      min(self.param_bounds['C'][1], individual[i]))
                elif i == 1:  # epsilon parameter
                    individual[i] = max(self.param_bounds['epsilon'][0], 
                                      min(self.param_bounds['epsilon'][1], individual[i]))
                elif i == 2:  # gamma parameter
                    individual[i] = max(self.param_bounds['gamma'][0], 
                                      min(self.param_bounds['gamma'][1], individual[i]))
                elif i == 3:  # kernel index
                    individual[i] = max(0, min(len(self.param_bounds['kernel_options']) - 1, 
                                              int(individual[i])))
        
        return individual,
    
    def optimize_parameters(self, progress_callback=None):
        """
        Use genetic algorithm to optimize SVR parameters.
        
        Args:
            progress_callback: Function to report optimization progress
            
        Returns:
            dict: Best parameters found
        """
        if self.processed_data is None:
            print("No data loaded. Please load data first.")
            return None
        
        if not DEAP_AVAILABLE:
            print("DEAP library is required for genetic algorithm optimization")
            return None
        
        try:
            print("Starting genetic algorithm optimization...")
            
            # Prepare data
            X = self.processed_data[self.feature_columns].values
            y = self.processed_data[self.target_column].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            print(f"Data shape: X={X_scaled.shape}, y={y.shape}")
            
            # Setup genetic algorithm
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            
            # Define individual creation
            toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Genetic operators
            toolbox.register("evaluate", self.evaluate_individual, X=X_scaled, y=y)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", self.custom_mutate)
            toolbox.register("select", tools.selTournament, tournsize=self.ga_params['tournament_size'])
            
            # Create initial population
            population = toolbox.population(n=self.ga_params['population_size'])
            
            # Statistics
            stats = tools.Statistics(key=lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Evolution
            print(f"Starting evolution: Population={self.ga_params['population_size']}, "
                  f"Generations={self.ga_params['generations']}")
            
            self.optimization_history = []
            
            # Evaluate initial population
            fitnesses = map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Evolution loop
            for gen in range(self.ga_params['generations']):
                # Selection
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))
                
                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.ga_params['crossover_prob']:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Mutation
                for mutant in offspring:
                    if random.random() < self.ga_params['mutation_prob']:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Evaluate offspring
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Replace population
                population[:] = offspring
                
                # Collect statistics
                record = stats.compile(population)
                self.optimization_history.append(record)
                
                # Report progress
                if progress_callback and (gen % 5 == 0 or gen == self.ga_params['generations'] - 1):
                    progress_callback(gen + 1, self.ga_params['generations'], record)
                elif gen % 5 == 0 or gen == self.ga_params['generations'] - 1:
                    print(f"Generation {gen + 1}/{self.ga_params['generations']}: "
                          f"Best fitness = {record['max']:.6f}")
            
            # Get best individual
            best_ind = tools.selBest(population, k=1)[0]
            
            # Decode best parameters
            best_C = max(0.1, best_ind[0])
            best_epsilon = max(0.001, best_ind[1])
            best_gamma = max(0.001, best_ind[2])
            best_kernel_idx = int(best_ind[3]) % len(self.param_bounds['kernel_options'])
            best_kernel = self.param_bounds['kernel_options'][best_kernel_idx]
            
            self.best_params = {
                'C': best_C,
                'epsilon': best_epsilon,
                'gamma': best_gamma,
                'kernel': best_kernel
            }
            
            print(f"\nOptimization completed!")
            print(f"Best parameters: {self.best_params}")
            print(f"Best fitness: {best_ind.fitness.values[0]:.6f}")
            
            return self.best_params
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            return None
    
    def train_final_model(self):
        """
        Train final SVR model with optimized parameters.
        
        Returns:
            bool: Success status
        """
        if self.best_params is None:
            print("No optimized parameters available. Please run optimization first.")
            return False
        
        try:
            # Prepare data
            X = self.processed_data[self.feature_columns].values
            y = self.processed_data[self.target_column].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create and train final model
            self.best_model = SVR(
                C=self.best_params['C'],
                epsilon=self.best_params['epsilon'],
                gamma=self.best_params['gamma'],
                kernel=self.best_params['kernel']
            )
            
            self.best_model.fit(X_scaled, y)
            
            # Generate predictions
            y_pred = self.best_model.predict(X_scaled)
            
            # Calculate evaluation metrics
            self.evaluation_results = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred)
            }
            
            # Calculate MAPE
            mask = y != 0
            if np.any(mask):
                mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
                self.evaluation_results['mape'] = mape
            
            print(f"\nFinal model performance:")
            for metric, value in self.evaluation_results.items():
                print(f"{metric.upper()}: {value:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error training final model: {e}")
            return False
    
    def predict(self, X_new):
        """
        Make predictions with the trained model.
        
        Args:
            X_new: New feature data
            
        Returns:
            np.array: Predictions
        """
        if self.best_model is None:
            print("No trained model available. Please train the model first.")
            return None
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X_new)
            
            # Make predictions
            predictions = self.best_model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def plot_results(self, save_plots=True):
        """
        Create visualization plots for optimization and model results.
        
        Args:
            save_plots: Whether to save plots to files
        """
        if not self.optimization_history or self.best_model is None:
            print("No results to plot. Please run optimization and training first.")
            return
        
        try:
            # Plot 1: Optimization history
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            generations = range(1, len(self.optimization_history) + 1)
            max_fitness = [record['max'] for record in self.optimization_history]
            avg_fitness = [record['avg'] for record in self.optimization_history]
            
            plt.plot(generations, max_fitness, label='Best Fitness', linewidth=2)
            plt.plot(generations, avg_fitness, label='Average Fitness', linewidth=2, alpha=0.7)
            plt.title('GA Optimization Progress')
            plt.xlabel('Generation')
            plt.ylabel('Fitness (Negative MSE)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Parameter evolution (best individual)
            plt.subplot(1, 3, 2)
            # This would require tracking best individual history
            # For now, show final parameters as bar chart
            params = list(self.best_params.keys())
            values = [self.best_params[p] if isinstance(self.best_params[p], (int, float)) else 0 for p in params]
            
            plt.bar(range(len(params)), values)
            plt.title('Optimized Parameters')
            plt.xlabel('Parameter')
            plt.ylabel('Value')
            plt.xticks(range(len(params)), params, rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Model predictions vs actual
            plt.subplot(1, 3, 3)
            X = self.processed_data[self.feature_columns].values
            y = self.processed_data[self.target_column].values
            X_scaled = self.scaler.transform(X)
            y_pred = self.best_model.predict(X_scaled)
            
            plt.scatter(y, y_pred, alpha=0.6)
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.title('Predictions vs Actual')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(self.output_folder, 'optimization_results.png'), 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
            # Plot 4: Time series prediction comparison
            plt.figure(figsize=(12, 6))
            
            time_index = range(len(y))
            plt.plot(time_index, y, label='Actual', linewidth=2, alpha=0.8)
            plt.plot(time_index, y_pred, label='Predicted', linewidth=2, alpha=0.8)
            plt.title(f'SVR Time Series Predictions - {self.target_column}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_plots:
                plt.savefig(os.path.join(self.output_folder, 'time_series_fit.png'), 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def save_model(self, filename=None):
        """
        Save the trained model and optimization results.
        
        Args:
            filename: Base filename for saved files
            
        Returns:
            str: Path to saved model
        """
        if self.best_model is None:
            print("No model to save. Please train the model first.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ga_svr_model_{timestamp}"
        
        try:
            # Save model and components
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'best_params': self.best_params,
                'ga_params': self.ga_params,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'evaluation_results': self.evaluation_results,
                'optimization_history': self.optimization_history
            }
            
            model_path = os.path.join(self.output_folder, f"{filename}.joblib")
            joblib.dump(model_data, model_path)
            
            print(f"Model saved to: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
    
    def load_model(self, model_path):
        """
        Load a previously saved model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            bool: Success status
        """
        try:
            model_data = joblib.load(model_path)
            
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.best_params = model_data['best_params']
            self.ga_params = model_data['ga_params']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.evaluation_results = model_data.get('evaluation_results')
            self.optimization_history = model_data.get('optimization_history', [])
            
            print(f"Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def export_results(self, filename=None):
        """
        Export optimization and prediction results to Excel.
        
        Args:
            filename: Output filename
            
        Returns:
            str: Path to exported file
        """
        if self.best_model is None:
            print("No results to export. Please train the model first.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ga_svr_results_{timestamp}.xlsx"
        
        try:
            # Prepare results data
            X = self.processed_data[self.feature_columns].values
            y = self.processed_data[self.target_column].values
            X_scaled = self.scaler.transform(X)
            y_pred = self.best_model.predict(X_scaled)
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Time_Step': range(len(y)),
                'Actual': y,
                'Predicted': y_pred,
                'Residual': y - y_pred,
                'Absolute_Error': np.abs(y - y_pred),
                'Percentage_Error': ((y - y_pred) / y) * 100
            })
            
            # Export to Excel
            excel_path = os.path.join(self.output_folder, filename)
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Main results
                results_df.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Optimized parameters
                params_df = pd.DataFrame(list(self.best_params.items()), columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Best_Parameters', index=False)
                
                # GA parameters
                ga_params_df = pd.DataFrame(list(self.ga_params.items()), columns=['GA_Parameter', 'Value'])
                ga_params_df.to_excel(writer, sheet_name='GA_Settings', index=False)
                
                # Evaluation metrics
                if self.evaluation_results:
                    eval_df = pd.DataFrame(list(self.evaluation_results.items()), columns=['Metric', 'Value'])
                    eval_df.to_excel(writer, sheet_name='Evaluation', index=False)
                
                # Optimization history
                if self.optimization_history:
                    history_df = pd.DataFrame(self.optimization_history)
                    history_df['Generation'] = range(1, len(history_df) + 1)
                    history_df.to_excel(writer, sheet_name='Optimization_History', index=False)
            
            print(f"Results exported to: {excel_path}")
            return excel_path
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return None


class GASVROptimizer:
    """
    Main class for running GA-SVR optimization on multiple sheets.
    """
    
    def __init__(self):
        self.excel_file = None
        self.sheets = []
        self.selected_sheets = []
        self.results = {}
    
    def load_excel_file(self, file_path):
        """
        Load Excel file and get sheet names.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            bool: Success status
        """
        try:
            self.excel_file = file_path
            xl = pd.ExcelFile(file_path)
            self.sheets = xl.sheet_names
            
            print(f"Loaded file: {os.path.basename(file_path)}")
            print(f"Found {len(self.sheets)} sheets")
            
            return True
            
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return False
    
    def optimize_all_sheets(self, progress_callback=None):
        """
        Run GA-SVR optimization on all selected sheets.
        
        Args:
            progress_callback: Function to report progress
            
        Returns:
            dict: Results for all sheets
        """
        if not self.selected_sheets:
            print("No sheets selected for optimization")
            return {}
        
        self.results = {}
        
        for i, sheet_name in enumerate(self.selected_sheets):
            print(f"\n--- Optimizing sheet {i+1}/{len(self.selected_sheets)}: {sheet_name} ---")
            
            if progress_callback:
                progress_callback(f"Processing sheet: {sheet_name}")
            
            try:
                # Create GA-SVR instance for this sheet
                sheet_dir = os.path.join("GA_SVR_Results", sheet_name.replace(" ", "_"))
                ga_svr = GeneticAlgorithmSVR(output_folder=sheet_dir)
                
                # Load data
                if ga_svr.load_data(self.excel_file, sheet_name):
                    # Optimize parameters
                    best_params = ga_svr.optimize_parameters()
                    
                    if best_params:
                        # Train final model
                        if ga_svr.train_final_model():
                            # Save results
                            ga_svr.plot_results()
                            model_path = ga_svr.save_model()
                            results_path = ga_svr.export_results()
                            
                            self.results[sheet_name] = {
                                'success': True,
                                'best_params': best_params,
                                'evaluation': ga_svr.evaluation_results,
                                'model_path': model_path,
                                'results_path': results_path
                            }
                        else:
                            self.results[sheet_name] = {'success': False, 'error': 'Model training failed'}
                    else:
                        self.results[sheet_name] = {'success': False, 'error': 'Parameter optimization failed'}
                else:
                    self.results[sheet_name] = {'success': False, 'error': 'Data loading failed'}
                    
            except Exception as e:
                self.results[sheet_name] = {'success': False, 'error': str(e)}
                print(f"Error processing sheet {sheet_name}: {e}")
        
        return self.results


class GASVROptimizerGUI:
    """
    GUI application for GA-SVR optimization.
    """
    
    def __init__(self, master):
        self.master = master
        master.title("SVR 参数优化工具 (GA)")
        master.geometry("800x600")
        
        # Initialize optimizer
        self.optimizer = GASVROptimizer()
        
        # State variables
        self.file_var = tk.StringVar()
        
        # Create GUI elements
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        main_frame = ttk.Frame(self.master, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Excel文件:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(file_frame, text="浏览...", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        file_frame.columnconfigure(1, weight=1)
        
        # Sheet selection
        sheet_frame = ttk.LabelFrame(main_frame, text="工作表选择", padding=10)
        sheet_frame.pack(fill=tk.X, pady=5)
        
        self.sheet_listbox = tk.Listbox(sheet_frame, selectmode=tk.MULTIPLE, height=6)
        self.sheet_listbox.pack(fill=tk.X, padx=5, pady=5)
        
        # GA parameters
        param_frame = ttk.LabelFrame(main_frame, text="优化参数", padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        # Parameter controls
        ttk.Label(param_frame, text="种群大小:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.pop_size_var = tk.IntVar(value=50)
        ttk.Entry(param_frame, textvariable=self.pop_size_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(param_frame, text="迭代次数:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.gen_size_var = tk.IntVar(value=50)
        ttk.Entry(param_frame, textvariable=self.gen_size_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(param_frame, text="交叉验证折数:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.cv_splits_var = tk.IntVar(value=5)
        ttk.Entry(param_frame, textvariable=self.cv_splits_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Control buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.run_btn = ttk.Button(btn_frame, text="开始优化", command=self.start_optimization)
        self.run_btn.pack(side=tk.RIGHT, padx=5)
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def browse_file(self):
        """Browse for Excel file."""
        file_path = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        
        if file_path:
            self.file_var.set(file_path)
            self.load_sheets(file_path)
    
    def load_sheets(self, file_path):
        """Load sheets from Excel file."""
        if self.optimizer.load_excel_file(file_path):
            # Clear and populate listbox
            self.sheet_listbox.delete(0, tk.END)
            for sheet in self.optimizer.sheets:
                self.sheet_listbox.insert(tk.END, sheet)
            
            self.log(f"已加载文件: {os.path.basename(file_path)}")
            self.log(f"找到 {len(self.optimizer.sheets)} 个工作表")
        else:
            messagebox.showerror("错误", "加载工作表时出错")
    
    def log(self, message):
        """Add message to log area."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.master.update_idletasks()
    
    def start_optimization(self):
        """Start optimization process."""
        # Get selected sheets
        selected_indices = self.sheet_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("警告", "请至少选择一个工作表")
            return
        
        self.optimizer.selected_sheets = [self.optimizer.sheets[i] for i in selected_indices]
        self.log(f"选中工作表: {', '.join(self.optimizer.selected_sheets)}")
        
        # Update GA parameters
        ga_params = {
            'population_size': self.pop_size_var.get(),
            'generations': self.gen_size_var.get(),
            'cv_splits': self.cv_splits_var.get()
        }
        
        # Disable button during optimization
        self.run_btn.config(state=tk.DISABLED)
        
        # Start optimization in thread
        def optimization_thread():
            try:
                def progress_callback(message):
                    self.master.after(0, lambda: self.log(message))
                
                results = self.optimizer.optimize_all_sheets(progress_callback)
                
                # Show results
                success_count = sum(1 for r in results.values() if r.get('success', False))
                self.master.after(0, lambda: self.log(f"\n优化完成! 成功: {success_count}/{len(results)}"))
                self.master.after(0, lambda: messagebox.showinfo("完成", "SVR参数优化完成"))
                
            except Exception as e:
                self.master.after(0, lambda: self.log(f"优化失败: {str(e)}"))
                self.master.after(0, lambda: messagebox.showerror("错误", f"优化过程中出错: {e}"))
            
            finally:
                self.master.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
        
        threading.Thread(target=optimization_thread, daemon=True).start()


def create_gui():
    """Create and run the GUI application."""
    root = tk.Tk()
    app = GASVROptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        file_path = sys.argv[1]
        sheet_name = sys.argv[2] if len(sys.argv) > 2 else None
        
        ga_svr = GeneticAlgorithmSVR()
        
        if ga_svr.load_data(file_path, sheet_name):
            print("Data loaded successfully")
            
            best_params = ga_svr.optimize_parameters()
            if best_params:
                print("Optimization completed successfully")
                
                if ga_svr.train_final_model():
                    print("Final model trained successfully")
                    ga_svr.plot_results()
                    ga_svr.save_model()
                    ga_svr.export_results()
                else:
                    print("Final model training failed")
            else:
                print("Optimization failed")
        else:
            print("Data loading failed")
    else:
        # GUI usage
        if not DEAP_AVAILABLE:
            print("Warning: DEAP library not available. Genetic algorithm optimization will not work.")
        create_gui()
