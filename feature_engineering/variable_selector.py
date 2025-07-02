#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variable Selector for Agricultural Data Analysis

This module provides comprehensive variable selection capabilities including:
- Correlation analysis with customizable thresholds
- VIF (Variance Inflation Factor) based iterative elimination
- Factor analysis for dimensionality reduction
- Multi-sheet Excel data processing and reshaping
- GUI interface for easy usage
- Statistical validation and reporting

Combines functionality from:
- 变量剔除.py

Author: Thesis Research Project
Date: 2025
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import traceback
import os

# Optional: Factor Analysis
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False

# Configure matplotlib for Chinese font support
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Warning: Could not set Chinese font for plots. Chinese characters might not display correctly.")
    plt.rcParams['axes.unicode_minus'] = False


class VariableSelector:
    """
    Comprehensive variable selector for agricultural research data.
    
    Supports correlation analysis, VIF-based elimination, and factor analysis.
    """
    
    def __init__(self):
        """Initialize with default parameters."""
        self.default_corr_threshold = 0.9
        self.default_vif_threshold = 10.0
        self.processed_data = None
        self.original_data = None
        self.numeric_cols = []
        self.final_selected_vars = []
        self.factor_loadings = None
        self.scaler = None
        
    def read_and_combine_sheets(self, excel_file, log_callback=None):
        """
        Read multi-sheet Excel file and convert to long format DataFrame.
        
        Args:
            excel_file: Path to Excel file
            log_callback: Function to log messages (optional)
            
        Returns:
            pd.DataFrame: Long format DataFrame with Province, Year, Indicator, Value columns
        """
        if log_callback is None:
            log_callback = print
            
        log_callback(f"开始读取 Excel 文件: {excel_file}")
        
        try:
            xls = pd.ExcelFile(excel_file)
            all_sheets = xls.sheet_names
            log_callback(f"找到 {len(all_sheets)} 个 Sheet: {', '.join(all_sheets)}")
        except FileNotFoundError:
            log_callback(f"错误：文件 '{excel_file}' 未找到。")
            return None
        except Exception as e:
            log_callback(f"读取 Excel 文件时出错: {e}")
            return None
        
        big_list = []
        problematic_sheets = []
        
        for sheet in all_sheets:
            try:
                df_sheet = pd.read_excel(excel_file, sheet_name=sheet, header=None)
                if df_sheet.shape[0] < 2 or df_sheet.shape[1] < 2:
                    log_callback(f"警告：Sheet '{sheet}' 行数或列数不足，跳过。Shape: {df_sheet.shape}")
                    problematic_sheets.append(sheet)
                    continue
                
                # Extract years from first row
                years_raw = df_sheet.iloc[0, 1:].tolist()
                years_cleaned = []
                valid_year_indices = []
                
                for idx, yr_val in enumerate(years_raw):
                    try:
                        if pd.isna(yr_val):
                            raise ValueError("空值")
                        year_int = int(float(yr_val))
                        if 1900 < year_int < 2100:
                            years_cleaned.append(str(year_int))
                            valid_year_indices.append(idx + 1)
                        else:
                            log_callback(f"警告：Sheet '{sheet}' 第1行第 {idx+2} 列值 '{yr_val}' 不在合理年份范围，忽略。")
                    except (ValueError, TypeError):
                        log_callback(f"警告：Sheet '{sheet}' 第1行第 {idx+2} 列非年份值 '{yr_val}'，忽略。")
                
                if not years_cleaned:
                    log_callback(f"警告：Sheet '{sheet}' 未提取到有效年份，跳过。")
                    problematic_sheets.append(sheet)
                    continue
                
                # Extract indicators and data
                years = years_cleaned
                indicators = df_sheet.iloc[1:, 0].tolist()
                data_values = df_sheet.iloc[1:, valid_year_indices].values
                
                if len(indicators) != data_values.shape[0] or len(years) != data_values.shape[1]:
                    log_callback(f"警告：Sheet '{sheet}' 维度不匹配，跳过。指标:{len(indicators)}, 年:{len(years)}, 数据:{data_values.shape}")
                    problematic_sheets.append(sheet)
                    continue
                
                # Create temporary DataFrame
                temp_df = pd.DataFrame(data_values, columns=years, index=indicators)
                temp_df['Province'] = sheet
                big_list.append(temp_df)
                
            except Exception as e:
                log_callback(f"处理 Sheet '{sheet}' 时出错: {e}，已跳过。")
                problematic_sheets.append(sheet)
        
        if not big_list:
            log_callback("错误：未能成功处理任何 Sheet。请检查 Excel 文件结构。")
            return None
        
        if problematic_sheets:
            log_callback(f"\n注意：以下 Sheet 处理有误或被跳过： {', '.join(problematic_sheets)}")
        
        # Combine all sheets
        full_df_wide_temp = pd.concat(big_list, axis=0)
        full_df_wide_temp = full_df_wide_temp.reset_index().rename(columns={'index': 'Indicator'})
        
        # Convert to long format
        year_cols = [col for col in full_df_wide_temp.columns if col not in ['Province', 'Indicator']]
        valid_year_cols = []
        for yr in year_cols:
            try:
                int(yr)
                valid_year_cols.append(yr)
            except ValueError:
                log_callback(f"警告：列名 '{yr}' 非有效年份格式，转换长表时忽略。")
        
        if not valid_year_cols:
            log_callback("错误：合并数据中找不到有效年份列。检查 Excel 首行年份。")
            return None
        
        full_df_long = pd.melt(
            full_df_wide_temp, id_vars=['Province', 'Indicator'], value_vars=valid_year_cols,
            var_name='Year', value_name='Value'
        )
        
        # Convert to numeric
        full_df_long['Year'] = pd.to_numeric(full_df_long['Year'], errors='coerce')
        full_df_long['Value'] = pd.to_numeric(full_df_long['Value'], errors='coerce')
        
        # Drop rows with invalid data
        rows_before = len(full_df_long)
        full_df_long.dropna(subset=['Year', 'Value'], inplace=True)
        rows_after = len(full_df_long)
        if rows_before > rows_after:
            log_callback(f"提示：因年份或数值转换失败，移除 {rows_before - rows_after} 行数据。")
        
        if not full_df_long.empty:
            full_df_long['Year'] = full_df_long['Year'].astype(int)
        else:
            log_callback("警告：数据处理后为空。")
            return None
        
        log_callback("数据读取和初步合并完成。")
        return full_df_long
    
    def reshape_to_wide(self, long_df, log_callback=None):
        """
        Convert long format DataFrame to wide format.
        
        Args:
            long_df: Long format DataFrame
            log_callback: Function to log messages (optional)
            
        Returns:
            pd.DataFrame: Wide format DataFrame
        """
        if log_callback is None:
            log_callback = print
            
        log_callback("开始转换数据为宽表格式...")
        try:
            pivot_df = long_df.pivot_table(
                index=['Province', 'Year'], columns='Indicator', values='Value'
            )
            pivot_df = pivot_df.reset_index()
            pivot_df.columns.name = None
            log_callback(f"宽表转换成功，Shape: {pivot_df.shape}")
            return pivot_df
        except Exception as e:
            log_callback(f"宽表转换失败: {e}")
            duplicates = long_df[long_df.duplicated(subset=['Province', 'Year', 'Indicator'], keep=False)]
            if not duplicates.empty:
                log_callback("检测到重复记录 (相同省份-年份-指标有多个值)，无法转换。请检查数据。")
                log_callback(f"重复记录示例:\n{duplicates.head()}")
            return None
    
    def preprocess_data(self, df, log_callback=None, id_cols=['Province', 'Year']):
        """
        Handle missing values and scale numeric data.
        
        Args:
            df: Input DataFrame
            log_callback: Function to log messages (optional)
            id_cols: ID columns to exclude from processing
            
        Returns:
            tuple: (processed_df, numeric_columns, scaler)
        """
        if log_callback is None:
            log_callback = print
            
        log_callback("\n--- 数据预处理 ---")
        numeric_cols = df.columns.difference(id_cols).tolist()
        if not numeric_cols:
            log_callback("错误：宽表中找不到数值型指标列。")
            return None, None, None
        
        log_callback(f"待处理指标列 ({len(numeric_cols)}个): {', '.join(numeric_cols)}")
        
        # Handle missing values
        missing_summary = df[numeric_cols].isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if not missing_summary.empty:
            log_callback(f"缺失值检查 (处理前):\n{missing_summary.to_string()}")
            rows_before = len(df)
            df.dropna(subset=numeric_cols, inplace=True)
            rows_after = len(df)
            log_callback(f"已删除含缺失值的行数: {rows_before - rows_after}。剩余行数: {rows_after}")
            if df.empty:
                log_callback("错误：删除缺失值后数据为空。")
                return None, None, None
        else:
            log_callback("数据中无缺失值。")
        
        # Min-Max scaling
        log_callback("进行 Min-Max 归一化...")
        scaler = MinMaxScaler()
        try:
            df_copy = df.copy()
            df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
            log_callback("归一化完成。")
            return df_copy, numeric_cols, scaler
        except Exception as e:
            log_callback(f"归一化时出错: {e}")
            return None, None, None
    
    def run_correlation_analysis(self, df, cols, threshold, log_callback=None):
        """
        Perform correlation analysis and visualization.
        
        Args:
            df: Input DataFrame
            cols: Columns to analyze
            threshold: Correlation threshold
            log_callback: Function to log messages (optional)
            
        Returns:
            tuple: (correlation_matrix, high_correlation_variables_set)
        """
        if log_callback is None:
            log_callback = print
            
        log_callback(f"\n--- 相关性分析 (阈值={threshold}) ---")
        if not cols:
            log_callback("没有可用于相关性分析的列。")
            return None, set()
        
        try:
            corr_matrix = df[cols].corr()
            
            # Create heatmap if manageable number of variables
            if len(cols) <= 50:
                log_callback("正在生成相关性热力图...")
                plt.figure(figsize=(max(10, len(cols)*0.4), max(8, len(cols)*0.4)))
                sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f")
                plt.title('指标相关性热力图 (归一化后数据)')
                plt.tight_layout()
                plt.show(block=False)
                log_callback("热力图已生成 (可能在后台窗口)。")
            else:
                log_callback("变量过多 (>50)，跳过生成热力图。")
            
            # Find high correlation pairs
            high_corr_pairs = []
            high_corr_vars_set = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > threshold:
                        high_corr_pairs.append((col1, col2, corr_value))
                        high_corr_vars_set.update([col1, col2])
            
            if high_corr_pairs:
                log_callback(f"\n发现变量对间存在强相关性 (>|{threshold}|):")
                sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
                for pair in sorted_pairs[:20]:  # Show top 20
                    log_callback(f"- {pair[0]} 与 {pair[1]}: {pair[2]:.3f}")
                if len(sorted_pairs) > 20:
                    log_callback("...")
                log_callback(f"涉及强相关的变量集合: {high_corr_vars_set}")
            else:
                log_callback(f"未发现绝对相关系数 > {threshold} 的变量对。")
            
            return corr_matrix, high_corr_vars_set
            
        except Exception as e:
            log_callback(f"相关性分析时出错: {e}")
            return None, set()
    
    def run_factor_analysis(self, df, cols, log_callback=None, n_factors_method='eigenvalue'):
        """
        Perform factor analysis if factor_analyzer is available.
        
        Args:
            df: Input DataFrame
            cols: Columns to analyze
            log_callback: Function to log messages (optional)
            n_factors_method: Method to determine number of factors
            
        Returns:
            pd.DataFrame: Factor loadings matrix or None
        """
        if log_callback is None:
            log_callback = print
            
        if not FACTOR_ANALYZER_AVAILABLE:
            log_callback("\n因子分析跳过：'factor_analyzer' 库不可用。")
            return None
        if not cols:
            log_callback("\n因子分析跳过：没有可用的列。")
            return None
        
        log_callback("\n--- 因子分析 ---")
        X = df[cols]
        
        # Suitability tests
        try:
            log_callback("执行适用性检验 (Bartlett & KMO)...")
            chi_square_value, p_value = calculate_bartlett_sphericity(X)
            log_callback(f"Bartlett 球形检验: Chi2={chi_square_value:.3f}, p={p_value:.3e}")
            if p_value > 0.05:
                log_callback("警告：Bartlett 检验不显著 (p > 0.05)，数据可能不适合因子分析。")
            
            kmo_all, kmo_model = calculate_kmo(X)
            log_callback(f"KMO 检验总体值: {kmo_model:.3f}")
            if kmo_model < 0.6:
                log_callback("警告：KMO 值较低 (< 0.6)，数据可能不太适合因子分析。")
        except Exception as e:
            log_callback(f"执行 KMO 或 Bartlett 检验时出错: {e}")
            return None
        
        # Determine number of factors
        try:
            log_callback("计算特征值以确定因子数量...")
            fa_check = FactorAnalyzer(n_factors=len(cols), rotation=None)
            fa_check.fit(X)
            ev, v = fa_check.get_eigenvalues()
            
            if n_factors_method == 'eigenvalue':
                n_factors = sum(ev > 1)
                log_callback(f"根据特征值 > 1 原则，建议因子数为 {n_factors}。")
            
            if n_factors == 0:
                n_factors = 1
            if n_factors >= len(cols):
                log_callback(f"警告：建议因子数({n_factors})>=变量数({len(cols)})，调整为{len(cols)-1}。")
                n_factors = max(1, len(cols) - 1)
            
            # Scree plot
            log_callback("正在生成碎石图...")
            plt.figure(figsize=(8, 5))
            plt.scatter(range(1, X.shape[1] + 1), ev)
            plt.plot(range(1, X.shape[1] + 1), ev)
            plt.title('碎石图 (Scree Plot)')
            plt.xlabel('因子序号')
            plt.ylabel('特征值 (Eigenvalue)')
            plt.axhline(y=1, color='r', linestyle='--', label='Eigenvalue=1')
            plt.grid()
            plt.legend()
            plt.show(block=False)
            log_callback("碎石图已生成 (可能在后台窗口)。")
            
        except Exception as e:
            log_callback(f"确定因子数量或绘图时出错: {e}")
            return None
        
        # Perform Factor Analysis
        log_callback(f"执行因子分析，提取 {n_factors} 个因子，使用 Varimax 旋转...")
        fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
        try:
            fa.fit(X)
        except Exception as e:
            log_callback(f"拟合因子分析模型时出错：{e}")
            return None
        
        # Results
        loadings = pd.DataFrame(fa.loadings_, index=cols, columns=[f'Factor{i+1}' for i in range(n_factors)])
        log_callback("\n因子载荷矩阵 (Factor Loadings):")
        log_callback(loadings.round(3).to_string())
        
        variance_explained = fa.get_factor_variance()
        variance_df = pd.DataFrame({
            'SS Loadings': variance_explained[0], 
            'Proportion Var': variance_explained[1],
            'Cumulative Var': variance_explained[2]
        }, index=[f'Factor{i+1}' for i in range(n_factors)])
        log_callback("\n因子方差解释:")
        log_callback(variance_df.round(3).to_string())
        log_callback(f"总方差解释率: {variance_df['Cumulative Var'].iloc[-1]:.3f}")
        
        return loadings
    
    def calculate_vif(self, df, cols):
        """
        Calculate VIF for specified columns.
        
        Args:
            df: Input DataFrame
            cols: Columns to calculate VIF for
            
        Returns:
            pd.DataFrame: VIF data sorted by VIF value
        """
        X = df[cols].copy()
        X['Intercept'] = 1.0
        vif_data = pd.DataFrame()
        vif_data["Variable"] = cols
        vif_values = []
        
        for i in range(len(cols)):
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_values.append(vif)
            except Exception:
                # Handle perfect multicollinearity
                vif_values.append(np.inf)
        
        vif_data["VIF"] = vif_values
        return vif_data.sort_values(by='VIF', ascending=False).reset_index(drop=True)
    
    def iterative_vif_elimination(self, df, cols, threshold, log_callback=None):
        """
        Iteratively remove variables with VIF above threshold.
        
        Args:
            df: Input DataFrame
            cols: Columns to analyze
            threshold: VIF threshold
            log_callback: Function to log messages (optional)
            
        Returns:
            list: Final selected variables
        """
        if log_callback is None:
            log_callback = print
            
        log_callback(f"\n--- VIF 逐步剔除 (阈值 = {threshold}) ---")
        variables = cols.copy()
        eliminated_count = 0
        
        while True:
            if len(variables) <= 1:
                log_callback("变量数<=1，停止 VIF 剔除。")
                break
            
            vif_df = self.calculate_vif(df, variables)
            max_vif_row = vif_df.iloc[0]
            max_vif = max_vif_row['VIF']
            
            if max_vif > threshold and np.isfinite(max_vif):
                var_to_remove = max_vif_row['Variable']
                log_callback(f"剔除 '{var_to_remove}' (VIF = {max_vif:.2f})")
                variables.remove(var_to_remove)
                eliminated_count += 1
            elif np.isinf(max_vif):
                var_to_remove = max_vif_row['Variable']
                log_callback(f"剔除 '{var_to_remove}' (VIF = inf, 完全共线性)")
                variables.remove(var_to_remove)
                eliminated_count += 1
            else:
                log_callback(f"所有剩余变量 VIF <= {threshold}。")
                break
        
        log_callback(f"\nVIF 剔除完成，共剔除 {eliminated_count} 个变量。")
        log_callback(f"最终保留变量 ({len(variables)}个): {', '.join(variables)}")
        
        if variables:
            final_vif_df = self.calculate_vif(df, variables)
            log_callback("\n最终变量 VIF 值:")
            log_callback(final_vif_df.round(3).to_string())
        
        return variables
    
    def perform_complete_analysis(self, excel_file, variables_to_remove=None, 
                                corr_threshold=None, vif_threshold=None, 
                                run_fa=True, log_callback=None):
        """
        Perform complete variable selection analysis.
        
        Args:
            excel_file: Path to Excel file
            variables_to_remove: List of variables to remove initially
            corr_threshold: Correlation threshold (default: 0.9)
            vif_threshold: VIF threshold (default: 10.0)
            run_fa: Whether to run factor analysis
            log_callback: Function to log messages (optional)
            
        Returns:
            dict: Analysis results
        """
        if log_callback is None:
            log_callback = print
            
        if corr_threshold is None:
            corr_threshold = self.default_corr_threshold
        if vif_threshold is None:
            vif_threshold = self.default_vif_threshold
        
        try:
            # Step 1: Read and combine data
            long_df = self.read_and_combine_sheets(excel_file, log_callback)
            if long_df is None or long_df.empty:
                raise ValueError("数据读取失败或为空。")
            
            # Step 2: Remove specified variables
            if variables_to_remove:
                log_callback(f"\n剔除指定变量: {', '.join(variables_to_remove)}")
                initial_rows = len(long_df)
                long_df = long_df[~long_df['Indicator'].isin(variables_to_remove)]
                log_callback(f"剔除后剩余数据比例: {len(long_df) / initial_rows:.2%}" if initial_rows else "N/A")
                if long_df.empty:
                    raise ValueError("剔除变量后数据为空。")
            
            current_indicators = long_df['Indicator'].unique().tolist()
            if not current_indicators:
                raise ValueError("没有剩余的指标可供分析。")
            log_callback(f"当前可用指标 ({len(current_indicators)}个): {', '.join(current_indicators)}")
            
            # Step 3: Reshape to wide format
            wide_df = self.reshape_to_wide(long_df, log_callback)
            if wide_df is None or wide_df.empty:
                raise ValueError("转换为宽表失败。")
            
            # Store original data
            self.original_data = wide_df.copy()
            
            # Step 4: Preprocess data
            processed_df, numeric_cols, scaler = self.preprocess_data(wide_df, log_callback)
            if processed_df is None or not numeric_cols:
                raise ValueError("数据预处理失败。")
            
            self.processed_data = processed_df
            self.numeric_cols = numeric_cols
            self.scaler = scaler
            
            # Step 5: Correlation analysis
            corr_matrix, high_corr_vars = self.run_correlation_analysis(
                processed_df, numeric_cols, corr_threshold, log_callback
            )
            
            current_vars_for_analysis = numeric_cols.copy()
            
            # Step 6: Factor analysis (optional)
            if run_fa and FACTOR_ANALYZER_AVAILABLE:
                self.factor_loadings = self.run_factor_analysis(
                    processed_df, current_vars_for_analysis, log_callback
                )
            
            # Step 7: VIF elimination
            self.final_selected_vars = self.iterative_vif_elimination(
                processed_df, current_vars_for_analysis, vif_threshold, log_callback
            )
            
            if not self.final_selected_vars:
                log_callback("\n警告：VIF 剔除后没有剩余变量。")
                raise ValueError("没有变量通过 VIF 筛选。")
            
            log_callback("\n=== 分析成功完成 ===")
            
            return {
                'success': True,
                'final_variables': self.final_selected_vars,
                'correlation_matrix': corr_matrix,
                'high_correlation_vars': high_corr_vars,
                'factor_loadings': self.factor_loadings,
                'original_data': self.original_data,
                'processed_data': self.processed_data,
                'scaler': self.scaler
            }
            
        except Exception as e:
            log_callback(f"\n!!! 分析过程中发生错误 !!!")
            log_callback(f"错误类型: {type(e).__name__}")
            log_callback(f"错误信息: {e}")
            log_callback(f"详细追溯:\n{traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'final_variables': [],
                'correlation_matrix': None,
                'high_correlation_vars': set(),
                'factor_loadings': None,
                'original_data': None,
                'processed_data': None,
                'scaler': None
            }
    
    def save_results(self, output_filepath, use_original_data=True):
        """
        Save final selected variables to CSV file.
        
        Args:
            output_filepath: Path to save results
            use_original_data: Whether to use original (unscaled) data
            
        Returns:
            bool: Success status
        """
        if not self.final_selected_vars or self.original_data is None:
            print("没有可保存的最终变量数据。请先成功运行分析。")
            return False
        
        try:
            print(f"\n准备保存最终结果到: {output_filepath}")
            
            # Select columns to save
            cols_to_save = ['Province', 'Year'] + self.final_selected_vars
            
            # Use original or processed data
            data_to_save = self.original_data if use_original_data else self.processed_data
            
            # Filter and clean data
            final_df_to_save = data_to_save[cols_to_save].copy()
            
            rows_before = len(final_df_to_save)
            final_df_to_save.dropna(inplace=True)
            rows_after = len(final_df_to_save)
            
            if rows_before > rows_after:
                print(f"注意：保存最终数据时，因所选变量存在缺失值，额外删除了 {rows_before - rows_after} 行。")
            
            # Save main results
            final_df_to_save.to_csv(output_filepath, index=False, encoding='utf-8-sig')
            print(f"最终数据已成功保存 (Shape: {final_df_to_save.shape})。")
            
            # Save factor loadings if available
            if self.factor_loadings is not None:
                try:
                    loadings_filepath = output_filepath.replace(".csv", "_factor_loadings.csv")
                    final_loadings = self.factor_loadings.loc[self.final_selected_vars]
                    final_loadings.to_csv(loadings_filepath, encoding='utf-8-sig')
                    print(f"最终变量的因子载荷已保存到: {loadings_filepath}")
                except Exception as load_e:
                    print(f"保存因子载荷时出错: {load_e}")
            
            return True
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
            return False


class VariableSelectorGUI:
    """
    GUI application for variable selection analysis.
    """
    
    def __init__(self, master):
        self.master = master
        master.title("多维变量分析工具")
        master.geometry("800x600")
        
        # State variables
        self.excel_path = tk.StringVar()
        self.all_indicators = []
        self.selector = VariableSelector()
        
        # Create GUI elements
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # File selection frame
        file_frame = ttk.LabelFrame(self.master, text="1. 选择文件")
        file_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ttk.Label(file_frame, text="Excel 文件:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.path_entry = ttk.Entry(file_frame, textvariable=self.excel_path, width=60)
        self.path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.browse_button = ttk.Button(file_frame, text="浏览...", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        file_frame.columnconfigure(1, weight=1)
        
        # Variable selection & parameters frame
        param_frame = ttk.LabelFrame(self.master, text="2. 参数设置 & 变量剔除")
        param_frame.pack(padx=10, pady=5, fill=tk.X)
        
        # Indicators listbox
        ttk.Label(param_frame, text="指标列表 (选择要剔除的项):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.NW)
        self.indicator_listbox = tk.Listbox(param_frame, selectmode=tk.MULTIPLE, height=8, width=40)
        self.indicator_scrollbar = ttk.Scrollbar(param_frame, orient=tk.VERTICAL, command=self.indicator_listbox.yview)
        self.indicator_listbox.configure(yscrollcommand=self.indicator_scrollbar.set)
        self.indicator_listbox.grid(row=1, column=0, padx=(5,0), pady=5, sticky=tk.NSEW)
        self.indicator_scrollbar.grid(row=1, column=1, padx=(0,5), pady=5, sticky=tk.NS)
        
        # Parameters sub-frame
        params_subframe = ttk.Frame(param_frame)
        params_subframe.grid(row=1, column=2, padx=10, pady=5, sticky=tk.NW)
        
        ttk.Label(params_subframe, text="相关性阈值 (筛选高相关):").pack(anchor=tk.W)
        self.corr_thresh_var = tk.DoubleVar(value=0.9)
        ttk.Entry(params_subframe, textvariable=self.corr_thresh_var, width=10).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Label(params_subframe, text="VIF 阈值 (逐步剔除):").pack(anchor=tk.W)
        self.vif_thresh_var = tk.DoubleVar(value=10.0)
        ttk.Entry(params_subframe, textvariable=self.vif_thresh_var, width=10).pack(anchor=tk.W, pady=(0, 5))
        
        # Factor analysis option
        if FACTOR_ANALYZER_AVAILABLE:
            fa_frame = ttk.Frame(params_subframe)
            ttk.Label(fa_frame, text="执行因子分析:").pack(side=tk.LEFT, anchor=tk.W)
            self.run_fa_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(fa_frame, variable=self.run_fa_var).pack(side=tk.LEFT, anchor=tk.W)
            fa_frame.pack(anchor=tk.W)
        else:
            self.run_fa_var = tk.BooleanVar(value=False)
        
        param_frame.columnconfigure(0, weight=1)
        param_frame.rowconfigure(1, weight=1)
        
        # Analysis control frame
        control_frame = ttk.Frame(self.master)
        control_frame.pack(padx=10, pady=5, fill=tk.X)
        self.run_button = ttk.Button(control_frame, text="运行分析", command=self.run_analysis_thread)
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(control_frame, text="保存最终结果", state=tk.DISABLED, command=self.save_results)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Log output area
        log_frame = ttk.LabelFrame(self.master, text="3. 日志输出")
        log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.configure(state='disabled')
    
    def log_message(self, message):
        """Thread-safe message logging."""
        def _append():
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)
        self.master.after(0, _append)
    
    def browse_file(self):
        """Browse for Excel file and load indicators."""
        filepath = filedialog.askopenfilename(
            title="选择 Excel 文件",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filepath:
            self.excel_path.set(filepath)
            self.log_message(f"已选择文件: {filepath}")
            self.load_indicators()
    
    def load_indicators(self):
        """Load indicator names from selected Excel file."""
        filepath = self.excel_path.get()
        if not filepath:
            messagebox.showerror("错误", "请先选择一个 Excel 文件。")
            return
        
        self.indicator_listbox.delete(0, tk.END)
        self.all_indicators = []
        self.log_message("正在读取指标名称...")
        
        try:
            xls = pd.ExcelFile(filepath)
            first_sheet = xls.sheet_names[0]
            df_indicators = pd.read_excel(filepath, sheet_name=first_sheet, usecols=[0], header=None, skiprows=1)
            indicators = df_indicators.iloc[:, 0].dropna().unique().tolist()
            indicators = [str(ind) for ind in indicators]
            
            if not indicators:
                self.log_message("错误：在第一个 Sheet 的 A 列（从A2开始）未找到指标名称。请检查文件结构。")
                messagebox.showerror("格式错误", "未能从文件第一列读取指标名称。请确保指标名从A2单元格开始。")
                return
            
            self.all_indicators = sorted(indicators)
            for indicator in self.all_indicators:
                self.indicator_listbox.insert(tk.END, indicator)
            self.log_message(f"成功加载 {len(self.all_indicators)} 个指标。请在列表中选择要剔除的变量。")
            
        except Exception as e:
            self.log_message(f"读取指标名称时出错: {e}")
            messagebox.showerror("读取错误", f"无法读取指标列表:\n{e}")
    
    def run_analysis_thread(self):
        """Start analysis in background thread."""
        filepath = self.excel_path.get()
        if not filepath:
            messagebox.showerror("错误", "请先选择一个 Excel 文件。")
            return
        
        # Get parameters
        selected_indices = self.indicator_listbox.curselection()
        variables_to_remove = [self.indicator_listbox.get(i) for i in selected_indices]
        try:
            corr_threshold = self.corr_thresh_var.get()
            vif_threshold = self.vif_thresh_var.get()
        except tk.TclError:
            messagebox.showerror("输入错误", "相关性阈值和 VIF 阈值必须是数字。")
            return
        
        run_fa = self.run_fa_var.get()
        
        # Disable controls and clear log
        self.run_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')
        
        self.log_message("=== 开始分析 ===")
        self.log_message(f"文件: {filepath}")
        self.log_message(f"选择剔除的变量: {variables_to_remove if variables_to_remove else '无'}")
        self.log_message(f"相关性阈值: {corr_threshold}")
        self.log_message(f"VIF 阈值: {vif_threshold}")
        self.log_message(f"执行因子分析: {'是' if run_fa else '否'}")
        
        # Start analysis thread
        analysis_thread = threading.Thread(
            target=self._perform_analysis,
            args=(filepath, variables_to_remove, corr_threshold, vif_threshold, run_fa),
            daemon=True
        )
        analysis_thread.start()
    
    def _perform_analysis(self, filepath, variables_to_remove, corr_threshold, vif_threshold, run_fa):
        """Perform analysis in background thread."""
        try:
            result = self.selector.perform_complete_analysis(
                filepath, variables_to_remove, corr_threshold, vif_threshold, run_fa, self.log_message
            )
            
            if result['success']:
                self.master.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            else:
                self.master.after(0, lambda: messagebox.showerror("分析失败", f"分析过程中发生错误:\n{result['error']}"))
                
        except Exception as e:
            self.log_message(f"\n!!! 分析过程中发生严重错误 !!!")
            self.log_message(f"错误: {e}")
            self.master.after(0, lambda: messagebox.showerror("分析失败", f"分析过程中发生错误:\n{e}"))
        
        finally:
            self.master.after(0, lambda: self.run_button.config(state=tk.NORMAL))
            self.log_message("=== 分析线程结束 ===")
    
    def save_results(self):
        """Save analysis results to file."""
        if not self.selector.final_selected_vars:
            messagebox.showwarning("无结果", "没有可保存的最终变量数据。请先成功运行分析。")
            return
        
        output_filepath = filedialog.asksaveasfilename(
            title="保存最终变量数据",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not output_filepath:
            self.log_message("保存操作已取消。")
            return
        
        if self.selector.save_results(output_filepath):
            messagebox.showinfo("保存成功", f"结果已保存到:\n{output_filepath}")
        else:
            messagebox.showerror("保存失败", "无法保存结果，请检查文件路径和权限。")


def create_gui():
    """Create and run the GUI application."""
    root = tk.Tk()
    app = VariableSelectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        input_file = sys.argv[1]
        selector = VariableSelector()
        
        result = selector.perform_complete_analysis(input_file)
        if result['success']:
            print(f"分析完成。最终选择的变量: {result['final_variables']}")
            
            # Save results if output file specified
            if len(sys.argv) > 2:
                output_file = sys.argv[2]
                selector.save_results(output_file)
                print(f"结果已保存到: {output_file}")
        else:
            print(f"分析失败: {result['error']}")
    else:
        # GUI usage
        create_gui()
