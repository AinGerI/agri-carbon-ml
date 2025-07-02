#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility Tools Module

通用工具函数集合，包含常用的数据处理和文件操作功能。

功能：
- 数据清洗和预处理
- 文件批量处理
- 统计分析工具
- 图表生成工具

Author: Thesis Research Project
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class DataProcessor:
    """数据处理工具类"""
    
    @staticmethod
    def clean_data(df, fill_method='mean'):
        """数据清洗"""
        # 移除完全空的行和列
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # 数值列处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fill_method == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_method == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_method == 'forward':
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        return df
    
    @staticmethod
    def standardize_data(df):
        """数据标准化"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_std = df.copy()
        df_std[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        return df_std
    
    @staticmethod
    def normalize_data(df):
        """数据归一化到0-1范围"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_norm = df.copy()
        df_norm[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
        return df_norm


class FileManager:
    """文件管理工具类"""
    
    @staticmethod
    def batch_rename_files(folder_path, old_pattern, new_pattern):
        """批量重命名文件"""
        files = glob.glob(os.path.join(folder_path, old_pattern))
        renamed_count = 0
        
        for file_path in files:
            dir_name = os.path.dirname(file_path)
            old_name = os.path.basename(file_path)
            new_name = old_name.replace(old_pattern.replace('*', ''), new_pattern)
            new_path = os.path.join(dir_name, new_name)
            
            try:
                os.rename(file_path, new_path)
                renamed_count += 1
            except Exception as e:
                print(f"重命名失败: {old_name} -> {new_name}, 错误: {e}")
        
        return renamed_count
    
    @staticmethod
    def backup_files(source_folder, backup_folder=None):
        """备份文件夹"""
        if backup_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_folder = f"{source_folder}_backup_{timestamp}"
        
        try:
            shutil.copytree(source_folder, backup_folder)
            return backup_folder
        except Exception as e:
            print(f"备份失败: {e}")
            return None
    
    @staticmethod
    def merge_excel_files(file_list, output_path):
        """合并多个Excel文件"""
        merged_data = {}
        
        for file_path in file_list:
            try:
                xls = pd.ExcelFile(file_path)
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    key = f"{file_name}_{sheet_name}"
                    merged_data[key] = df
            except Exception as e:
                print(f"读取文件失败: {file_path}, 错误: {e}")
        
        # 保存合并结果
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, df in merged_data.items():
                df.to_excel(writer, sheet_name=sheet_name[:30], index=False)  # Excel工作表名长度限制
        
        return len(merged_data)


class StatAnalyzer:
    """统计分析工具类"""
    
    @staticmethod
    def basic_stats(df):
        """基础统计信息"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats = {
            'shape': df.shape,
            'numeric_columns': len(numeric_cols),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        if len(numeric_cols) > 0:
            stats['summary'] = df[numeric_cols].describe()
            stats['correlation'] = df[numeric_cols].corr()
        
        return stats
    
    @staticmethod
    def outlier_detection(df, method='iqr', threshold=1.5):
        """异常值检测"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = df[z_scores > threshold].index.tolist()
        
        return outliers


class PlotGenerator:
    """图表生成工具类"""
    
    @staticmethod
    def quick_plot(df, plot_type='line', save_path=None):
        """快速生成图表"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("没有数值列可以绘图")
            return
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'line':
            for col in numeric_cols[:5]:  # 最多显示5列
                plt.plot(df.index, df[col], label=col, marker='o')
            plt.legend()
        
        elif plot_type == 'bar':
            df[numeric_cols[:5]].plot(kind='bar')
        
        elif plot_type == 'hist':
            df[numeric_cols].hist(bins=20, figsize=(15, 10))
        
        elif plot_type == 'box':
            df[numeric_cols].plot(kind='box')
        
        plt.title(f'{plot_type.capitalize()} Plot')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def correlation_heatmap(df, save_path=None):
        """相关性热力图"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("需要至少2个数值列才能绘制相关性图")
            return
        
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='相关系数')
        
        # 添加标签
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        
        # 添加数值
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center')
        
        plt.title('相关性热力图')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def quick_data_analysis():
    """快速数据分析工具"""
    root = tk.Tk()
    root.withdraw()
    
    # 选择文件
    file_path = filedialog.askopenfilename(
        title="选择数据文件",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")]
    )
    
    if not file_path:
        print("未选择文件")
        return
    
    try:
        # 读取数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        else:
            df = pd.read_excel(file_path)
        
        print(f"数据加载成功: {df.shape}")
        
        # 数据清洗
        df_clean = DataProcessor.clean_data(df)
        print(f"清洗后数据: {df_clean.shape}")
        
        # 基础统计
        stats = StatAnalyzer.basic_stats(df_clean)
        print("\n=== 基础统计信息 ===")
        print(f"数据形状: {stats['shape']}")
        print(f"数值列数: {stats['numeric_columns']}")
        print(f"缺失值数: {stats['missing_values']}")
        print(f"重复行数: {stats['duplicate_rows']}")
        
        # 异常值检测
        outliers = StatAnalyzer.outlier_detection(df_clean)
        print("\n=== 异常值检测 ===")
        for col, indices in outliers.items():
            if indices:
                print(f"{col}: {len(indices)} 个异常值")
        
        # 生成图表
        PlotGenerator.quick_plot(df_clean, 'line')
        PlotGenerator.correlation_heatmap(df_clean)
        
        # 保存结果
        output_dir = os.path.dirname(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存清洗后的数据
        clean_file = os.path.join(output_dir, f"cleaned_data_{timestamp}.xlsx")
        df_clean.to_excel(clean_file, index=False)
        print(f"\n清洗后数据已保存: {clean_file}")
        
        # 保存统计报告
        report_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("数据分析报告\n")
            f.write("=" * 30 + "\n")
            f.write(f"原始文件: {file_path}\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"数据形状: {stats['shape']}\n")
            f.write(f"数值列数: {stats['numeric_columns']}\n")
            f.write(f"缺失值数: {stats['missing_values']}\n")
            f.write(f"重复行数: {stats['duplicate_rows']}\n\n")
            f.write("异常值检测结果:\n")
            for col, indices in outliers.items():
                f.write(f"{col}: {len(indices)} 个异常值\n")
        
        print(f"分析报告已保存: {report_file}")
        
    except Exception as e:
        print(f"分析失败: {e}")
    
    finally:
        root.destroy()


def file_manager_gui():
    """文件管理GUI"""
    def batch_rename():
        folder = filedialog.askdirectory(title="选择文件夹")
        if folder:
            old_pattern = input("输入旧文件名模式 (如: *.txt): ")
            new_pattern = input("输入新文件名模式: ")
            count = FileManager.batch_rename_files(folder, old_pattern, new_pattern)
            print(f"成功重命名 {count} 个文件")
    
    def merge_excel():
        files = filedialog.askopenfilenames(
            title="选择要合并的Excel文件",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if files:
            output = filedialog.asksaveasfilename(
                title="保存合并文件",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")]
            )
            if output:
                count = FileManager.merge_excel_files(files, output)
                print(f"成功合并 {count} 个工作表")
    
    root = tk.Tk()
    root.title("文件管理工具")
    root.geometry("300x200")
    
    tk.Button(root, text="批量重命名文件", command=batch_rename, width=20).pack(pady=10)
    tk.Button(root, text="合并Excel文件", command=merge_excel, width=20).pack(pady=10)
    tk.Button(root, text="快速数据分析", command=quick_data_analysis, width=20).pack(pady=10)
    
    root.mainloop()


if __name__ == "__main__":
    # 显示工具选择菜单
    choice = input("选择工具: 1-快速数据分析, 2-文件管理GUI: ")
    
    if choice == "1":
        quick_data_analysis()
    elif choice == "2":
        file_manager_gui()
    else:
        print("无效选择")