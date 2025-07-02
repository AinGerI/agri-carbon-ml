#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering Analysis Module

简化的聚类分析工具，支持K-means和层次聚类。

功能：
- K-means聚类分析
- 层次聚类分析
- 聚类评估指标
- 结果可视化
- GUI界面

Author: Thesis Research Project
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ClusteringAnalyzer:
    """聚类分析器"""
    
    def __init__(self):
        self.data = None
        self.scaled_data = None
        self.scaler = StandardScaler()
        self.kmeans_results = None
        self.hierarchical_results = None
    
    def load_data(self, file_path, sheet_name=None):
        """加载数据"""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            # 设置索引
            if '地区' in df.columns:
                df.set_index('地区', inplace=True)
            elif '省份' in df.columns:
                df.set_index('省份', inplace=True)
            elif df.columns[0] in ['地区', '省份', '省市']:
                df.set_index(df.columns[0], inplace=True)
            
            # 数值化处理
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 填充缺失值
            df.fillna(df.mean(), inplace=True)
            
            self.data = df
            self.scaled_data = self.scaler.fit_transform(df.values)
            
            print(f"数据加载成功: {df.shape}")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def kmeans_clustering(self, n_clusters=4):
        """K-means聚类"""
        # 寻找最优聚类数
        k_range = range(2, min(11, len(self.data)))
        inertias = []
        silhouettes = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.scaled_data)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.scaled_data, labels))
        
        # 执行指定聚类数的聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.scaled_data)
        
        self.kmeans_results = {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertias': inertias,
            'silhouettes': silhouettes,
            'k_range': list(k_range)
        }
        
        # 添加到数据框
        self.data['KMeans_类别'] = labels
        
        return self.kmeans_results
    
    def hierarchical_clustering(self, n_clusters=4):
        """层次聚类"""
        Z = linkage(self.scaled_data, method='ward')
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        self.hierarchical_results = {
            'labels': labels,
            'linkage_matrix': Z
        }
        
        # 添加到数据框
        self.data['层次_类别'] = labels
        
        return self.hierarchical_results
    
    def plot_results(self, output_folder=None):
        """绘制聚类结果"""
        if self.kmeans_results is None or self.hierarchical_results is None:
            print("请先运行聚类分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 肘部法则
        axes[0, 0].plot(self.kmeans_results['k_range'], self.kmeans_results['inertias'], 'o-')
        axes[0, 0].set_title('肘部法则')
        axes[0, 0].set_xlabel('聚类数K')
        axes[0, 0].set_ylabel('惯性')
        
        # 2. 轮廓系数
        axes[0, 1].plot(self.kmeans_results['k_range'], self.kmeans_results['silhouettes'], 'o-g')
        axes[0, 1].set_title('轮廓系数')
        axes[0, 1].set_xlabel('聚类数K')
        axes[0, 1].set_ylabel('轮廓系数')
        
        # 3. PCA可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.scaled_data)
        
        for i, label in enumerate(self.kmeans_results['labels']):
            axes[1, 0].scatter(X_pca[i, 0], X_pca[i, 1], c=f'C{label}', s=80)
            axes[1, 0].text(X_pca[i, 0], X_pca[i, 1], self.data.index[i], 
                           fontsize=8, ha='right')
        axes[1, 0].set_title('K-means聚类结果 (PCA投影)')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        
        # 4. 树状图
        dendrogram(self.hierarchical_results['linkage_matrix'], 
                  labels=self.data.index.tolist(),
                  ax=axes[1, 1], leaf_rotation=90, leaf_font_size=8)
        axes[1, 1].set_title('层次聚类树状图')
        
        plt.tight_layout()
        
        if output_folder:
            plt.savefig(os.path.join(output_folder, 'clustering_results.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_cluster_summary(self):
        """获取聚类结果摘要"""
        if self.data is None:
            return None
        
        results = {}
        
        # K-means结果
        if 'KMeans_类别' in self.data.columns:
            kmeans_groups = {}
            for i, row in self.data.iterrows():
                label = row['KMeans_类别']
                if label not in kmeans_groups:
                    kmeans_groups[label] = []
                kmeans_groups[label].append(i)
            results['kmeans'] = kmeans_groups
        
        # 层次聚类结果
        if '层次_类别' in self.data.columns:
            hierarchical_groups = {}
            for i, row in self.data.iterrows():
                label = row['层次_类别']
                if label not in hierarchical_groups:
                    hierarchical_groups[label] = []
                hierarchical_groups[label].append(i)
            results['hierarchical'] = hierarchical_groups
        
        return results


def create_simple_clustering():
    """简单聚类分析函数"""
    # 创建分析器
    analyzer = ClusteringAnalyzer()
    
    # 文件选择
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="选择Excel文件",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    
    if not file_path:
        print("未选择文件")
        return
    
    # 加载数据
    if not analyzer.load_data(file_path):
        print("数据加载失败")
        return
    
    # 运行聚类
    analyzer.kmeans_clustering(4)
    analyzer.hierarchical_clustering(4)
    
    # 显示结果
    analyzer.plot_results()
    
    # 打印结果
    summary = analyzer.get_cluster_summary()
    if summary:
        print("\n=== K-means聚类结果 ===")
        for label, members in summary['kmeans'].items():
            print(f"聚类 {label}: {', '.join(members)}")
        
        print("\n=== 层次聚类结果 ===")
        for label, members in summary['hierarchical'].items():
            print(f"聚类 {label}: {', '.join(members)}")
    
    root.destroy()


if __name__ == "__main__":
    create_simple_clustering()