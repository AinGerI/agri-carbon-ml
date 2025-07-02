# 农业碳排放机器学习分析工具
# Agricultural Carbon Emission ML Analysis Tools (agri-carbon-ml)

这是一个专门为中国农业碳排放潜力研究设计的机器学习分析工具集，基于集成机器学习模型，包含了数据预处理、特征工程、机器学习建模、聚类分析等功能。

## 项目结构

```
agri-carbon-ml/
├── data_preprocessing/          # 数据预处理模块
│   ├── missing_value_handler.py    # 缺失值处理
│   ├── data_reorganizer.py         # 数据重组
│   └── indicator_processor.py      # 指标处理
├── feature_engineering/         # 特征工程模块
│   ├── pca_analyzer.py             # 主成分分析
│   ├── variable_selector.py        # 变量选择
│   └── data_manipulation_tools.py  # 数据操作工具
├── machine_learning/           # 机器学习模块
│   ├── lstm_model.py              # LSTM模型
│   ├── ga_svr_model.py            # 遗传算法优化SVR
│   └── ensemble_lstm_svr.py       # LSTM-SVR集成模型
├── clustering/                 # 聚类分析模块
│   └── clustering_analysis.py     # 聚类分析工具
├── utilities/                  # 工具集模块
│   └── utility_tools.py          # 通用工具函数
├── requirements.txt            # 依赖包列表
└── README.md                  # 项目说明
```

## 功能特性

### 1. 数据预处理
- 缺失值处理（三次样条插值）
- 数据重组（指标转省份视角）
- 指标重命名和标准化

### 2. 特征工程
- 主成分分析（PCA）
- 变量选择和相关性分析
- 数据结构操作工具

### 3. 机器学习
- LSTM时间序列预测
- 遗传算法优化的SVR
- LSTM-SVR集成模型

### 4. 聚类分析
- K-means聚类
- 层次聚类
- 聚类评估和可视化

### 5. 通用工具
- 数据清洗和预处理
- 文件批量操作
- 统计分析工具
- 图表生成

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

#### 1. 数据预处理
```python
from data_preprocessing.missing_value_handler import MissingValueHandler

# 处理缺失值
handler = MissingValueHandler()
handler.process_file("data.xlsx", "output.xlsx")
```

#### 2. LSTM模型训练
```python
from machine_learning.lstm_model import LSTMPredictor

# 训练LSTM模型
predictor = LSTMPredictor()
predictor.load_data("data.xlsx")
predictor.train_model()
```

#### 3. 聚类分析
```python
from clustering.clustering_analysis import create_simple_clustering

# 运行聚类分析
create_simple_clustering()
```

#### 4. 工具集使用
```python
from utilities.utility_tools import quick_data_analysis

# 快速数据分析
quick_data_analysis()
```

## GUI界面

大部分模块都提供了图形化界面，可以直接运行Python文件启动GUI：

```bash
# 启动LSTM训练界面
python machine_learning/lstm_model.py

# 启动聚类分析界面
python clustering/clustering_analysis.py

# 启动工具集界面
python utilities/utility_tools.py
```

## 主要功能模块说明

### 数据预处理模块
- **MissingValueHandler**: 使用三次样条插值处理缺失值
- **DataReorganizer**: 将指标维度数据转换为省份维度
- **IndicatorProcessor**: 处理指标重命名和标准化

### 机器学习模块
- **LSTMPredictor**: 基于TensorFlow的LSTM时间序列预测
- **GeneticAlgorithmSVR**: 使用DEAP库的遗传算法优化SVR参数
- **LSTMSVREnsemble**: 两阶段集成模型，结合LSTM和SVR优势

### 特征工程模块
- **PCAAnalyzer**: 主成分分析和方差贡献度计算
- **VariableSelector**: VIF消除和相关性分析
- **DataManipulator**: 交互式数据结构操作

## 注意事项

1. 确保Python版本 >= 3.8
2. 某些功能需要TensorFlow，如遇到安装问题可参考官方文档
3. GUI界面需要tkinter支持（通常随Python自带）
4. 大型数据集处理时注意内存使用
5. 建议在虚拟环境中运行

## 贡献指南

这是一个研究项目代码整理，主要用于学术研究。如有问题或建议，请联系项目维护者。

## 许可证

本项目仅用于学术研究，请勿用于商业用途。