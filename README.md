# 农业碳排放机器学习分析工具 | Agricultural Carbon Emission ML Analysis Tools

[中文](#中文) | [English](#english)

---

## 中文

这是论文《基于集成机器学习模型与蒙特卡洛的中国农业碳排放潜力研究》的代码实现。主要用Python写了一套农业数据分析工具，包括数据处理、机器学习建模和结果可视化等功能。

由于论文需要处理大量的农业数据，涉及多种机器学习方法，所以把常用的功能整理成了这个工具包，方便复现和后续研究使用。

### 主要功能

这个工具包主要有以下几个模块：

#### 📊 数据预处理
- 处理农业数据中的缺失值（用三次样条插值）
- 把数据从指标维度转换成省份维度（论文需要）
- 指标标准化和重命名

#### 🔧 特征工程
- PCA降维分析
- 变量筛选（去多重共线性）
- 一些数据操作的小工具

#### 🤖 机器学习
- LSTM时间序列预测模型
- 用遗传算法优化的SVR
- LSTM+SVR的集成模型（论文的主要方法）

#### 📈 聚类分析
- K-means聚类
- 层次聚类
- 聚类结果可视化

#### 🛠️ 通用工具
- 数据清洗
- 批量文件处理
- 快速统计分析
- 图表生成

### 项目结构

```
agri-carbon-ml/
├── data_preprocessing/          # 数据预处理
├── feature_engineering/         # 特征工程
├── machine_learning/           # 机器学习模型
├── clustering/                 # 聚类分析
├── utilities/                  # 工具函数
├── requirements.txt            # 依赖包
└── README.md                  # 说明文档
```

### 快速上手

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要用到pandas、numpy、scikit-learn、tensorflow这些包。

#### 2. 运行示例

大部分模块都有GUI界面，直接运行就行：

```bash
# LSTM模型训练
python machine_learning/lstm_model.py

# 聚类分析
python clustering/clustering_analysis.py

# 数据处理工具
python utilities/utility_tools.py
```

也可以在代码中调用：

```python
# 处理缺失值
from data_preprocessing.missing_value_handler import MissingValueHandler
handler = MissingValueHandler()
handler.process_file("data.xlsx", "output.xlsx")

# 训练LSTM
from machine_learning.lstm_model import LSTMPredictor
predictor = LSTMPredictor()
predictor.load_data("data.xlsx")
predictor.train_model()
```

### 数据格式

工具主要处理Excel格式的农业数据，一般第一列是地区名称，后面是各年份的指标数据。

### 注意事项

- Python版本建议3.8以上
- 如果要用LSTM功能需要装TensorFlow，第一次安装可能比较慢
- 处理大数据集时注意内存，可能需要分批处理
- GUI界面需要tkinter支持（一般系统都有）

### 关于这个项目

这是我整理论文代码时写的，主要目的是让代码更规范、更容易复现。如果你也在做相关研究，希望这些工具能帮到你。

代码写得不一定完美，如果有bug或者改进建议欢迎提issue。

### 许可

这个项目主要用于学术研究，如果要用于其他目的请先联系我。

---

## English

This is the code implementation for the thesis "Research on China's Agricultural Carbon Emission Potential Based on Ensemble Machine Learning Models and Monte Carlo Methods". It's a Python toolkit for agricultural data analysis, including data processing, machine learning modeling, and result visualization.

Since the thesis involves processing large amounts of agricultural data with various machine learning methods, I organized commonly used functions into this toolkit to facilitate reproduction and future research.

### Key Features

This toolkit mainly includes the following modules:

#### 📊 Data Preprocessing
- Handle missing values in agricultural data (using cubic spline interpolation)
- Convert data from indicator dimension to province dimension (required for thesis)
- Indicator standardization and renaming

#### 🔧 Feature Engineering
- PCA dimensionality reduction analysis
- Variable selection (multicollinearity removal)
- Various data manipulation utilities

#### 🤖 Machine Learning
- LSTM time series prediction model
- Genetic algorithm optimized SVR
- LSTM+SVR ensemble model (main method in thesis)

#### 📈 Clustering Analysis
- K-means clustering
- Hierarchical clustering
- Clustering result visualization

#### 🛠️ Utilities
- Data cleaning
- Batch file processing
- Quick statistical analysis
- Chart generation

### Project Structure

```
agri-carbon-ml/
├── data_preprocessing/          # Data preprocessing
├── feature_engineering/         # Feature engineering
├── machine_learning/           # ML models
├── clustering/                 # Clustering analysis
├── utilities/                  # Utility functions
├── requirements.txt            # Dependencies
└── README.md                  # Documentation
```

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Main packages include pandas, numpy, scikit-learn, tensorflow.

#### 2. Running Examples

Most modules have GUI interfaces, just run directly:

```bash
# LSTM model training
python machine_learning/lstm_model.py

# Clustering analysis
python clustering/clustering_analysis.py

# Data processing tools
python utilities/utility_tools.py
```

You can also call them in code:

```python
# Handle missing values
from data_preprocessing.missing_value_handler import MissingValueHandler
handler = MissingValueHandler()
handler.process_file("data.xlsx", "output.xlsx")

# Train LSTM
from machine_learning.lstm_model import LSTMPredictor
predictor = LSTMPredictor()
predictor.load_data("data.xlsx")
predictor.train_model()
```

### Data Format

The toolkit mainly processes Excel format agricultural data, typically with region names in the first column followed by yearly indicator data.

### Notes

- Python 3.8+ recommended
- TensorFlow required for LSTM functionality, first installation might be slow
- Watch memory usage when processing large datasets, may need batch processing
- GUI requires tkinter support (usually built-in)

### About This Project

This was written while organizing thesis code, mainly to make the code more standardized and easier to reproduce. If you're doing related research, hope these tools can help.

The code isn't perfect - welcome to submit issues for bugs or improvement suggestions.

### License

This project is mainly for academic research. Please contact me for other uses.