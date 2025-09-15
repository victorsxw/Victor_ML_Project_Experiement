# 情感分类假设分析

## 项目概述

本项目旨在验证不同类型情感与股价变动之间的相关性假设。具体来说，我们将情感分为三类：

1. **财务相关情感**：包括对盈利报告、政府政策和经济指标的讨论，可能具有更强的长期影响。
2. **数据驱动情感**：基于定量分析和金融模型，可能与股价变动密切相关。
3. **观点驱动情感**：更具投机性和情绪驱动，可能对短期波动性产生影响。

## 研究假设

- **H0（零假设）**：不同类型的情感与股价变动的相关性没有显著差异。
- **H1（备择假设）**：不同类型的情感与股价变动的相关性有显著差异。

## 项目结构

```
Hypo3_sentiment_classification/
├── data/
│   ├── raw/           # 原始数据
│   └── processed/     # 处理后的数据
├── results/
│   ├── figures/       # 可视化图表
│   └── reports/       # 分析报告
├── src/
│   ├── config.py      # 配置文件
│   ├── sentiment_classifier.py  # 情感分类器
│   ├── data_analyzer.py         # 数据分析器
│   └── main.py        # 主程序
└── README.md          # 项目说明
```

## 使用方法

### 环境准备

确保已安装以下Python包：
```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
statsmodels
```

可以通过以下命令安装：
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

### 运行分析

1. 将数据文件放入 `data/raw/` 目录下
2. 运行主程序：

```bash
cd Hypo3_sentiment_classification
python src/main.py --news_path path/to/news_data.csv --reddit_path path/to/reddit_data.csv --stock_path path/to/stock_price.csv
```

### 命令行参数

- `--news_path`: 新闻数据文件路径
- `--reddit_path`: Reddit数据文件路径
- `--stock_path`: 股票价格数据文件路径
- `--start_date`: 分析开始日期，默认为 '2024-03-01'
- `--time_window`: 分析时间窗口，可选 'all', 'short_term', 'medium_term', 'long_term'，默认为 'all'
- `--output_dir`: 输出目录路径，默认使用配置文件中的路径

## 输出结果

分析完成后，将在 `results/` 目录下生成以下内容：

1. **报告**：
   - `hypothesis_test_report.md`: 假设检验报告，包含相关性分析、差异检验和结论
   - `sentiment_analysis_2024_report.md`: 2024年3月至今的情感分析报告

2. **可视化**：
   - `correlation_comparison.png`: 不同情感类别相关性比较图
   - `time_window_comparison.png`: 不同时间窗口下的情感相关性比较图
   - `sentiment_distribution_2024.png`: 情感分布图
   - `sentiment_correlations_2024.png`: 情感相关性热图
   - `sentiment_time_series_2024.png`: 情感时间序列图

## 数据要求

输入数据应包含以下列：

1. **新闻数据**：
   - `datetime`: 日期时间
   - `title`: 新闻标题
   - `content`: 新闻内容

2. **Reddit数据**：
   - `datetime`: 日期时间
   - `title`: 帖子标题
   - `content`: 帖子内容

3. **股票数据**：
   - `Date`: 交易日期
   - `Open`: 开盘价
   - `High`: 最高价
   - `Low`: 最低价
   - `Close`: 收盘价
   - `Volume`: 交易量 