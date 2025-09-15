"""
情感分类分析的配置文件
包含关键词定义、参数设置等
"""

# 情感类型定义及其关键词
SENTIMENT_CATEGORIES = {
    'fiscal': {
        'keywords': [
            'earnings', 'revenue', 'profit', 'loss', 'eps', 'margin',
            'guidance', 'forecast', 'outlook', 'dividend',
            'gdp', 'inflation', 'rate', 'fed', 'policy',
            'regulation', 'tax', 'budget', 'debt', 'deficit',
            'economic', 'economy', 'fiscal', 'monetary'
        ],
        'description': '财务相关情绪',
        'weight': 1.0
    },
    'data_driven': {
        'keywords': [
            'analysis', 'model', 'technical', 'trend', 'indicator',
            'chart', 'pattern', 'volume', 'momentum', 'resistance',
            'support', 'average', 'ratio', 'metric', 'data',
            'statistics', 'correlation', 'probability', 'algorithm',
            'backtest', 'historical', 'quantitative', 'systematic'
        ],
        'description': '数据驱动情绪',
        'weight': 1.0
    },
    'opinion': {
        'keywords': [
            'believe', 'think', 'feel', 'expect', 'hope', 'worry',
            'bullish', 'bearish', 'optimistic', 'pessimistic',
            'confident', 'concerned', 'uncertain', 'sure', 'doubt',
            'speculation', 'potential', 'might', 'could', 'should',
            'opinion', 'view', 'sentiment', 'gut', 'intuition'
        ],
        'description': '观点驱动情绪',
        'weight': 1.0
    }
}

# 时间窗口设置
TIME_WINDOWS = {
    'short_term': {'days': 3, 'description': '短期'},
    'medium_term': {'days': 14, 'description': '中期'},
    'long_term': {'days': 30, 'description': '长期'}
}

# 相关性分析参数
CORRELATION_PARAMS = {
    'lag_days': [0, 1, 3, 7],  # 滞后天数
    'min_samples': 30,  # 最小样本数
    'significance_level': 0.05  # 显著性水平
}

# 文本预处理参数
TEXT_PREPROCESSING = {
    'min_word_length': 2,
    'max_word_length': 50,
    'min_df': 5,  # 最小文档频率
    'max_df': 0.95  # 最大文档频率（百分比）
}

# 情感强度阈值
SENTIMENT_THRESHOLDS = {
    'strong_positive': 0.6,
    'positive': 0.2,
    'neutral': (-0.2, 0.2),
    'negative': -0.2,
    'strong_negative': -0.6
}

# 输出路径配置
PATHS = {
    'data': {
        'raw': 'data/raw',
        'processed': 'data/processed'
    },
    'results': {
        'figures': 'results/figures',
        'reports': 'results/reports'
    }
}

# 可视化参数
VISUALIZATION = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'default',
    'color_palette': 'tab10'
} 