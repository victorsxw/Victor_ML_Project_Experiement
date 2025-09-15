"""
情感分类器模块
实现文本的情感分类功能
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import re
from collections import Counter
from typing import Dict, List, Tuple, Union
from .config import (
    SENTIMENT_CATEGORIES,
    TEXT_PREPROCESSING,
    SENTIMENT_THRESHOLDS
)

class SentimentClassifier:
    """情感分类器类"""
    
    def __init__(self):
        """初始化分类器"""
        self.categories = SENTIMENT_CATEGORIES
        self.vectorizer = TfidfVectorizer(
            min_df=TEXT_PREPROCESSING['min_df'],
            max_df=TEXT_PREPROCESSING['max_df'],
            token_pattern=r'\b\w+\b'
        )
        self.scaler = MinMaxScaler()
        
    def preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        if not isinstance(text, str):
            return ""
            
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def calculate_category_scores(self, text: str) -> Dict[str, float]:
        """
        计算文本在各个情感类别上的得分
        
        Args:
            text: 输入文本
            
        Returns:
            各类别得分字典
        """
        text = self.preprocess_text(text)
        words = text.split()
        
        # 计算每个类别的得分
        scores = {}
        for category, info in self.categories.items():
            # 计算关键词匹配数
            keyword_counts = sum(1 for word in words if word in info['keywords'])
            # 计算得分（考虑文本长度）
            score = keyword_counts / len(words) if words else 0
            # 应用权重
            scores[category] = score * info['weight']
            
        return scores
        
    def classify_text(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        对文本进行情感分类
        
        Args:
            text: 输入文本
            
        Returns:
            (主导情感类别, 各类别得分)
        """
        # 计算各类别得分
        scores = self.calculate_category_scores(text)
        
        # 确定主导类别
        if not scores:
            return 'neutral', scores
            
        dominant_category = max(scores.items(), key=lambda x: x[1])[0]
        
        return dominant_category, scores
        
    def batch_classify(self, texts: List[str]) -> pd.DataFrame:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            包含分类结果的DataFrame
        """
        results = []
        for text in texts:
            category, scores = self.classify_text(text)
            result = {
                'text': text,
                'dominant_category': category,
                **scores
            }
            results.append(result)
            
        return pd.DataFrame(results)
        
    def analyze_text_distribution(self, texts: List[str]) -> Dict[str, int]:
        """
        分析文本集合中各情感类别的分布
        
        Args:
            texts: 文本列表
            
        Returns:
            各类别数量统计
        """
        categories = [self.classify_text(text)[0] for text in texts]
        return dict(Counter(categories))
        
    def get_category_keywords(self, text: str) -> Dict[str, List[str]]:
        """
        获取文本中各类别的关键词
        
        Args:
            text: 输入文本
            
        Returns:
            各类别找到的关键词列表
        """
        text = self.preprocess_text(text)
        words = set(text.split())
        
        keywords = {}
        for category, info in self.categories.items():
            category_keywords = [word for word in words if word in info['keywords']]
            if category_keywords:
                keywords[category] = category_keywords
                
        return keywords
        
    def get_sentiment_strength(self, scores: Dict[str, float]) -> str:
        """
        根据得分确定情感强度
        
        Args:
            scores: 情感得分字典
            
        Returns:
            情感强度标签
        """
        # 计算总得分
        total_score = sum(scores.values())
        
        # 根据阈值判断强度
        if total_score >= SENTIMENT_THRESHOLDS['strong_positive']:
            return 'strong_positive'
        elif total_score >= SENTIMENT_THRESHOLDS['positive']:
            return 'positive'
        elif total_score <= SENTIMENT_THRESHOLDS['strong_negative']:
            return 'strong_negative'
        elif total_score <= SENTIMENT_THRESHOLDS['negative']:
            return 'negative'
        else:
            return 'neutral' 