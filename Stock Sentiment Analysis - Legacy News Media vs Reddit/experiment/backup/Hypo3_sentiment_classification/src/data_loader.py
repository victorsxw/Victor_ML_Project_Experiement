"""
数据加载器模块
负责加载和预处理数据
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from .config import PATHS

class DataLoader:
    """数据加载类"""
    
    def __init__(self):
        """初始化数据加载器"""
        # 获取项目根目录的绝对路径
        self.project_root = Path(__file__).parent.parent.absolute()
        # 获取工作空间根目录
        self.workspace_root = self.project_root.parent
        
        # 数据文件路径
        self.news_data_path = self.workspace_root / "dataset" / "news_sentiment.csv"
        self.reddit_data_path = self.workspace_root / "dataset" / "reddit_sentiment_data.csv"
        self.stock_data_path = self.workspace_root / "dataset" / "6datasets-2024-2025" / "backup" / "spy_processed.csv"
        
        # 确保输出目录存在
        self._ensure_dirs()
        
    def _ensure_dirs(self):
        """确保所有必要的目录存在"""
        for path_key, path_value in PATHS.items():
            if 'dir' in path_key.lower():
                os.makedirs(self.project_root / path_value, exist_ok=True)
    
    def load_news_data(self) -> pd.DataFrame:
        """加载新闻数据"""
        print(f"正在加载新闻数据: {self.news_data_path}")
        try:
            news_data = pd.read_csv(self.news_data_path)
            print(f"新闻数据加载成功，形状: {news_data.shape}")
            
            # 确保datetime列是日期时间格式
            if 'datetime' in news_data.columns:
                news_data['datetime'] = pd.to_datetime(news_data['datetime'])
            
            return news_data
        except Exception as e:
            print(f"加载新闻数据时出错: {e}")
            return pd.DataFrame()
    
    def load_reddit_data(self) -> pd.DataFrame:
        """加载Reddit数据"""
        print(f"正在加载Reddit数据: {self.reddit_data_path}")
        try:
            reddit_data = pd.read_csv(self.reddit_data_path)
            print(f"Reddit数据加载成功，形状: {reddit_data.shape}")
            
            # 确保datetime列是日期时间格式
            if 'datetime' in reddit_data.columns:
                reddit_data['datetime'] = pd.to_datetime(reddit_data['datetime'])
            
            return reddit_data
        except Exception as e:
            print(f"加载Reddit数据时出错: {e}")
            return pd.DataFrame()
    
    def load_stock_data(self) -> pd.DataFrame:
        """加载股票数据"""
        print(f"正在加载股票数据: {self.stock_data_path}")
        try:
            stock_data = pd.read_csv(self.stock_data_path)
            print(f"股票数据加载成功，形状: {stock_data.shape}")
            
            # 确保Date列是日期时间格式
            if 'Date' in stock_data.columns:
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            return stock_data
        except Exception as e:
            print(f"加载股票数据时出错: {e}")
            return pd.DataFrame()
    
    def filter_recent_data(self, data: pd.DataFrame, date_col: str = 'datetime', 
                           start_date: str = '2024-03-01') -> pd.DataFrame:
        """筛选最近的数据（2024年3月至今）"""
        if data.empty:
            print(f"警告：输入数据为空")
            return pd.DataFrame()
            
        if date_col not in data.columns:
            print(f"警告：数据中缺少日期列 '{date_col}'")
            # 如果是股票数据，可能使用'Date'列
            if date_col == 'Date' and 'date' in data.columns:
                date_col = 'date'
                print(f"使用替代日期列: '{date_col}'")
            else:
                return pd.DataFrame()
        
        # 打印日期列的类型和前几个值，用于调试
        print(f"日期列 '{date_col}' 的类型: {data[date_col].dtype}")
        print(f"日期列的前5个值: {data[date_col].head().tolist()}")
        
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            print(f"将日期列 '{date_col}' 转换为datetime类型")
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        
        # 处理时区问题
        if hasattr(data[date_col].dtype, 'tz') and data[date_col].dtype.tz is not None:
            print(f"日期列 '{date_col}' 有时区信息，正在转换为无时区")
            data[date_col] = data[date_col].dt.tz_localize(None)
        
        start_date = pd.to_datetime(start_date)
        print(f"筛选开始日期: {start_date}")
        
        # 检查是否有任何日期大于等于开始日期
        if (data[date_col] >= start_date).any():
            filtered_data = data[data[date_col] >= start_date].copy()
            print(f"筛选后的数据（{start_date}至今）: {filtered_data.shape[0]}条记录")
            return filtered_data
        else:
            print(f"警告：没有日期大于等于 {start_date} 的数据")
            # 对于股票数据，可能需要返回所有数据
            if 'Open' in data.columns or 'open' in data.columns:
                print("检测到股票数据，返回所有数据")
                return data
            return pd.DataFrame()
    
    def merge_data(self, news_data: pd.DataFrame, reddit_data: pd.DataFrame, 
                  stock_data: pd.DataFrame) -> pd.DataFrame:
        """合并数据集"""
        if news_data.empty or reddit_data.empty or stock_data.empty:
            print("无法合并数据：一个或多个数据集为空")
            return pd.DataFrame()
        
        # 打印数据集的列名，用于调试
        print(f"新闻数据列名: {news_data.columns.tolist()}")
        print(f"Reddit数据列名: {reddit_data.columns.tolist()}")
        print(f"股票数据列名: {stock_data.columns.tolist()}")
        
        # 确保日期列名一致
        if 'Date' in stock_data.columns and 'datetime' not in stock_data.columns:
            stock_data = stock_data.rename(columns={'Date': 'datetime'})
        
        # 将日期转换为日期格式（不含时间）
        if 'datetime' in news_data.columns and 'date' not in news_data.columns:
            news_data['date'] = news_data['datetime'].dt.date
        
        if 'datetime' in reddit_data.columns and 'date' not in reddit_data.columns:
            reddit_data['date'] = reddit_data['datetime'].dt.date
        
        if 'datetime' in stock_data.columns and 'date' not in stock_data.columns:
            stock_data['date'] = stock_data['datetime'].dt.date
        elif 'date' in stock_data.columns and pd.api.types.is_object_dtype(stock_data['date']):
            # 如果date列是字符串类型，转换为日期类型
            stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date
        
        # 按日期聚合新闻和Reddit数据
        news_daily = self._aggregate_daily_data(news_data)
        reddit_daily = self._aggregate_daily_data(reddit_data)
        
        # 重命名Reddit数据的列，以避免合并时的冲突
        reddit_cols = reddit_daily.columns.tolist()
        rename_dict = {}
        for col in reddit_cols:
            if col != 'date' and col in news_daily.columns:
                rename_dict[col] = f"{col}_reddit"
        
        if rename_dict:
            reddit_daily = reddit_daily.rename(columns=rename_dict)
        
        # 确保所有数据集的date列类型一致
        print(f"股票数据date列类型: {stock_data['date'].dtype}")
        print(f"新闻数据date列类型: {news_daily['date'].dtype}")
        print(f"Reddit数据date列类型: {reddit_daily['date'].dtype}")
        
        # 将所有date列转换为字符串，以确保类型一致
        stock_data['date'] = stock_data['date'].astype(str)
        news_daily['date'] = news_daily['date'].astype(str)
        reddit_daily['date'] = reddit_daily['date'].astype(str)
        
        # 合并所有数据集
        merged = pd.merge(stock_data, news_daily, on='date', how='left')
        merged = pd.merge(merged, reddit_daily, on='date', how='left')
        
        # 检查是否有股票价格列
        if 'spy_stock' not in merged.columns and 'stock' in stock_data.columns:
            merged['spy_stock'] = merged['stock']
            print("添加spy_stock列")
        
        # 打印合并后的列名，用于调试
        print(f"合并后的数据列名: {merged.columns.tolist()}")
        print(f"合并后的数据形状: {merged.shape}")
        
        return merged
    
    def _aggregate_daily_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """将数据按日期聚合"""
        # 打印数据列名，用于调试
        print(f"聚合前的数据列名: {data.columns.tolist()}")
        
        # 确保数据包含必要的列
        required_cols = ['date']
        for col in required_cols:
            if col not in data.columns:
                print(f"警告：数据中缺少列 '{col}'")
                if col == 'date' and 'datetime' in data.columns:
                    # 如果缺少date列但有datetime列，创建date列
                    data['date'] = data['datetime'].dt.date
        
        # 确定内容列
        content_cols = []
        if 'headline' in data.columns:
            content_cols.append('headline')
        elif 'title' in data.columns:
            content_cols.append('title')
            
        # 确定情感列
        sentiment_cols = []
        if 'sentiment_chat2' in data.columns:
            sentiment_cols.append('sentiment_chat2')
        if 'answer_chat2' in data.columns:
            sentiment_cols.append('answer_chat2')
            
        # 按日期分组并聚合
        agg_dict = {}
        
        # 为内容列添加聚合函数
        for col in content_cols:
            agg_dict[col] = lambda x: ' '.join(str(i) for i in x if pd.notna(i) and i)
        
        # 为情感列添加聚合函数
        for col in sentiment_cols:
            # 检查列的类型
            if col in data.columns:
                # 打印前几个值，用于调试
                print(f"{col}列的前5个值: {data[col].head().tolist()}")
                
                # 检查是否为字符串类型
                if data[col].dtype == 'object':
                    print(f"{col}列是字符串类型，将其转换为连接字符串")
                    # 修改聚合方式为连接字符串
                    agg_dict[col] = lambda x: ' '.join(str(i) for i in x if pd.notna(i) and i)
                else:
                    # 如果是数值类型，使用平均值
                    agg_dict[col] = 'mean'
        
        # 如果没有可聚合的列，返回原始数据
        if not agg_dict:
            print("警告：没有可聚合的列")
            return data
        
        daily_data = data.groupby('date').agg(agg_dict).reset_index()
        print(f"聚合后的数据列名: {daily_data.columns.tolist()}")
        
        return daily_data 