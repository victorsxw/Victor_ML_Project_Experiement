"""
数据分析器模块
实现数据加载、处理、分析和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .sentiment_classifier import SentimentClassifier
from .config import (
    TIME_WINDOWS,
    CORRELATION_PARAMS,
    PATHS,
    VISUALIZATION
)

class DataAnalyzer:
    """数据分析类"""
    
    def __init__(self):
        """初始化数据分析器"""
        self.sentiment_classifier = SentimentClassifier()
        self.news_data = None
        self.reddit_data = None
        self.stock_data = None
        self.merged_data = None
        self.recent_data = None  # 2024年3月至今的数据
        
        # 设置可视化参数
        try:
            plt.style.use(VISUALIZATION['style'])
        except:
            # 如果指定样式不可用，使用默认样式
            plt.style.use('default')
            
        plt.rcParams['figure.figsize'] = VISUALIZATION['figure_size']
        plt.rcParams['figure.dpi'] = VISUALIZATION['dpi']
        
        try:
            sns.set_palette(VISUALIZATION['color_palette'])
        except:
            # 如果指定调色板不可用，使用默认调色板
            sns.set_palette('tab10')
        
        # 创建输出目录
        self._create_output_dirs()
        
    def _create_output_dirs(self):
        """创建输出目录"""
        for path_type, path_dict in PATHS.items():
            for _, path in path_dict.items():
                os.makedirs(path, exist_ok=True)
                
    def load_data(self, news_path: str, reddit_path: str, stock_path: str) -> None:
        """
        加载数据文件
        
        Args:
            news_path: 新闻数据文件路径
            reddit_path: Reddit数据文件路径
            stock_path: 股票价格数据文件路径
        """
        # 加载新闻数据
        self.news_data = pd.read_csv(news_path)
        if 'datetime' in self.news_data.columns:
            self.news_data['datetime'] = pd.to_datetime(self.news_data['datetime'], errors='coerce')
        
        # 加载Reddit数据
        self.reddit_data = pd.read_csv(reddit_path)
        if 'datetime' in self.reddit_data.columns:
            self.reddit_data['datetime'] = pd.to_datetime(self.reddit_data['datetime'], errors='coerce')
        
        # 加载股票数据
        self.stock_data = pd.read_csv(stock_path)
        if 'Date' in self.stock_data.columns:
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], errors='coerce')
            
        print(f"数据加载完成:")
        print(f"新闻数据: {self.news_data.shape if self.news_data is not None else '未加载'}")
        print(f"Reddit数据: {self.reddit_data.shape if self.reddit_data is not None else '未加载'}")
        print(f"股票数据: {self.stock_data.shape if self.stock_data is not None else '未加载'}")
        
    def filter_recent_data(self, start_date: str = '2024-03-01') -> pd.DataFrame:
        """
        筛选2024年3月至今的数据
        
        Args:
            start_date: 开始日期，默认2024-03-01
            
        Returns:
            筛选后的合并数据
        """
        # 确保数据已加载
        if any(data is None for data in [self.news_data, self.reddit_data, self.stock_data]):
            raise ValueError("请先加载数据")
            
        start_timestamp = pd.Timestamp(start_date)
        end_timestamp = pd.Timestamp(datetime.now().date())
        
        # 筛选新闻数据
        filtered_news = self.news_data[
            (self.news_data['datetime'] >= start_timestamp) & 
            (self.news_data['datetime'] <= end_timestamp)
        ].copy()
        
        # 筛选Reddit数据
        filtered_reddit = self.reddit_data[
            (self.reddit_data['datetime'] >= start_timestamp) & 
            (self.reddit_data['datetime'] <= end_timestamp)
        ].copy()
        
        # 筛选股票数据
        filtered_stock = self.stock_data[
            (self.stock_data['Date'] >= start_timestamp) & 
            (self.stock_data['Date'] <= end_timestamp)
        ].copy()
        
        print(f"筛选 {start_date} 至今的数据:")
        print(f"新闻数据: {filtered_news.shape}")
        print(f"Reddit数据: {filtered_reddit.shape}")
        print(f"股票数据: {filtered_stock.shape}")
        
        # 将筛选后的数据合并
        self.recent_data = self._merge_data(filtered_news, filtered_reddit, filtered_stock)
        
        return self.recent_data
        
    def _merge_data(
        self, 
        news_data: pd.DataFrame, 
        reddit_data: pd.DataFrame, 
        stock_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        合并不同来源的数据
        
        Args:
            news_data: 新闻数据
            reddit_data: Reddit数据
            stock_data: 股票价格数据
            
        Returns:
            合并后的数据
        """
        # 确保日期列名统一
        news_df = news_data.copy()
        reddit_df = reddit_data.copy()
        stock_df = stock_data.copy()
        
        # 转换日期列为日期格式(无时间)
        if 'datetime' in news_df.columns:
            news_df['date'] = news_df['datetime'].dt.date
            news_df = news_df.groupby('date').agg({
                'title': lambda x: ' '.join(x),
                'content': lambda x: ' '.join(x),
                # 如果有其他需要聚合的列，可以在这里添加
            }).reset_index()
            
        if 'datetime' in reddit_df.columns:
            reddit_df['date'] = reddit_df['datetime'].dt.date
            reddit_df = reddit_df.groupby('date').agg({
                'title': lambda x: ' '.join(x),
                'content': lambda x: ' '.join(x),
                # 如果有其他需要聚合的列，可以在这里添加
            }).reset_index()
            
        if 'Date' in stock_df.columns:
            stock_df['date'] = stock_df['Date'].dt.date
            stock_df = stock_df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
        
        # 合并数据
        merged_df = pd.merge(stock_df, news_df, on='date', how='left', suffixes=('', '_news'))
        merged_df = pd.merge(merged_df, reddit_df, on='date', how='left', suffixes=('', '_reddit'))
        
        # 填充缺失值
        text_columns = [col for col in merged_df.columns if col in ['title', 'content', 'title_reddit', 'content_reddit']]
        for col in text_columns:
            merged_df[col] = merged_df[col].fillna('')
            
        # 计算价格变动
        merged_df['price_change'] = merged_df['close'].pct_change()
        merged_df['price_change_next_day'] = merged_df['price_change'].shift(-1)
        
        # 添加日期特征
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
        merged_df['month'] = merged_df['date'].dt.month
        merged_df['year'] = merged_df['date'].dt.year
        
        return merged_df
        
    def analyze_recent_sentiments(self) -> pd.DataFrame:
        """
        分析2024年3月至今数据的情感分类
        
        Returns:
            带有情感分类的DataFrame
        """
        if self.recent_data is None:
            raise ValueError("请先筛选2024年3月至今的数据")
        
        # 打印列名，用于调试
        print("合并后的数据列名:", self.recent_data.columns.tolist())
        
        # 检查必要的列是否存在
        news_content_col = 'headline' if 'headline' in self.recent_data.columns else 'title'
        reddit_content_col = 'title' if 'title_reddit' in self.recent_data.columns else 'title_reddit'
        
        # 为新闻和Reddit内容创建组合文本
        if news_content_col in self.recent_data.columns:
            self.recent_data['news_text'] = self.recent_data[news_content_col].fillna('')
        else:
            print(f"警告：数据中缺少新闻内容列")
            self.recent_data['news_text'] = ''
            
        if reddit_content_col in self.recent_data.columns:
            self.recent_data['reddit_text'] = self.recent_data[reddit_content_col].fillna('')
        else:
            print(f"警告：数据中缺少Reddit内容列")
            self.recent_data['reddit_text'] = ''
        
        # 分析新闻情感
        results_news = []
        for idx, row in self.recent_data.iterrows():
            if not pd.isna(row['news_text']) and row['news_text']:
                category, scores = self.sentiment_classifier.classify_text(row['news_text'])
                strength = self.sentiment_classifier.get_sentiment_strength(scores)
                results_news.append({
                    'date': row['date'],
                    'dominant_category': category,
                    'sentiment_strength': strength,
                    **scores
                })
        news_sentiments = pd.DataFrame(results_news)
        
        # 分析Reddit情感
        results_reddit = []
        for idx, row in self.recent_data.iterrows():
            if not pd.isna(row['reddit_text']) and row['reddit_text']:
                category, scores = self.sentiment_classifier.classify_text(row['reddit_text'])
                strength = self.sentiment_classifier.get_sentiment_strength(scores)
                results_reddit.append({
                    'date': row['date'],
                    'dominant_category': category,
                    'sentiment_strength': strength,
                    **scores
                })
        reddit_sentiments = pd.DataFrame(results_reddit)
        
        # 重命名列以区分来源
        if not news_sentiments.empty:
            news_sentiments = news_sentiments.add_suffix('_news')
            news_sentiments = news_sentiments.rename(columns={'date_news': 'date'})
            
        if not reddit_sentiments.empty:
            reddit_sentiments = reddit_sentiments.add_suffix('_reddit')
            reddit_sentiments = reddit_sentiments.rename(columns={'date_reddit': 'date'})
            
        # 合并情感分析结果到原始数据
        if not news_sentiments.empty:
            self.recent_data = pd.merge(self.recent_data, news_sentiments, on='date', how='left')
            
        if not reddit_sentiments.empty:
            self.recent_data = pd.merge(self.recent_data, reddit_sentiments, on='date', how='left')
            
        return self.recent_data
        
    def calculate_correlations(self, time_window: str = 'all') -> dict:
        """
        计算情感类别与股价变动的相关性
        
        Args:
            time_window: 时间窗口，可选 'all', 'short_term', 'medium_term', 'long_term'
            
        Returns:
            相关性结果字典
        """
        if self.recent_data is None:
            raise ValueError("请先分析2024年3月至今的数据情感")
            
        # 准备数据
        df = self.recent_data.copy()
        
        # 检查是否有股价数据列
        stock_col = None
        for col_name in ['spy_stock', 'stock', 'nvda_stock', 'close']:
            if col_name in df.columns:
                stock_col = col_name
                print(f"使用{stock_col}列作为股价数据")
                break
                
        if stock_col is None:
            print("警告：数据中缺少股价列，无法计算相关性")
            print("可用的列名:", df.columns.tolist())
            return {'news': {}, 'reddit': {}}
        
        # 计算股价变动
        df['price_change'] = df[stock_col].pct_change()
        df['price_change_next_day'] = df['price_change'].shift(-1)
        
        # 打印股价变动的统计信息
        print(f"股价变动统计信息:")
        print(f"均值: {df['price_change_next_day'].mean()}")
        print(f"标准差: {df['price_change_next_day'].std()}")
        print(f"最小值: {df['price_change_next_day'].min()}")
        print(f"最大值: {df['price_change_next_day'].max()}")
        
        # 删除缺失值
        df = df.dropna(subset=['price_change_next_day'])
        
        # 应用时间窗口筛选
        if time_window != 'all' and time_window in TIME_WINDOWS:
            window_days = TIME_WINDOWS[time_window]['days']
            latest_date = pd.to_datetime(df['date']).max()
            cutoff_date = latest_date - pd.Timedelta(days=window_days)
            df = df[pd.to_datetime(df['date']) >= cutoff_date]
            
        # 计算相关性
        correlation_results = {}
        
        # 检查情感列是否存在
        fiscal_news_col = 'fiscal_news' if 'fiscal_news' in df.columns else None
        data_driven_news_col = 'data_driven_news' if 'data_driven_news' in df.columns else None
        opinion_news_col = 'opinion_news' if 'opinion_news' in df.columns else None
        
        fiscal_reddit_col = 'fiscal_reddit' if 'fiscal_reddit' in df.columns else None
        data_driven_reddit_col = 'data_driven_reddit' if 'data_driven_reddit' in df.columns else None
        opinion_reddit_col = 'opinion_reddit' if 'opinion_reddit' in df.columns else None
        
        # 如果情感列不存在，使用sentiment_chat2列
        if not any([fiscal_news_col, data_driven_news_col, opinion_news_col]):
            print("使用sentiment_chat2列作为情感数据")
            # 创建情感类别列
            if 'sentiment_chat2' in df.columns:
                # 计算每种情感的频率
                sentiment_counts = df['sentiment_chat2'].str.count('positive')
                df['fiscal_news'] = sentiment_counts
                df['data_driven_news'] = sentiment_counts
                df['opinion_news'] = sentiment_counts
                
                fiscal_news_col = 'fiscal_news'
                data_driven_news_col = 'data_driven_news'
                opinion_news_col = 'opinion_news'
        
        if not any([fiscal_reddit_col, data_driven_reddit_col, opinion_reddit_col]):
            print("使用sentiment_chat2_reddit列作为Reddit情感数据")
            # 创建情感类别列
            if 'sentiment_chat2_reddit' in df.columns:
                # 计算每种情感的频率
                sentiment_counts = df['sentiment_chat2_reddit'].str.count('positive')
                df['fiscal_reddit'] = sentiment_counts
                df['data_driven_reddit'] = sentiment_counts
                df['opinion_reddit'] = sentiment_counts
                
                fiscal_reddit_col = 'fiscal_reddit'
                data_driven_reddit_col = 'data_driven_reddit'
                opinion_reddit_col = 'opinion_reddit'
        
        # 新闻情感相关性
        news_fiscal = df[fiscal_news_col].corr(df['price_change_next_day']) if fiscal_news_col else np.nan
        news_data_driven = df[data_driven_news_col].corr(df['price_change_next_day']) if data_driven_news_col else np.nan
        news_opinion = df[opinion_news_col].corr(df['price_change_next_day']) if opinion_news_col else np.nan
        
        # Reddit情感相关性
        reddit_fiscal = df[fiscal_reddit_col].corr(df['price_change_next_day']) if fiscal_reddit_col else np.nan
        reddit_data_driven = df[data_driven_reddit_col].corr(df['price_change_next_day']) if data_driven_reddit_col else np.nan
        reddit_opinion = df[opinion_reddit_col].corr(df['price_change_next_day']) if opinion_reddit_col else np.nan
        
        # 存储结果
        correlation_results['news'] = {
            'fiscal': news_fiscal,
            'data_driven': news_data_driven,
            'opinion': news_opinion
        }
        
        correlation_results['reddit'] = {
            'fiscal': reddit_fiscal,
            'data_driven': reddit_data_driven,
            'opinion': reddit_opinion
        }
        
        # 统计显著性检验
        for source in ['news', 'reddit']:
            for category in ['fiscal', 'data_driven', 'opinion']:
                col_name = f"{category}_{source}"
                if col_name in df.columns:
                    try:
                        correlation, p_value = stats.pearsonr(
                            df[col_name].fillna(0), 
                            df['price_change_next_day']
                        )
                        correlation_results[source][f"{category}_p_value"] = p_value
                    except:
                        print(f"无法计算{col_name}的相关性")
                        correlation_results[source][f"{category}_p_value"] = np.nan
                    
        return correlation_results
        
    def test_hypothesis(self) -> dict:
        """
        验证情感分类对股价预测的假设
        
        Returns:
            假设检验结果字典
        """
        if self.recent_data is None:
            raise ValueError("请先分析2024年3月至今的数据情感")
            
        # 准备数据
        df = self.recent_data.copy()
        df = df.dropna(subset=['price_change_next_day'])
        
        # 定义不同情感类别的变量
        sources = ['news', 'reddit']
        categories = ['fiscal', 'data_driven', 'opinion']
        
        # 存储结果
        hypothesis_results = {}
        
        # 对每个数据源进行验证
        for source in sources:
            # 准备回归分析数据
            X_cols = [f"{category}_{source}" for category in categories]
            X_cols = [col for col in X_cols if col in df.columns]
            
            if not X_cols:
                continue
                
            # 去除缺失值
            regression_df = df.dropna(subset=X_cols + ['price_change_next_day'])
            
            # 简单线性回归
            for category in categories:
                col_name = f"{category}_{source}"
                if col_name in df.columns:
                    X = regression_df[col_name].values.reshape(-1, 1)
                    y = regression_df['price_change_next_day'].values
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # R²值
                    r_squared = model.score(X, y)
                    
                    # 系数显著性检验
                    X_with_const = sm.add_constant(X)
                    sm_model = sm.OLS(y, X_with_const).fit()
                    
                    # 存储结果
                    if source not in hypothesis_results:
                        hypothesis_results[source] = {}
                        
                    hypothesis_results[source][category] = {
                        'coefficient': model.coef_[0],
                        'r_squared': r_squared,
                        'p_value': sm_model.pvalues[1],
                        'summary': sm_model.summary().as_text()
                    }
            
        return hypothesis_results
        
    def visualize_sentiment_distribution(self) -> None:
        """
        可视化情感分布
        """
        if self.recent_data is None:
            raise ValueError("请先分析2024年3月至今的数据情感")
            
        # 情感类型分布
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 新闻情感分布
        if 'dominant_category_news' in self.recent_data.columns:
            news_counts = self.recent_data['dominant_category_news'].value_counts()
            news_counts.plot(kind='bar', ax=axes[0], color='skyblue')
            axes[0].set_title('新闻情感分类分布')
            axes[0].set_ylabel('数量')
            axes[0].set_xlabel('情感类别')
            
        # Reddit情感分布
        if 'dominant_category_reddit' in self.recent_data.columns:
            reddit_counts = self.recent_data['dominant_category_reddit'].value_counts()
            reddit_counts.plot(kind='bar', ax=axes[1], color='salmon')
            axes[1].set_title('Reddit情感分类分布')
            axes[1].set_ylabel('数量')
            axes[1].set_xlabel('情感类别')
            
        plt.tight_layout()
        output_path = os.path.join(PATHS['results']['figures'], 'sentiment_distribution_2024.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"情感分布可视化已保存至: {output_path}")
        
    def visualize_correlations(self, correlations: dict) -> None:
        """
        可视化相关性分析结果
        
        Args:
            correlations: 相关性分析结果字典
        """
        # 准备数据
        sources = list(correlations.keys())
        categories = ['fiscal', 'data_driven', 'opinion']
        
        # 创建DataFrame
        corr_data = []
        for source in sources:
            for category in categories:
                if category in correlations[source]:
                    corr_data.append({
                        'source': source,
                        'category': category,
                        'correlation': correlations[source][category],
                        'p_value': correlations[source].get(f"{category}_p_value", np.nan)
                    })
                    
        corr_df = pd.DataFrame(corr_data)
        
        # 绘制相关性热图
        if not corr_df.empty:
            plt.figure(figsize=(10, 8))
            
            # 创建相关性矩阵
            corr_matrix = corr_df.pivot(index='category', columns='source', values='correlation')
            
            # 标记显著的相关性
            annot_matrix = corr_matrix.copy()
            for idx, row in corr_df.iterrows():
                if pd.notna(row['p_value']) and row['p_value'] < 0.05:
                    annot_matrix.loc[row['category'], row['source']] = f"{row['correlation']:.2f}*"
                else:
                    annot_matrix.loc[row['category'], row['source']] = f"{row['correlation']:.2f}"
                    
            # 绘制热图
            sns.heatmap(corr_matrix, annot=annot_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='')
            
            plt.title('情感类别与股价变动相关性 (2024年3月至今)')
            plt.tight_layout()
            
            output_path = os.path.join(PATHS['results']['figures'], 'sentiment_correlations_2024.png')
            plt.savefig(output_path)
            plt.close()
            
            print(f"相关性可视化已保存至: {output_path}")
            
    def visualize_time_series(self) -> None:
        """
        可视化时间序列分析结果
        """
        if self.recent_data is None:
            raise ValueError("请先分析2024年3月至今的数据情感")
            
        # 准备数据
        df = self.recent_data.copy().sort_values('date')
        
        # 设置日期索引
        df.set_index('date', inplace=True)
        
        # 情感得分与股价变动的时间序列图
        plt.figure(figsize=(16, 12))
        
        # 绘制股价变动
        ax1 = plt.subplot(311)
        ax1.plot(df.index, df['price_change_next_day'], 'g-', label='次日股价变动')
        ax1.set_ylabel('股价变动百分比')
        ax1.legend(loc='best')
        ax1.set_title('股价变动与情感得分时间序列 (2024年3月至今)')
        
        # 绘制新闻情感得分
        ax2 = plt.subplot(312, sharex=ax1)
        for category in ['fiscal_news', 'data_driven_news', 'opinion_news']:
            if category in df.columns:
                ax2.plot(df.index, df[category], label=category)
        ax2.set_ylabel('情感得分')
        ax2.legend(loc='best')
        ax2.set_title('新闻情感得分')
        
        # 绘制Reddit情感得分
        ax3 = plt.subplot(313, sharex=ax1)
        for category in ['fiscal_reddit', 'data_driven_reddit', 'opinion_reddit']:
            if category in df.columns:
                ax3.plot(df.index, df[category], label=category)
        ax3.set_ylabel('情感得分')
        ax3.legend(loc='best')
        ax3.set_title('Reddit情感得分')
        
        plt.tight_layout()
        output_path = os.path.join(PATHS['results']['figures'], 'sentiment_time_series_2024.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"时间序列可视化已保存至: {output_path}")
        
    def generate_summary_report(self, correlations: dict, hypothesis_results: dict) -> None:
        """
        生成分析总结报告
        
        Args:
            correlations: 相关性分析结果
            hypothesis_results: 假设检验结果
        """
        report_path = os.path.join(PATHS['results']['reports'], 'sentiment_analysis_2024_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 情感分类分析报告 (2024年3月至今)\n\n")
            
            # 写入分析概述
            f.write("## 分析概述\n\n")
            if self.recent_data is not None:
                f.write(f"- 分析时间段: 2024-03-01 至 {self.recent_data['date'].max().strftime('%Y-%m-%d')}\n")
                f.write(f"- 样本天数: {self.recent_data['date'].nunique()}\n")
                f.write(f"- 股票价格数据条数: {self.recent_data.shape[0]}\n\n")
                
            # 写入情感分布分析
            f.write("## 情感分布\n\n")
            for source in ['news', 'reddit']:
                f.write(f"### {source.capitalize()}情感分布\n\n")
                col_name = f"dominant_category_{source}"
                if col_name in self.recent_data.columns:
                    counts = self.recent_data[col_name].value_counts()
                    f.write("| 情感类别 | 数量 | 占比 |\n")
                    f.write("|---------|------|------|\n")
                    for category, count in counts.items():
                        percentage = count / counts.sum() * 100
                        f.write(f"| {category} | {count} | {percentage:.2f}% |\n")
                else:
                    f.write(f"无{source}情感分类数据\n")
                f.write("\n")
                
            # 写入相关性分析
            f.write("## 相关性分析\n\n")
            for source, source_correlations in correlations.items():
                f.write(f"### {source.capitalize()}情感与股价相关性\n\n")
                f.write("| 情感类别 | 相关系数 | P值 | 显著性 |\n")
                f.write("|---------|----------|-----|--------|\n")
                
                for category in ['fiscal', 'data_driven', 'opinion']:
                    if category in source_correlations:
                        corr = source_correlations[category]
                        p_value = source_correlations.get(f"{category}_p_value", np.nan)
                        is_significant = "是" if p_value is not None and p_value < 0.05 else "否"
                        f.write(f"| {category} | {corr:.4f} | {p_value:.4f} | {is_significant} |\n")
                f.write("\n")
                
            # 写入假设检验结果
            f.write("## 假设检验结果\n\n")
            
            for source, source_results in hypothesis_results.items():
                f.write(f"### {source.capitalize()}情感的线性回归结果\n\n")
                
                for category, results in source_results.items():
                    f.write(f"#### {category}情感\n\n")
                    f.write(f"- 回归系数: {results['coefficient']:.4f}\n")
                    f.write(f"- R²值: {results['r_squared']:.4f}\n")
                    f.write(f"- P值: {results['p_value']:.4f}\n")
                    f.write(f"- 显著性: {'显著' if results['p_value'] < 0.05 else '不显著'}\n\n")
                    
            # 写入结论
            f.write("## 结论\n\n")
            
            # 找出显著的相关性
            significant_correlations = []
            for source, source_correlations in correlations.items():
                for category in ['fiscal', 'data_driven', 'opinion']:
                    if category in source_correlations:
                        p_value = source_correlations.get(f"{category}_p_value", 1.0)
                        if p_value < 0.05:
                            significant_correlations.append((source, category, source_correlations[category], p_value))
            
            if significant_correlations:
                f.write("### 显著的情感与股价相关性\n\n")
                for source, category, corr, p_value in significant_correlations:
                    f.write(f"- {source.capitalize()}的{category}情感与股价变动显著相关 (r = {corr:.4f}, p = {p_value:.4f})。\n")
            else:
                f.write("- 在分析的时间范围内，未发现情感类别与股价变动之间存在显著相关性。\n")
            
            f.write("\n### 假设验证\n\n")
            
            # 检查假设是否成立
            null_hypothesis_rejected = False
            for source, source_results in hypothesis_results.items():
                for category, results in source_results.items():
                    if results['p_value'] < 0.05:
                        null_hypothesis_rejected = True
                        break
                if null_hypothesis_rejected:
                    break
                    
            if null_hypothesis_rejected:
                f.write("- **拒绝零假设 (H0)**：至少有一种情感类别与股价变动之间存在显著相关性。\n")
                f.write("- **接受备择假设 (H1)**：不同类型的情感对股价变动的预测能力不同。\n")
            else:
                f.write("- **接受零假设 (H0)**：不同类型的情感对股价变动的预测能力无显著差异。\n")
                f.write("- **拒绝备择假设 (H1)**：没有足够证据表明不同类型的情感对股价变动有不同的预测能力。\n")
                
            f.write("\n### 建议\n\n")
            
            if null_hypothesis_rejected:
                f.write("1. 投资决策中应考虑特定类型的情感信号。\n")
                for source, category, corr, p_value in significant_correlations:
                    if corr > 0:
                        f.write(f"2. {source.capitalize()}中的{category}情感增强可能预示着积极的股价变动。\n")
                    else:
                        f.write(f"2. {source.capitalize()}中的{category}情感增强可能预示着消极的股价变动。\n")
            else:
                f.write("1. 基于当前数据，不建议仅依靠情感类别作为预测股价变动的指标。\n")
                f.write("2. 考虑扩大数据样本或结合其他指标进行综合分析。\n")
                
        print(f"分析报告已生成: {report_path}")
        
    def analyze_2024_data(self, news_path: str, reddit_path: str, stock_path: str) -> None:
        """
        完整分析2024年3月至今的数据
        
        Args:
            news_path: 新闻数据文件路径
            reddit_path: Reddit数据文件路径
            stock_path: 股票价格数据文件路径
        """
        # 加载数据
        self.load_data(news_path, reddit_path, stock_path)
        
        # 筛选2024年3月至今的数据
        self.filter_recent_data()
        
        # 情感分析
        self.analyze_recent_sentiments()
        
        # 相关性分析
        correlations = self.calculate_correlations()
        
        # 假设检验
        hypothesis_results = self.test_hypothesis()
        
        # 可视化
        self.visualize_sentiment_distribution()
        self.visualize_correlations(correlations)
        self.visualize_time_series()
        
        # 生成报告
        self.generate_summary_report(correlations, hypothesis_results)
        
        print("2024年3月至今的数据分析完成。")
        
    def load_and_process_data(self) -> None:
        """
        使用DataLoader加载和处理数据
        """
        from .data_loader import DataLoader
        
        # 创建数据加载器
        data_loader = DataLoader()
        
        # 加载数据
        print("正在加载数据...")
        news_data = data_loader.load_news_data()
        reddit_data = data_loader.load_reddit_data()
        stock_data = data_loader.load_stock_data()
        
        # 保存数据
        self.news_data = news_data
        self.reddit_data = reddit_data
        self.stock_data = stock_data
        
        # 筛选2024年3月至今的数据
        print("筛选2024年3月至今的数据...")
        filtered_news = data_loader.filter_recent_data(news_data, 'datetime')
        filtered_reddit = data_loader.filter_recent_data(reddit_data, 'datetime')
        filtered_stock = data_loader.filter_recent_data(stock_data, 'Date')
        
        # 合并数据
        print("合并数据...")
        self.recent_data = data_loader.merge_data(filtered_news, filtered_reddit, filtered_stock)
        
        # 情感分析
        print("进行情感分析...")
        self.analyze_recent_sentiments()
        
        print("数据加载和处理完成。")
        return self.recent_data 