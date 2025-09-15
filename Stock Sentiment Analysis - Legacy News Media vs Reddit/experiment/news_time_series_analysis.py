import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    print("警告：无法设置中文字体，图表中的中文可能无法正确显示")

# 创建输出目录
output_dir = "news_time_series_analysis"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
print("正在加载新闻情感数据...")
news_data = pd.read_csv("dataset/news_sentiment.csv")
print(f"数据加载完成，共 {news_data.shape[0]} 条记录")

# 数据预处理
print("正在进行数据预处理...")
# 转换日期时间
news_data['datetime'] = pd.to_datetime(news_data['datetime'])
# 确保日期列是日期时间格式
news_data['date'] = pd.to_datetime(news_data['date'])

# 提取情感列
sentiment_columns = [col for col in news_data.columns if 'sentiment' in col]
print(f"情感分析列: {sentiment_columns}")

# 选择chat2模型的情感分析结果
sentiment_col = 'sentiment_chat2'
if sentiment_col in news_data.columns:
    print(f"使用 {sentiment_col} 列进行分析")
    
    # 检查情感值的分布
    sentiment_counts = news_data[sentiment_col].value_counts()
    print("\n情感分布:")
    print(sentiment_counts)
    
    # 按日期聚合数据
    print("\n按日期聚合数据...")
    # 创建日期列（只包含年月日）
    news_data['date_only'] = news_data['datetime'].dt.date
    
    # 按日期和情感分组，计算每种情感的数量
    daily_sentiment = news_data.groupby(['date_only', sentiment_col]).size().unstack(fill_value=0)
    
    # 如果列名不是positive, negative, neutral，则重命名
    if 'positive' not in daily_sentiment.columns:
        # 尝试找到可能的积极、消极、中性列名
        for col in daily_sentiment.columns:
            if 'pos' in col.lower():
                daily_sentiment = daily_sentiment.rename(columns={col: 'positive'})
            elif 'neg' in col.lower():
                daily_sentiment = daily_sentiment.rename(columns={col: 'negative'})
            elif 'neu' in col.lower():
                daily_sentiment = daily_sentiment.rename(columns={col: 'neutral'})
    
    # 计算情感比例
    daily_sentiment_ratio = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0)
    
    # 计算情感得分（正面-负面）
    if 'positive' in daily_sentiment.columns and 'negative' in daily_sentiment.columns:
        daily_sentiment['sentiment_score'] = daily_sentiment['positive'] - daily_sentiment['negative']
        daily_sentiment_ratio['sentiment_score'] = daily_sentiment_ratio.get('positive', 0) - daily_sentiment_ratio.get('negative', 0)
    
    # 时间序列可视化
    print("\n生成时间序列可视化...")
    
    # 1. 每日情感数量时间序列
    plt.figure(figsize=(15, 8))
    for col in daily_sentiment.columns:
        if col != 'sentiment_score':
            plt.plot(daily_sentiment.index, daily_sentiment[col], label=col)
    plt.title('每日新闻情感数量时间序列')
    plt.xlabel('日期')
    plt.ylabel('数量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daily_sentiment_count.png")
    plt.close()
    
    # 2. 每日情感比例时间序列
    plt.figure(figsize=(15, 8))
    for col in daily_sentiment_ratio.columns:
        if col != 'sentiment_score':
            plt.plot(daily_sentiment_ratio.index, daily_sentiment_ratio[col], label=col)
    plt.title('每日新闻情感比例时间序列')
    plt.xlabel('日期')
    plt.ylabel('比例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daily_sentiment_ratio.png")
    plt.close()
    
    # 3. 情感得分时间序列
    if 'sentiment_score' in daily_sentiment.columns:
        plt.figure(figsize=(15, 8))
        plt.plot(daily_sentiment.index, daily_sentiment['sentiment_score'], 'g-')
        plt.title('每日新闻情感得分时间序列 (正面-负面)')
        plt.xlabel('日期')
        plt.ylabel('情感得分')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/daily_sentiment_score.png")
        plt.close()
    
    # 4. 情感堆叠面积图
    plt.figure(figsize=(15, 8))
    sentiment_cols = [col for col in daily_sentiment.columns if col != 'sentiment_score']
    daily_sentiment[sentiment_cols].plot.area(stacked=True, alpha=0.7, ax=plt.gca())
    plt.title('每日新闻情感数量堆叠面积图')
    plt.xlabel('日期')
    plt.ylabel('数量')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daily_sentiment_stacked_area.png")
    plt.close()
    
    # 5. 情感比例堆叠面积图
    plt.figure(figsize=(15, 8))
    sentiment_ratio_cols = [col for col in daily_sentiment_ratio.columns if col != 'sentiment_score']
    daily_sentiment_ratio[sentiment_ratio_cols].plot.area(stacked=True, alpha=0.7, ax=plt.gca())
    plt.title('每日新闻情感比例堆叠面积图')
    plt.xlabel('日期')
    plt.ylabel('比例')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daily_sentiment_ratio_stacked_area.png")
    plt.close()
    
    # 6. 按年度分析情感变化
    print("\n按年度分析情感变化...")
    # 提取年份
    news_data['year'] = news_data['datetime'].dt.year
    yearly_sentiment = news_data.groupby(['year', sentiment_col]).size().unstack(fill_value=0)
    yearly_sentiment_ratio = yearly_sentiment.div(yearly_sentiment.sum(axis=1), axis=0)
    
    # 年度情感比例柱状图
    plt.figure(figsize=(15, 8))
    yearly_sentiment_ratio.plot(kind='bar', stacked=False, alpha=0.7, ax=plt.gca())
    plt.title('年度新闻情感比例')
    plt.xlabel('年份')
    plt.ylabel('比例')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/yearly_sentiment_ratio.png")
    plt.close()
    
    # 7. 按月度分析情感变化
    print("\n按月度分析情感变化...")
    # 提取年月
    news_data['year_month'] = news_data['datetime'].dt.to_period('M')
    monthly_sentiment = news_data.groupby(['year_month', sentiment_col]).size().unstack(fill_value=0)
    monthly_sentiment_ratio = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0)
    
    # 月度情感比例热图
    # 将Period索引转换为datetime
    monthly_sentiment_ratio.index = monthly_sentiment_ratio.index.to_timestamp()
    
    # 创建年份和月份列
    monthly_data = monthly_sentiment_ratio.reset_index()
    monthly_data['year'] = monthly_data['year_month'].dt.year
    monthly_data['month'] = monthly_data['year_month'].dt.month
    
    # 透视表以创建热图数据
    if 'positive' in monthly_sentiment_ratio.columns:
        heatmap_data = monthly_data.pivot(index='month', columns='year', values='positive')
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(heatmap_data, cmap='RdYlGn', annot=True, fmt='.2f', linewidths=.5)
        plt.title('月度正面情感比例热图')
        plt.xlabel('年份')
        plt.ylabel('月份')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_positive_sentiment_heatmap.png")
        plt.close()
    
    # 8. 最近一年的情感变化
    print("\n分析最近一年的情感变化...")
    # 获取最近一年的数据
    last_year = datetime.now().year - 1
    recent_data = news_data[news_data['datetime'].dt.year >= last_year]
    
    if not recent_data.empty:
        # 按日期聚合
        recent_data['date_only'] = recent_data['datetime'].dt.date
        recent_daily = recent_data.groupby(['date_only', sentiment_col]).size().unstack(fill_value=0)
        
        # 计算情感比例
        recent_daily_ratio = recent_daily.div(recent_daily.sum(axis=1), axis=0)
        
        # 最近一年情感变化折线图
        plt.figure(figsize=(15, 8))
        for col in recent_daily_ratio.columns:
            plt.plot(recent_daily_ratio.index, recent_daily_ratio[col], label=col)
        plt.title(f'{last_year}-{datetime.now().year}年新闻情感比例变化')
        plt.xlabel('日期')
        plt.ylabel('比例')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/recent_year_sentiment_ratio.png")
        plt.close()
    
    # 9. 2024年3月至今的情感变化
    print("\n分析2024年3月至今的情感变化...")
    # 筛选2024年3月至今的数据
    start_date = pd.Timestamp('2024-03-01')
    recent_2024_data = news_data[news_data['datetime'] >= start_date]
    
    if not recent_2024_data.empty:
        # 按日期聚合
        recent_2024_data['date_only'] = recent_2024_data['datetime'].dt.date
        recent_2024_daily = recent_2024_data.groupby(['date_only', sentiment_col]).size().unstack(fill_value=0)
        
        # 计算情感比例
        recent_2024_daily_ratio = recent_2024_daily.div(recent_2024_daily.sum(axis=1), axis=0)
        
        # 2024年3月至今情感变化折线图
        plt.figure(figsize=(15, 8))
        for col in recent_2024_daily_ratio.columns:
            plt.plot(recent_2024_daily_ratio.index, recent_2024_daily_ratio[col], label=col, marker='o')
        plt.title('2024年3月至今新闻情感比例变化')
        plt.xlabel('日期')
        plt.ylabel('比例')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/2024_march_to_now_sentiment_ratio.png")
        plt.close()
        
        # 2024年3月至今情感数量柱状图
        plt.figure(figsize=(15, 8))
        recent_2024_daily.plot(kind='bar', stacked=True, alpha=0.7, ax=plt.gca())
        plt.title('2024年3月至今每日新闻情感数量')
        plt.xlabel('日期')
        plt.ylabel('数量')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/2024_march_to_now_sentiment_count.png")
        plt.close()
    
    print(f"\n分析完成！可视化结果已保存到 {output_dir} 目录")
else:
    print(f"错误：数据中不存在 {sentiment_col} 列") 