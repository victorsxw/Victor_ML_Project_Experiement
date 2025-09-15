import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = 'dataset/reddit_dataset_analysis'
os.makedirs(output_dir, exist_ok=True)

# 读取Reddit数据
print("正在读取Reddit数据...")
try:
    file_path = 'dataset/reddit_sentiment.pkl'
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        exit(1)
    
    reddit_data = pd.read_pickle(file_path)
    print(f"成功读取数据，大小：{reddit_data.shape}")
    
    # 将PKL转换为CSV
    csv_path = 'dataset/reddit_sentiment_data.csv'
    print(f"正在将PKL文件转换为CSV文件: {csv_path}")
    reddit_data.to_csv(csv_path, index=False)
    print(f"转换完成！CSV文件已保存到: {csv_path}")
    
except Exception as e:
    print(f"读取数据时出错：{str(e)}")
    exit(1)

# 1. 基本信息
print("\n1. 数据基本信息：")
print("\n数据形状：", reddit_data.shape)
print("\n列名：", reddit_data.columns.tolist())
print("\n数据类型：")
print(reddit_data.dtypes)
print("\n数据描述性统计：")
print(reddit_data.describe(include='all'))

# 2. 时间分布分析
print("\n2. 时间分布分析：")
if 'datetime' in reddit_data.columns:
    # 确保datetime列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(reddit_data['datetime']):
        reddit_data['datetime'] = pd.to_datetime(reddit_data['datetime'])
    
    print("\n时间范围：")
    print("开始时间：", reddit_data['datetime'].min())
    print("结束时间：", reddit_data['datetime'].max())

    # 按天统计帖子数量
    daily_posts = reddit_data.groupby(reddit_data['datetime'].dt.date).size()
    print("\n每日帖子数量统计：")
    print("最少帖子数：", daily_posts.min())
    print("最多帖子数：", daily_posts.max())
    print("平均帖子数：", daily_posts.mean())

    # 绘制每日帖子数量趋势图
    plt.figure(figsize=(15, 6))
    daily_posts.plot()
    plt.title('每日Reddit帖子数量趋势')
    plt.xlabel('日期')
    plt.ylabel('帖子数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reddit_daily_posts.png'))
    plt.close()
else:
    print("警告：数据中未找到datetime列！")

# 3. 情感分析特征分析
sentiment_columns = [col for col in reddit_data.columns if 'sentiment' in col]
print("\n3. 情感分析特征：")
if sentiment_columns:
    for col in sentiment_columns:
        print(f"\n{col} 的值分布：")
        value_counts = reddit_data[col].value_counts()
        print(value_counts)
        
        # 绘制情感分布饼图
        plt.figure(figsize=(10, 6))
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        plt.title(f'{col} 分布')
        plt.savefig(os.path.join(output_dir, f'reddit_{col}_distribution.png'))
        plt.close()
else:
    print("警告：数据中未找到情感分析列！")

# 4. 标题长度分析
if 'title' in reddit_data.columns:
    reddit_data['title_length'] = reddit_data['title'].str.len()
    print("\n4. 标题长度分析：")
    print("\n标题长度统计：")
    print(reddit_data['title_length'].describe())

    # 绘制标题长度分布直方图
    plt.figure(figsize=(12, 6))
    sns.histplot(data=reddit_data, x='title_length', bins=50)
    plt.title('Reddit标题长度分布')
    plt.xlabel('标题长度')
    plt.ylabel('频数')
    plt.savefig(os.path.join(output_dir, 'reddit_title_length_distribution.png'))
    plt.close()
else:
    print("\n警告：数据中未找到title列！")

# 5. 分析answer和sentiment的关系
answer_columns = [col for col in reddit_data.columns if 'answer' in col]
if answer_columns:
    print("\n5. Answer分析：")
    for answer_col in answer_columns:
        print(f"\n{answer_col} 的值分布：")
        value_counts = reddit_data[answer_col].value_counts()
        print(value_counts)
        
        # 绘制分布饼图
        plt.figure(figsize=(10, 6))
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        plt.title(f'{answer_col} 分布')
        plt.savefig(os.path.join(output_dir, f'reddit_{answer_col}_distribution.png'))
        plt.close()
else:
    print("\n警告：数据中未找到answer列！")

# 6. 其他特征分析
if 'upvote_ratio' in reddit_data.columns and 'num_comments' in reddit_data.columns:
    print("\n6. 互动数据分析：")
    print("\n点赞率统计：")
    print(reddit_data['upvote_ratio'].describe())
    print("\n评论数统计：")
    print(reddit_data['num_comments'].describe())

    # 绘制点赞率分布图
    plt.figure(figsize=(12, 6))
    sns.histplot(data=reddit_data, x='upvote_ratio', bins=50)
    plt.title('点赞率分布')
    plt.xlabel('点赞率')
    plt.ylabel('频数')
    plt.savefig(os.path.join(output_dir, 'reddit_upvote_ratio_distribution.png'))
    plt.close()

    # 绘制评论数分布图
    plt.figure(figsize=(12, 6))
    sns.histplot(data=reddit_data, x='num_comments', bins=50)
    plt.title('评论数分布')
    plt.xlabel('评论数')
    plt.ylabel('频数')
    plt.savefig(os.path.join(output_dir, 'reddit_comments_distribution.png'))
    plt.close()

# 7. 样本数据展示
print("\n7. 数据样本展示（前5条）：")
print(reddit_data.head())

# 保存分析结果到文件
report_path = os.path.join(output_dir, 'reddit_data_analysis.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("Reddit数据分析报告\n")
    f.write("="*50 + "\n\n")
    
    f.write("1. 数据基本信息\n")
    f.write("-"*30 + "\n")
    f.write(f"总记录数：{len(reddit_data)}\n")
    f.write(f"特征数量：{len(reddit_data.columns)}\n")
    f.write(f"特征列表：{', '.join(reddit_data.columns)}\n\n")
    
    if 'datetime' in reddit_data.columns:
        f.write("2. 时间范围\n")
        f.write("-"*30 + "\n")
        f.write(f"开始时间：{reddit_data['datetime'].min()}\n")
        f.write(f"结束时间：{reddit_data['datetime'].max()}\n")
        f.write(f"每日平均帖子数：{daily_posts.mean():.2f}\n\n")
    
    if sentiment_columns:
        f.write("3. 情感分析特征\n")
        f.write("-"*30 + "\n")
        for col in sentiment_columns:
            f.write(f"\n{col} 分布：\n")
            f.write(reddit_data[col].value_counts().to_string())
            f.write("\n")
    
    if 'title' in reddit_data.columns:
        f.write("\n4. 标题长度统计\n")
        f.write("-"*30 + "\n")
        f.write(reddit_data['title_length'].describe().to_string())
    
    if answer_columns:
        f.write("\n\n5. Answer分析\n")
        f.write("-"*30 + "\n")
        for answer_col in answer_columns:
            f.write(f"\n{answer_col} 分布：\n")
            f.write(reddit_data[answer_col].value_counts().to_string())
            f.write("\n")
    
    if 'upvote_ratio' in reddit_data.columns and 'num_comments' in reddit_data.columns:
        f.write("\n\n6. 互动数据统计\n")
        f.write("-"*30 + "\n")
        f.write("\n点赞率统计：\n")
        f.write(reddit_data['upvote_ratio'].describe().to_string())
        f.write("\n\n评论数统计：\n")
        f.write(reddit_data['num_comments'].describe().to_string())
    
print(f"\n分析完成！结果已保存到 {report_path}") 