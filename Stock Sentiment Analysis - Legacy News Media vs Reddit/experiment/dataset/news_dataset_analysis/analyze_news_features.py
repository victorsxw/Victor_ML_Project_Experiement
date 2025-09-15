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
output_dir = 'dataset/news_dataset_analysis'
os.makedirs(output_dir, exist_ok=True)

# 读取新闻数据
print("正在读取新闻数据...")
try:
    file_path = 'dataset/news_sentiment.pkl'
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        exit(1)
    
    news_data = pd.read_pickle(file_path)
    print(f"成功读取数据，大小：{news_data.shape}")
    
    # 将PKL转换为CSV
    csv_path = 'dataset/news_sentiment.csv'
    print(f"正在将PKL文件转换为CSV文件: {csv_path}")
    news_data.to_csv(csv_path, index=False)
    print(f"转换完成！CSV文件已保存到: {csv_path}")
    
except Exception as e:
    print(f"读取数据时出错：{str(e)}")
    exit(1)

# 1. 基本信息
print("\n1. 数据基本信息：")
print("\n数据形状：", news_data.shape)
print("\n列名：", news_data.columns.tolist())
print("\n数据类型：")
print(news_data.dtypes)
print("\n数据描述性统计：")
print(news_data.describe(include='all'))

# 2. 时间分布分析
print("\n2. 时间分布分析：")
if not isinstance(news_data.index, pd.DatetimeIndex):
    if 'datetime' in news_data.columns:
        news_data['datetime'] = pd.to_datetime(news_data['datetime'])
        date_column = 'datetime'
    else:
        print("警告：未找到时间列！")
        date_column = None
else:
    date_column = news_data.index.name or 'index'
    if date_column == 'index':
        news_data = news_data.reset_index()

if date_column:
    print("\n时间范围：")
    print("开始时间：", news_data[date_column].min())
    print("结束时间：", news_data[date_column].max())

    # 按天统计新闻数量
    daily_news = news_data.groupby(news_data[date_column].dt.date).size()
    print("\n每日新闻数量统计：")
    print("最少新闻数：", daily_news.min())
    print("最多新闻数：", daily_news.max())
    print("平均新闻数：", daily_news.mean())

    # 绘制每日新闻数量趋势图
    plt.figure(figsize=(15, 6))
    daily_news.plot()
    plt.title('每日新闻数量趋势')
    plt.xlabel('日期')
    plt.ylabel('新闻数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'news_daily_count.png'))
    plt.close()

# 3. 情感分析特征分析
sentiment_columns = [col for col in news_data.columns if 'sentiment' in col]
print("\n3. 情感分析特征：")
for col in sentiment_columns:
    print(f"\n{col} 的值分布：")
    value_counts = news_data[col].value_counts()
    print(value_counts)
    
    # 绘制情感分布饼图
    plt.figure(figsize=(10, 6))
    plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
    plt.title(f'{col} 分布')
    plt.savefig(os.path.join(output_dir, f'news_{col}_distribution.png'))
    plt.close()

# 4. 标题长度分析
if 'headline' in news_data.columns:
    news_data['headline_length'] = news_data['headline'].str.len()
    print("\n4. 标题长度分析：")
    print("\n标题长度统计：")
    print(news_data['headline_length'].describe())

    # 绘制标题长度分布直方图
    plt.figure(figsize=(12, 6))
    sns.histplot(data=news_data, x='headline_length', bins=50)
    plt.title('新闻标题长度分布')
    plt.xlabel('标题长度')
    plt.ylabel('频数')
    plt.savefig(os.path.join(output_dir, 'news_headline_length_distribution.png'))
    plt.close()
else:
    print("\n警告：数据中未找到headline列！")

# 5. 情感标签一致性分析
print("\n5. 不同情感分析模型的标签一致性：")
for i in range(len(sentiment_columns)):
    for j in range(i+1, len(sentiment_columns)):
        agreement = (news_data[sentiment_columns[i]] == news_data[sentiment_columns[j]]).mean()
        print(f"{sentiment_columns[i]} 和 {sentiment_columns[j]} 的一致性: {agreement:.2%}")

# 6. 样本数据展示
print("\n6. 数据样本展示（前5条）：")
print(news_data.head())

# 7. 分析prompt和sentiment的关系
prompt_columns = [col for col in news_data.columns if 'prompt' in col]
if prompt_columns:
    print("\n7. Prompt分析：")
    for prompt_col in prompt_columns:
        print(f"\n{prompt_col} 的唯一值：")
        unique_prompts = news_data[prompt_col].unique()
        print(f"唯一prompt数量：{len(unique_prompts)}")
        if len(unique_prompts) < 10:  # 如果唯一值较少，显示所有值
            print("具体prompt值：")
            for prompt in unique_prompts:
                print(f"- {prompt}")
else:
    print("\n警告：数据中未找到prompt列！")

# 保存分析结果到文件
report_path = os.path.join(output_dir, 'news_data_analysis.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("新闻数据分析报告\n")
    f.write("="*50 + "\n\n")
    
    f.write("1. 数据基本信息\n")
    f.write("-"*30 + "\n")
    f.write(f"总记录数：{len(news_data)}\n")
    f.write(f"特征数量：{len(news_data.columns)}\n")
    f.write(f"特征列表：{', '.join(news_data.columns)}\n\n")
    
    if date_column:
        f.write("2. 时间范围\n")
        f.write("-"*30 + "\n")
        f.write(f"开始时间：{news_data[date_column].min()}\n")
        f.write(f"结束时间：{news_data[date_column].max()}\n")
        f.write(f"每日平均新闻数：{daily_news.mean():.2f}\n\n")
    
    f.write("3. 情感分析特征\n")
    f.write("-"*30 + "\n")
    for col in sentiment_columns:
        f.write(f"\n{col} 分布：\n")
        f.write(news_data[col].value_counts().to_string())
        f.write("\n")
    
    if 'headline' in news_data.columns:
        f.write("\n4. 标题长度统计\n")
        f.write("-"*30 + "\n")
        f.write(news_data['headline_length'].describe().to_string())
    
    if prompt_columns:
        f.write("\n\n5. Prompt分析\n")
        f.write("-"*30 + "\n")
        for prompt_col in prompt_columns:
            f.write(f"\n{prompt_col}：\n")
            unique_prompts = news_data[prompt_col].unique()
            f.write(f"唯一prompt数量：{len(unique_prompts)}\n")
            if len(unique_prompts) < 10:
                f.write("具体prompt值：\n")
                for prompt in unique_prompts:
                    f.write(f"- {prompt}\n")
    
print(f"\n分析完成！结果已保存到 {report_path}") 