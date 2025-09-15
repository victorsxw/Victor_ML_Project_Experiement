import pandas as pd
import os
import sys
import numpy as np

# 列出所有数据文件
data_dir = 'dataset/6datasets-2024-2025'
files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

print(f"Found {len(files)} pickle files: {files}")

# 检查所有文件的结构
for file_name in files:
    file_path = os.path.join(data_dir, file_name)
    print(f"\n{'='*50}")
    print(f"Examining file: {file_path}")
    
    try:
        df = pd.read_pickle(file_path)
        print(f"Data type: {type(df)}")
        
        if isinstance(df, pd.DataFrame):
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame index: {df.index.name or 'Unnamed'}")
            
            # 检查列名
            columns = df.columns.tolist()
            print(f"DataFrame columns ({len(columns)}): {columns}")
            
            # 检查数据类型
            print("\nData types:")
            for col in df.columns:
                print(f"  {col}: {df[col].dtype}")
            
            # 显示前几行数据
            print("\nFirst 3 rows:")
            print(df.head(3))
            
            # 检查是否有缺失值
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print("\nMissing values:")
                for col in df.columns:
                    if missing[col] > 0:
                        print(f"  {col}: {missing[col]} missing values")
            else:
                print("\nNo missing values found")
        else:
            print("Data is not a pandas DataFrame")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        
    # 限制输出，避免过多信息
    if files.index(file_name) >= 2:  # 只显示前3个文件的详细信息
        print(f"\nSkipping detailed analysis of remaining {len(files) - 3} files...")
        break

# 读取新闻数据
print("Reading news data...")
news_data = pd.read_csv('dataset/news_sentiment.csv')
print("\nNews data info:")
print(news_data.info())
print("\nNews data first 5 rows:")
print(news_data[['datetime', 'headline']].head())

print("\n" + "="*80 + "\n")

# 读取Reddit数据
print("Reading Reddit data...")
reddit_data = pd.read_csv('dataset/reddit_sentiment_data.csv')
print("\nReddit data info:")
print(reddit_data.info())
print("\nReddit data first 5 rows:")
print(reddit_data[['datetime', 'headline']].head())

# 比较数据
print("\nData Comparison:")
print(f"Total records - News: {len(news_data)}, Reddit: {len(reddit_data)}")

# 检查标题重复
print("\nChecking for duplicate headlines:")
news_duplicates = news_data['headline'].duplicated().sum()
reddit_duplicates = reddit_data['headline'].duplicated().sum()
print(f"Duplicate headlines in news data: {news_duplicates}")
print(f"Duplicate headlines in reddit data: {reddit_duplicates}")

# 检查内容重叠
print("\nChecking content overlap:")
common_headlines = set(news_data['headline']) & set(reddit_data['headline'])
print(f"Number of headlines appearing in both datasets: {len(common_headlines)}")
print(f"Percentage of overlap: {len(common_headlines)/len(news_data)*100:.2f}%")

# 显示一些不同的标题示例
print("\nSample of unique news headlines:")
unique_news = set(news_data['headline']) - set(reddit_data['headline'])
if unique_news:
    print("\n".join(list(unique_news)[:5]))
else:
    print("No unique news headlines found!")

print("\nSample of unique reddit headlines:")
unique_reddit = set(reddit_data['headline']) - set(news_data['headline'])
if unique_reddit:
    print("\n".join(list(unique_reddit)[:5]))
else:
    print("No unique reddit headlines found!")

# 检查时间分布
print("\nChecking datetime distribution:")
print("\nNews data date range:")
print(f"Start: {news_data['datetime'].min()}")
print(f"End: {news_data['datetime'].max()}")

print("\nReddit data date range:")
print(f"Start: {reddit_data['datetime'].min()}")
print(f"End: {reddit_data['datetime'].max()}") 