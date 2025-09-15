import pandas as pd
import os
from datetime import datetime

# 设置文件路径
news_file = 'dataset/20250315/news_spy_compare1.csv'
reddit_file = 'dataset/20250315/reddit_spy_compare1.csv'
stock_file = 'dataset/20250315/stock_spy_compare1.csv'
output_file = 'dataset/20250315/spy_merged_data.csv'

# 检查文件是否存在
for file_path in [news_file, reddit_file, stock_file]:
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        exit(1)

print("正在读取数据文件...")

# 读取数据
try:
    # 读取新闻数据
    news_df = pd.read_csv(news_file, index_col=0)
    # 确保索引被解析为日期时间
    news_df.index = pd.to_datetime(news_df.index)
    print(f"成功读取新闻数据，形状：{news_df.shape}")
    
    # 读取Reddit数据
    reddit_df = pd.read_csv(reddit_file, index_col=0)
    # 确保索引被解析为日期时间
    reddit_df.index = pd.to_datetime(reddit_df.index)
    print(f"成功读取Reddit数据，形状：{reddit_df.shape}")
    
    # 读取股票数据
    stock_df = pd.read_csv(stock_file, index_col=0)
    # 确保索引被解析为日期时间
    stock_df.index = pd.to_datetime(stock_df.index)
    print(f"成功读取股票数据，形状：{stock_df.shape}")
    
except Exception as e:
    print(f"读取数据时出错：{str(e)}")
    exit(1)

# 查看每个数据集的列
print("\n新闻数据列：", news_df.columns.tolist())
print("\nReddit数据列：", reddit_df.columns.tolist())
print("\n股票数据列：", stock_df.columns.tolist())

# 需要重命名的列
columns_to_rename = [
    'SMA1', 'SMA1_scaled', 
    'EMA0.2', 'EMA0.2_scaled', 
    'EMA0.1', 'EMA0.1_scaled', 
    'EMA0.05', 'EMA0.05_scaled', 
    'EMA0.02', 'EMA0.02_scaled'
]

# 重命名列
print("\n正在重命名列...")
news_columns = {}
reddit_columns = {}
stock_columns = {}

for col in columns_to_rename:
    if col in news_df.columns:
        news_columns[col] = f"news_spy_{col}"
    if col in reddit_df.columns:
        reddit_columns[col] = f"reddit_spy_{col}"
    if col in stock_df.columns:
        stock_columns[col] = f"stock_spy_{col}"

# 应用重命名
news_df = news_df.rename(columns=news_columns)
reddit_df = reddit_df.rename(columns=reddit_columns)
stock_df = stock_df.rename(columns=stock_columns)

print("列重命名完成")

# 设置过滤日期
start_date = pd.Timestamp('2024-03-01')
print(f"\n筛选从 {start_date} 开始的数据...")

# 筛选数据
news_filtered = news_df[news_df.index >= start_date]
reddit_filtered = reddit_df[reddit_df.index >= start_date]
stock_filtered = stock_df[stock_df.index >= start_date]

print(f"筛选后的新闻数据行数：{len(news_filtered)}")
print(f"筛选后的Reddit数据行数：{len(reddit_filtered)}")
print(f"筛选后的股票数据行数：{len(stock_filtered)}")

# 合并数据
print("\n正在合并数据...")

# 首先合并新闻和Reddit数据
merged_df = pd.merge(
    news_filtered[list(news_columns.values())], 
    reddit_filtered[list(reddit_columns.values())],
    left_index=True, 
    right_index=True, 
    how='outer'
)

# 然后合并股票数据
merged_df = pd.merge(
    merged_df, 
    stock_filtered[list(stock_columns.values())],
    left_index=True, 
    right_index=True, 
    how='outer'
)

print(f"合并后的数据形状：{merged_df.shape}")

# 打印合并后的列
print("\n合并后的数据列：", merged_df.columns.tolist())

# 保存合并后的数据
print(f"\n正在保存合并后的数据到 {output_file}...")
merged_df.to_csv(output_file)
print("保存完成！")

# 显示一些结果样本
print("\n合并后的数据前5行：")
print(merged_df.head())

# 检查是否有缺失值
missing_values = merged_df.isnull().sum()
print("\n缺失值统计：")
print(missing_values)

print("\n处理完成！") 