import pandas as pd
import os
from datetime import datetime

# 要处理的股票列表
stocks = ['aapl', 'nvda', 'pltr', 'tsla', 'meta']

# 需要重命名的列
columns_to_rename = [
    'SMA1', 'SMA1_scaled', 
    'EMA0.2', 'EMA0.2_scaled', 
    'EMA0.1', 'EMA0.1_scaled', 
    'EMA0.05', 'EMA0.05_scaled', 
    'EMA0.02', 'EMA0.02_scaled'
]

# 设置过滤日期
start_date = pd.Timestamp('2024-03-01')

# 处理每个股票的数据
for stock in stocks:
    print(f"\n开始处理 {stock.upper()} 数据...")
    
    # 设置文件路径
    news_file = f'dataset/20250315/news_{stock}_compare1.csv'
    reddit_file = f'dataset/20250315/reddit_{stock}_compare1.csv'
    stock_file = f'dataset/20250315/stock_{stock}_compare1.csv'
    output_file = f'dataset/20250315/{stock}_merged_data.csv'
    
    # 检查文件是否存在
    files_exist = True
    for file_path in [news_file, reddit_file, stock_file]:
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在")
            files_exist = False
    
    if not files_exist:
        print(f"跳过 {stock.upper()} 处理，因为部分文件不存在")
        continue
    
    try:
        # 读取新闻数据
        news_df = pd.read_csv(news_file, index_col=0)
        # 确保索引被解析为日期时间
        news_df.index = pd.to_datetime(news_df.index)
        print(f"成功读取 {stock} 新闻数据，形状：{news_df.shape}")
        
        # 读取Reddit数据
        reddit_df = pd.read_csv(reddit_file, index_col=0)
        # 确保索引被解析为日期时间
        reddit_df.index = pd.to_datetime(reddit_df.index)
        print(f"成功读取 {stock} Reddit数据，形状：{reddit_df.shape}")
        
        # 读取股票数据
        stock_df = pd.read_csv(stock_file, index_col=0)
        # 确保索引被解析为日期时间
        stock_df.index = pd.to_datetime(stock_df.index)
        print(f"成功读取 {stock} 股票数据，形状：{stock_df.shape}")
        
        # 重命名列
        news_columns = {}
        reddit_columns = {}
        stock_columns = {}
        
        for col in columns_to_rename:
            if col in news_df.columns:
                news_columns[col] = f"news_{stock}_{col}"
            if col in reddit_df.columns:
                reddit_columns[col] = f"reddit_{stock}_{col}"
            if col in stock_df.columns:
                stock_columns[col] = f"stock_{stock}_{col}"
        
        # 应用重命名
        news_df = news_df.rename(columns=news_columns)
        reddit_df = reddit_df.rename(columns=reddit_columns)
        stock_df = stock_df.rename(columns=stock_columns)
        
        # 筛选数据
        news_filtered = news_df[news_df.index >= start_date]
        reddit_filtered = reddit_df[reddit_df.index >= start_date]
        stock_filtered = stock_df[stock_df.index >= start_date]
        
        print(f"筛选后的 {stock} 新闻数据行数：{len(news_filtered)}")
        print(f"筛选后的 {stock} Reddit数据行数：{len(reddit_filtered)}")
        print(f"筛选后的 {stock} 股票数据行数：{len(stock_filtered)}")
        
        # 合并数据
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
        
        print(f"{stock} 合并后的数据形状：{merged_df.shape}")
        
        # 保存合并后的数据
        merged_df.to_csv(output_file)
        print(f"{stock} 数据已保存到 {output_file}")
        
        # 统计缺失值
        missing_values = merged_df.isnull().sum().sum()
        print(f"{stock} 数据中的缺失值总数：{missing_values}")
        
    except Exception as e:
        print(f"处理 {stock} 数据时出错：{str(e)}")

print("\n所有股票数据处理完成！")

# 创建一个汇总文件，列出所有生成的数据文件
summary_file = 'dataset/20250315/merged_data_summary.txt'
try:
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Merged Data Files List:\n")
        for stock in stocks:
            output_file = f'dataset/20250315/{stock}_merged_data.csv'
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024  # KB
                f.write(f"{output_file}: {file_size:.2f} KB\n")
        
        # 检查SPY文件是否存在
        spy_file = 'dataset/20250315/spy_merged_data.csv'
        if os.path.exists(spy_file):
            f.write(f"\n{spy_file}: {os.path.getsize(spy_file) / 1024:.2f} KB\n")
    
    print(f"汇总信息已保存到 {summary_file}")
except Exception as e:
    print(f"创建汇总文件时出错: {str(e)}") 