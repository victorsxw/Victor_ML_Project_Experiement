import pandas as pd
import os
import sys
import numpy as np

# 设置要检查的数据目录
data_dir = 'dataset/20250315'

# 检查目录是否存在
if not os.path.exists(data_dir):
    print(f"错误：目录 {data_dir} 不存在")
    sys.exit(1)

# 列出所有数据文件
files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
pkl_files = [f for f in files if f.endswith('.pkl')]
csv_files = [f for f in files if f.endswith('.csv')]
other_files = [f for f in files if not (f.endswith('.pkl') or f.endswith('.csv'))]

print(f"在 {data_dir} 目录中找到:")
print(f"- {len(pkl_files)} 个PKL文件: {pkl_files}")
print(f"- {len(csv_files)} 个CSV文件: {csv_files}")
print(f"- {len(other_files)} 个其他文件: {other_files}")

# 检查所有PKL文件的结构
print("\n===== 检查PKL文件 =====")
for file_name in pkl_files:
    file_path = os.path.join(data_dir, file_name)
    print(f"\n{'='*50}")
    print(f"检查文件: {file_path}")
    
    try:
        df = pd.read_pickle(file_path)
        print(f"数据类型: {type(df)}")
        
        if isinstance(df, pd.DataFrame):
            print(f"DataFrame形状: {df.shape}")
            print(f"DataFrame索引: {df.index.name or '未命名'}")
            
            # 检查列名
            columns = df.columns.tolist()
            print(f"DataFrame列 ({len(columns)}): {columns}")
            
            # 检查数据类型
            print("\n数据类型:")
            for col in df.columns:
                print(f"  {col}: {df[col].dtype}")
            
            # 显示前几行数据
            print("\n前3行数据:")
            print(df.head(3))
            
            # 检查是否有缺失值
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print("\n缺失值:")
                for col in df.columns:
                    if missing[col] > 0:
                        print(f"  {col}: {missing[col]} 个缺失值")
            else:
                print("\n没有发现缺失值")
                
            # 检查时间范围（如果有日期列）
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                print("\n时间范围:")
                for date_col in date_cols:
                    try:
                        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                            print(f"  {date_col}: {df[date_col].min()} 至 {df[date_col].max()}")
                        else:
                            # 尝试转换为日期时间
                            print(f"  {date_col}: 不是日期时间类型，尝试转换...")
                            try:
                                dates = pd.to_datetime(df[date_col])
                                print(f"  {date_col} (转换后): {dates.min()} 至 {dates.max()}")
                            except:
                                print(f"  {date_col}: 无法转换为日期时间类型")
                    except:
                        print(f"  {date_col}: 无法确定时间范围")
        else:
            print("数据不是pandas DataFrame")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")

# 检查所有CSV文件的结构
print("\n\n===== 检查CSV文件 =====")
for file_name in csv_files:
    file_path = os.path.join(data_dir, file_name)
    print(f"\n{'='*50}")
    print(f"检查文件: {file_path}")
    
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(file_path)
        print(f"数据类型: {type(df)}")
        
        if isinstance(df, pd.DataFrame):
            print(f"DataFrame形状: {df.shape}")
            print(f"DataFrame索引: {df.index.name or '未命名'}")
            
            # 检查列名
            columns = df.columns.tolist()
            print(f"DataFrame列 ({len(columns)}): {columns}")
            
            # 检查数据类型
            print("\n数据类型:")
            for col in df.columns:
                print(f"  {col}: {df[col].dtype}")
            
            # 显示前几行数据
            print("\n前3行数据:")
            print(df.head(3))
            
            # 检查是否有缺失值
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print("\n缺失值:")
                for col in df.columns:
                    if missing[col] > 0:
                        print(f"  {col}: {missing[col]} 个缺失值")
            else:
                print("\n没有发现缺失值")
                
            # 检查时间范围（如果有日期列）
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                print("\n时间范围:")
                for date_col in date_cols:
                    try:
                        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                            print(f"  {date_col}: {df[date_col].min()} 至 {df[date_col].max()}")
                        else:
                            # 尝试转换为日期时间
                            print(f"  {date_col}: 不是日期时间类型，尝试转换...")
                            try:
                                dates = pd.to_datetime(df[date_col])
                                print(f"  {date_col} (转换后): {dates.min()} 至 {dates.max()}")
                            except:
                                print(f"  {date_col}: 无法转换为日期时间类型")
                    except:
                        print(f"  {date_col}: 无法确定时间范围")
                        
            # 检查特定列（如果存在）
            sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
            if sentiment_cols:
                print("\n情感分析列:")
                for col in sentiment_cols:
                    try:
                        value_counts = df[col].value_counts()
                        print(f"  {col} 值分布:")
                        print(f"  {value_counts}")
                    except:
                        print(f"  {col}: 无法计算值分布")
        else:
            print("数据不是pandas DataFrame")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")

# 如果有新闻和Reddit数据，比较它们
news_files = [f for f in files if 'news' in f.lower()]
reddit_files = [f for f in files if 'reddit' in f.lower()]

if news_files and reddit_files:
    print("\n\n===== 新闻和Reddit数据比较 =====")
    
    # 尝试加载第一个新闻文件和第一个Reddit文件
    try:
        news_file = os.path.join(data_dir, news_files[0])
        reddit_file = os.path.join(data_dir, reddit_files[0])
        
        print(f"比较文件: {news_file} 和 {reddit_file}")
        
        # 根据文件扩展名加载数据
        if news_file.endswith('.csv'):
            news_data = pd.read_csv(news_file)
        elif news_file.endswith('.pkl'):
            news_data = pd.read_pickle(news_file)
        else:
            print(f"无法加载新闻文件: 不支持的格式")
            news_data = None
            
        if reddit_file.endswith('.csv'):
            reddit_data = pd.read_csv(reddit_file)
        elif reddit_file.endswith('.pkl'):
            reddit_data = pd.read_pickle(reddit_file)
        else:
            print(f"无法加载Reddit文件: 不支持的格式")
            reddit_data = None
        
        if news_data is not None and reddit_data is not None:
            print("\n数据比较:")
            print(f"总记录数 - 新闻: {len(news_data)}, Reddit: {len(reddit_data)}")
            
            # 检查标题列
            news_title_col = next((col for col in news_data.columns if 'headline' in col.lower() or 'title' in col.lower()), None)
            reddit_title_col = next((col for col in reddit_data.columns if 'headline' in col.lower() or 'title' in col.lower()), None)
            
            if news_title_col and reddit_title_col:
                # 检查标题重复
                print("\n检查标题重复:")
                news_duplicates = news_data[news_title_col].duplicated().sum()
                reddit_duplicates = reddit_data[reddit_title_col].duplicated().sum()
                print(f"新闻数据中的重复标题: {news_duplicates}")
                print(f"Reddit数据中的重复标题: {reddit_duplicates}")
                
                # 检查内容重叠
                print("\n检查内容重叠:")
                common_titles = set(news_data[news_title_col]) & set(reddit_data[reddit_title_col])
                print(f"同时出现在两个数据集中的标题数量: {len(common_titles)}")
                if len(news_data) > 0:
                    print(f"重叠百分比: {len(common_titles)/len(news_data)*100:.2f}%")
                
                # 显示一些不同的标题示例
                print("\n新闻独有标题示例:")
                unique_news = set(news_data[news_title_col]) - set(reddit_data[reddit_title_col])
                if unique_news:
                    print("\n".join(list(unique_news)[:5]))
                else:
                    print("没有找到新闻独有标题!")
                
                print("\nReddit独有标题示例:")
                unique_reddit = set(reddit_data[reddit_title_col]) - set(news_data[news_title_col])
                if unique_reddit:
                    print("\n".join(list(unique_reddit)[:5]))
                else:
                    print("没有找到Reddit独有标题!")
            
            # 检查时间分布
            news_date_col = next((col for col in news_data.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            reddit_date_col = next((col for col in reddit_data.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            
            if news_date_col and reddit_date_col:
                print("\n检查时间分布:")
                
                # 尝试转换为日期时间类型
                try:
                    if not pd.api.types.is_datetime64_any_dtype(news_data[news_date_col]):
                        news_dates = pd.to_datetime(news_data[news_date_col])
                    else:
                        news_dates = news_data[news_date_col]
                        
                    if not pd.api.types.is_datetime64_any_dtype(reddit_data[reddit_date_col]):
                        reddit_dates = pd.to_datetime(reddit_data[reddit_date_col])
                    else:
                        reddit_dates = reddit_data[reddit_date_col]
                    
                    print("\n新闻数据日期范围:")
                    print(f"开始: {news_dates.min()}")
                    print(f"结束: {news_dates.max()}")
                    
                    print("\nReddit数据日期范围:")
                    print(f"开始: {reddit_dates.min()}")
                    print(f"结束: {reddit_dates.max()}")
                except Exception as e:
                    print(f"处理日期时出错: {str(e)}")
    except Exception as e:
        print(f"比较数据时出错: {str(e)}")

print("\n检查完成!") 