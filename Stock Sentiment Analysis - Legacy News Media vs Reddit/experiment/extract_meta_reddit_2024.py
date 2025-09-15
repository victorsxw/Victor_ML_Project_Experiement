import pandas as pd
import os
from datetime import datetime
import sys

def extract_meta_reddit_data():
    """
    从reddit_chat7.pkl文件中提取META相关的数据，
    筛选2024年3月1日以后的数据，并保存到meta_chat7_reddit.pkl
    """
    print("开始提取META相关的Reddit数据...")
    
    # 加载Reddit数据
    try:
        reddit_data_path = 'dataset/20250315/reddit_chat7.pkl'
        print(f"正在加载 {reddit_data_path}...")
        
        # 检查文件是否存在
        if not os.path.exists(reddit_data_path):
            print(f"错误: 文件 {reddit_data_path} 不存在")
            return
            
        # 读取pickle文件
        reddit_df = pd.read_pickle(reddit_data_path)
        print(f"成功加载Reddit数据，共 {len(reddit_df)} 条记录")
        print(f"数据结构: {type(reddit_df)}")
        
        # 检查是否是MultiIndex
        if isinstance(reddit_df.index, pd.MultiIndex):
            print("数据使用MultiIndex索引，索引级别:", reddit_df.index.names)
            
            # 将索引级别重置为列
            reddit_df = reddit_df.reset_index()
            print("已将索引转换为列，现在的列:", reddit_df.columns.tolist())
        
        # 检查数据格式
        print("数据列名:", reddit_df.columns.tolist())
        print("数据前两行:\n", reddit_df.head(2).to_string())
        
    except Exception as e:
        print(f"加载Reddit数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # 筛选META相关的数据
    try:
        # 确保tic列存在
        if 'tic' not in reddit_df.columns:
            print("错误: 找不到'tic'列")
            return None
        
        # 筛选META相关数据
        meta_df = reddit_df[reddit_df['tic'].astype(str).str.upper() == 'META']
        print(f"筛选出 {len(meta_df)} 条META相关记录")
        
        # 确保datetime列存在
        if 'datetime' not in meta_df.columns:
            print("错误: 找不到'datetime'列")
            return None
        
        # 将datetime列转换为datetime类型，并处理时区
        meta_df['datetime'] = pd.to_datetime(meta_df['datetime'])
        
        # 检查datetime列是否有时区信息
        has_timezone = meta_df['datetime'].dt.tz is not None
        print(f"datetime列是否有时区信息: {has_timezone}")
        
        if has_timezone:
            # 转换为无时区的日期时间
            print("正在将datetime转换为无时区格式...")
            meta_df['datetime'] = meta_df['datetime'].dt.tz_localize(None)
        
        # 获取完整时间范围
        earliest_date = meta_df['datetime'].min()
        latest_date = meta_df['datetime'].max()
        print(f"META数据的时间范围: {earliest_date} 到 {latest_date}")
        
        # 统计每年的数据量
        meta_df['year'] = meta_df['datetime'].dt.year
        yearly_counts = meta_df['year'].value_counts().sort_index()
        print("每年的数据量:")
        for year, count in yearly_counts.items():
            print(f"  {year}: {count}条")
        
        # 筛选2024年3月1日以后的数据
        start_date = pd.Timestamp('2024-03-01')
        recent_meta_df = meta_df[meta_df['datetime'] >= start_date]
        print(f"筛选出 {len(recent_meta_df)} 条2024年3月以后的META相关记录")
        
        if len(recent_meta_df) == 0:
            print("注意: 没有找到2024年3月以后的META数据")
            print("将使用最新的50条META记录代替")
            recent_meta_df = meta_df.sort_values('datetime', ascending=False).head(50)
            print(f"使用最新的 {len(recent_meta_df)} 条记录，时间范围: {recent_meta_df['datetime'].min()} 到 {recent_meta_df['datetime'].max()}")
        
        # 删除临时年份列
        if 'year' in recent_meta_df.columns:
            recent_meta_df = recent_meta_df.drop('year', axis=1)
        
        # 选择需要的列
        columns_to_keep = ['date', 'tic', 'datetime', 'headline', 'question', 'answer']
        available_columns = [col for col in columns_to_keep if col in recent_meta_df.columns]
        
        # 如果缺少列，输出警告
        missing_columns = set(columns_to_keep) - set(available_columns)
        if missing_columns:
            print(f"警告: 缺少以下列: {missing_columns}")
        
        # 使用可用的列
        selected_df = recent_meta_df[available_columns]
        print(f"选择了以下列: {available_columns}")
        
        # 保存结果
        output_pkl = 'meta_chat7_reddit.pkl'
        output_csv = 'meta_chat7_reddit.csv'
        
        selected_df.to_pickle(output_pkl)
        selected_df.to_csv(output_csv, index=False)
        
        print(f"已保存提取的数据到 {output_pkl} 和 {output_csv}")
        print(f"总共 {len(selected_df)} 条记录")
        
        # 打印数据样本
        print("\n数据样本:")
        print(selected_df.head(3).to_string())
        
        return selected_df
        
    except Exception as e:
        print(f"处理META数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(f"Python版本: {sys.version}")
    print(f"Pandas版本: {pd.__version__}")
    
    # 执行提取
    extract_meta_reddit_data()
    print("脚本执行完成") 