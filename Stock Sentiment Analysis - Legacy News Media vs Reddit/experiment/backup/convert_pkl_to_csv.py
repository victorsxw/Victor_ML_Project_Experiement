import pandas as pd
import os
import sys
from pathlib import Path

# 设置要处理的数据目录
data_dir = 'dataset/20250315'

# 检查目录是否存在
if not os.path.exists(data_dir):
    print(f"错误：目录 {data_dir} 不存在")
    sys.exit(1)

# 列出所有PKL文件
pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and os.path.isfile(os.path.join(data_dir, f))]

print(f"在 {data_dir} 目录中找到 {len(pkl_files)} 个PKL文件")

# 转换每个PKL文件为CSV
for file_name in pkl_files:
    pkl_path = os.path.join(data_dir, file_name)
    csv_path = os.path.join(data_dir, file_name.replace('.pkl', '.csv'))
    overview_path = os.path.join(data_dir, file_name.replace('.pkl', '_overview.txt'))
    
    print(f"正在转换: {pkl_path} -> {csv_path}")
    
    try:
        # 读取PKL文件
        df = pd.read_pickle(pkl_path)
        
        if isinstance(df, pd.DataFrame):
            # 保存为CSV，包含索引
            df.to_csv(csv_path, index=True)  # 将index设置为True以包含索引
            print(f"  成功: 已保存CSV文件 ({df.shape[0]} 行, {df.shape[1]} 列)")
            
            # 生成概述文件
            with open(overview_path, 'w', encoding='utf-8') as f:
                f.write(f"{file_name} 数据集概览：\n")
                f.write(f"行数：{df.shape[0]}\n")
                f.write(f"列数：{df.shape[1]}\n\n")
                
                f.write("数据类型：\n")
                for col in df.columns:
                    f.write(f"{col}: {df[col].dtype}\n")
                
                f.write("\n缺失值统计：\n")
                missing = df.isnull().sum()
                for col in df.columns:
                    f.write(f"{col}: {missing[col]}\n")
                
                f.write("\n唯一值统计：\n")
                unique_counts = df.nunique()
                for col in df.columns:
                    f.write(f"{col}: {unique_counts[col]}\n")
                
            print(f"  概述文件已保存: {overview_path}")
        else:
            print(f"  错误: {file_name} 不是pandas DataFrame")
    except Exception as e:
        print(f"  错误: 处理 {file_name} 时出错: {str(e)}")

print("\n转换完成！")
print(f"已将 {len(pkl_files)} 个PKL文件转换为CSV格式并保存在 {data_dir} 目录中") 