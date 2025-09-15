import pandas as pd
import os

# 设置文件路径
input_file = 'dataset/NVDA_with_category.pkl'
output_file = 'dataset/NVDA_with_category.csv'
overview_file = 'dataset/NVDA_with_category_overview.txt'

print(f"正在转换文件: {input_file} -> {output_file}")

try:
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
    else:
        # 读取PKL文件
        df = pd.read_pickle(input_file)
        
        if isinstance(df, pd.DataFrame):
            # 保存为CSV，包含索引
            df.to_csv(output_file, index=True)
            print(f"成功: 已保存CSV文件 ({df.shape[0]} 行, {df.shape[1]} 列)")
            
            # 生成概述文件
            with open(overview_file, 'w', encoding='utf-8') as f:
                f.write(f"NVDA_with_category 数据集概览：\n")
                f.write(f"行数：{df.shape[0]}\n")
                f.write(f"列数：{df.shape[1]}\n\n")
                
                # 打印索引信息
                f.write(f"索引名称: {df.index.name if df.index.name else '未命名'}\n")
                f.write(f"索引类型: {df.index.dtype}\n\n")
                
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
                
                # 前几行数据示例
                f.write("\n数据前5行示例：\n")
                f.write(str(df.head()))
                
            print(f"概述文件已保存: {overview_file}")
        else:
            print(f"错误: {input_file} 不是pandas DataFrame")
except Exception as e:
    print(f"错误: 处理文件时出错: {str(e)}")

print("\n转换完成！") 