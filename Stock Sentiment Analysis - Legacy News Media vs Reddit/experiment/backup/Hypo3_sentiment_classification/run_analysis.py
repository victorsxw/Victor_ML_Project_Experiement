#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情感分类假设分析运行脚本
初始化项目环境并执行分析流程
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行情感分类假设分析')
    
    parser.add_argument('--generate_data', action='store_true',
                        help='是否生成测试数据')
    
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                        help='数据开始日期 (YYYY-MM-DD)')
    
    parser.add_argument('--recent_only', action='store_true',
                        help='是否只分析2024年3月至今的数据')
    
    parser.add_argument('--time_window', type=str, 
                        choices=['all', 'short_term', 'medium_term', 'long_term'],
                        default='all',
                        help='分析时间窗口')
    
    return parser.parse_args()

def check_environment():
    """检查环境依赖"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scipy', 'scikit-learn', 'statsmodels'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请使用以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_project_structure():
    """设置项目目录结构"""
    # 项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 创建目录结构
    directories = [
        os.path.join(project_root, 'data', 'raw'),
        os.path.join(project_root, 'data', 'processed'),
        os.path.join(project_root, 'results', 'figures'),
        os.path.join(project_root, 'results', 'reports')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("项目目录结构已创建")
    
    return project_root

def generate_test_data(project_root, start_date):
    """生成测试数据"""
    print("\n正在生成测试数据...")
    
    # 构建命令
    cmd = [
        sys.executable,
        os.path.join(project_root, 'src', 'generate_test_data.py'),
        '--start_date', start_date
    ]
    
    # 执行命令
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode == 0:
        print(process.stdout)
        
        # 提取数据文件路径
        lines = process.stdout.strip().split('\n')
        paths = {}
        
        for line in lines:
            if "股票数据已保存至" in line:
                paths['stock_path'] = line.split(': ')[1]
            elif "新闻数据已保存至" in line:
                paths['news_path'] = line.split(': ')[1]
            elif "Reddit数据已保存至" in line:
                paths['reddit_path'] = line.split(': ')[1]
                
        return paths
    else:
        print("生成测试数据失败:")
        print(process.stderr)
        return None

def run_analysis(project_root, data_paths, recent_only=False, time_window='all'):
    """运行分析"""
    print("\n正在运行分析...")
    
    # 构建命令
    cmd = [
        sys.executable,
        os.path.join(project_root, 'src', 'main.py'),
        '--news_path', data_paths['news_path'],
        '--reddit_path', data_paths['reddit_path'],
        '--stock_path', data_paths['stock_path'],
        '--time_window', time_window
    ]
    
    if recent_only:
        cmd.extend(['--start_date', '2024-03-01'])
    
    # 执行命令
    process = subprocess.run(cmd, capture_output=False, text=True)
    
    if process.returncode == 0:
        print("\n分析完成！")
        return True
    else:
        print("\n分析过程中出错")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("情感分类假设分析")
    print("=" * 60)
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查环境
    if not check_environment():
        return 1
    
    # 设置项目结构
    project_root = setup_project_structure()
    
    # 数据路径
    data_paths = None
    
    # 生成测试数据
    if args.generate_data:
        data_paths = generate_test_data(project_root, args.start_date)
        if data_paths is None:
            return 1
    else:
        # 使用默认路径
        data_dir = os.path.join(project_root, 'data', 'raw')
        data_paths = {
            'news_path': os.path.join(data_dir, 'news_data.csv'),
            'reddit_path': os.path.join(data_dir, 'reddit_data.csv'),
            'stock_path': os.path.join(data_dir, 'stock_data.csv')
        }
        
        # 检查文件是否存在
        missing_files = []
        for key, path in data_paths.items():
            if not os.path.exists(path):
                missing_files.append(path)
                
        if missing_files:
            print("以下数据文件不存在:")
            for path in missing_files:
                print(f"  - {path}")
            print("\n请先生成测试数据:")
            print(f"python {os.path.basename(__file__)} --generate_data")
            return 1
    
    # 运行分析
    success = run_analysis(
        project_root, 
        data_paths, 
        recent_only=args.recent_only,
        time_window=args.time_window
    )
    
    if success:
        # 显示结果位置
        results_dir = os.path.join(project_root, 'results')
        print(f"\n分析结果保存在: {results_dir}")
        print("- 报告: results/reports/")
        print("- 图表: results/figures/")
        
        # 显示主要报告文件
        report_path = os.path.join(results_dir, 'reports', 'hypothesis_test_report.md')
        if os.path.exists(report_path):
            print(f"\n假设检验报告: {report_path}")
            
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 