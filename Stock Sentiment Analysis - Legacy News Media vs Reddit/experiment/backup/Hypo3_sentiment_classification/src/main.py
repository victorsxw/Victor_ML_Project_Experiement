#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情感分类假设分析主程序
分析不同类型情感与股价变动的相关性
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from pathlib import Path
import scipy.stats as stats

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.sentiment_classifier import SentimentClassifier
from src.data_analyzer import DataAnalyzer
from src.config import PATHS, TIME_WINDOWS, CORRELATION_PARAMS

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='情感分类假设分析')
    
    parser.add_argument('--news_path', type=str, required=False,
                        default='../data/news_data.csv',
                        help='新闻数据文件路径')
    
    parser.add_argument('--reddit_path', type=str, required=False,
                        default='../data/reddit_data.csv',
                        help='Reddit数据文件路径')
    
    parser.add_argument('--stock_path', type=str, required=False,
                        default='../data/stock_data.csv',
                        help='股票价格数据文件路径')
    
    parser.add_argument('--start_date', type=str, required=False,
                        default='2024-03-01',
                        help='分析开始日期 (YYYY-MM-DD)')
    
    parser.add_argument('--time_window', type=str, required=False,
                        choices=['all', 'short_term', 'medium_term', 'long_term'],
                        default='all',
                        help='分析时间窗口')
    
    parser.add_argument('--output_dir', type=str, required=False,
                        default=None,
                        help='输出目录路径')
    
    return parser.parse_args()

def test_correlation_difference(correlations):
    """
    测试不同情感类别相关性之间的差异显著性
    
    Args:
        correlations: 相关性结果字典
        
    Returns:
        显著性检验结果字典
    """
    results = {}
    
    # 对每个数据源分别进行检验
    for source in correlations.keys():
        source_correlations = correlations[source]
        categories = ['fiscal', 'data_driven', 'opinion']
        
        # 获取相关系数
        corr_values = [source_correlations.get(cat, np.nan) for cat in categories]
        
        # 获取p值
        p_values = [source_correlations.get(f"{cat}_p_value", 1.0) for cat in categories]
        
        # 检查是否有显著的相关性
        significant_correlations = [i for i, p in enumerate(p_values) if p < 0.05]
        
        if len(significant_correlations) >= 2:
            # 至少有两个显著的相关性，可以比较它们的差异
            # 使用Fisher's z变换比较相关系数
            results[source] = {}
            
            for i in range(len(categories)):
                for j in range(i+1, len(categories)):
                    if i in significant_correlations and j in significant_correlations:
                        cat1, cat2 = categories[i], categories[j]
                        r1, r2 = corr_values[i], corr_values[j]
                        
                        # Fisher's z变换
                        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
                        z2 = 0.5 * np.log((1 + r2) / (1 - r2))
                        
                        # 假设样本量相同
                        n = 30  # 这里使用一个保守的估计，实际应该使用真实样本量
                        se = np.sqrt(1/(n-3) + 1/(n-3))
                        z = (z1 - z2) / se
                        
                        # 计算p值
                        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                        
                        results[source][f"{cat1}_vs_{cat2}"] = {
                            'z_score': z,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
        else:
            # 没有足够的显著相关性进行比较
            results[source] = {
                'message': '没有足够的显著相关性进行比较',
                'significant_correlations': [categories[i] for i in significant_correlations]
            }
            
    return results

def generate_hypothesis_report(correlations, diff_tests, output_path):
    """
    生成假设检验报告
    
    Args:
        correlations: 相关性结果字典
        diff_tests: 相关性差异检验结果
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 情感分类假设检验报告\n\n")
        
        f.write("## 研究假设\n\n")
        f.write("情感可以进一步分解为财务相关情感、数据驱动情感和观点驱动情感。每种类型的情感与股价变动可能有不同的相关性。\n\n")
        f.write("- **财务相关情感**：包括对盈利报告、政府政策和经济指标的讨论，可能具有更强的长期影响。\n")
        f.write("- **数据驱动情感**：基于定量分析和金融模型，可能与股价变动密切相关。\n")
        f.write("- **观点驱动情感**：更具投机性和情绪驱动，可能对短期波动性产生影响。\n\n")
        
        f.write("### 假设\n\n")
        f.write("- **H0（零假设）**：不同类型的情感与股价变动的相关性没有显著差异。\n")
        f.write("- **H1（备择假设）**：不同类型的情感与股价变动的相关性有显著差异。\n\n")
        
        f.write("## 相关性分析结果\n\n")
        
        # 写入相关性结果
        for source, source_correlations in correlations.items():
            f.write(f"### {source.capitalize()}情感与股价相关性\n\n")
            f.write("| 情感类别 | 相关系数 | P值 | 显著性 |\n")
            f.write("|---------|----------|-----|--------|\n")
            
            for category in ['fiscal', 'data_driven', 'opinion']:
                if category in source_correlations:
                    corr = source_correlations[category]
                    p_value = source_correlations.get(f"{category}_p_value", np.nan)
                    is_significant = "是" if p_value is not None and p_value < 0.05 else "否"
                    f.write(f"| {category} | {corr:.4f} | {p_value:.4f} | {is_significant} |\n")
            f.write("\n")
            
        # 写入相关性差异检验结果
        f.write("## 相关性差异检验\n\n")
        
        for source, source_results in diff_tests.items():
            f.write(f"### {source.capitalize()}情感相关性差异\n\n")
            
            if isinstance(source_results, dict) and 'message' in source_results:
                f.write(f"{source_results['message']}\n\n")
                if 'significant_correlations' in source_results and source_results['significant_correlations']:
                    f.write("显著的相关性：\n")
                    for cat in source_results['significant_correlations']:
                        f.write(f"- {cat}\n")
                f.write("\n")
            else:
                f.write("| 比较 | Z分数 | P值 | 显著差异 |\n")
                f.write("|------|-------|-----|----------|\n")
                
                for comparison, results in source_results.items():
                    cat1, cat2 = comparison.split('_vs_')
                    z_score = results['z_score']
                    p_value = results['p_value']
                    is_significant = "是" if results['significant'] else "否"
                    
                    f.write(f"| {cat1} vs {cat2} | {z_score:.4f} | {p_value:.4f} | {is_significant} |\n")
                f.write("\n")
                
        # 写入结论
        f.write("## 结论\n\n")
        
        # 检查是否有显著的相关性差异
        has_significant_diff = False
        for source, source_results in diff_tests.items():
            if isinstance(source_results, dict) and 'message' not in source_results:
                for comparison, results in source_results.items():
                    if results['significant']:
                        has_significant_diff = True
                        break
            if has_significant_diff:
                break
                
        if has_significant_diff:
            f.write("### 假设检验结果\n\n")
            f.write("- **拒绝零假设 (H0)**：不同类型的情感与股价变动的相关性存在显著差异。\n")
            f.write("- **接受备择假设 (H1)**：不同类型的情感对股价变动的预测能力不同。\n\n")
            
            f.write("### 发现\n\n")
            for source, source_results in diff_tests.items():
                if isinstance(source_results, dict) and 'message' not in source_results:
                    for comparison, results in source_results.items():
                        if results['significant']:
                            cat1, cat2 = comparison.split('_vs_')
                            f.write(f"- {source.capitalize()}中的{cat1}情感与{cat2}情感对股价变动的影响存在显著差异 (z = {results['z_score']:.4f}, p = {results['p_value']:.4f})。\n")
        else:
            f.write("### 假设检验结果\n\n")
            f.write("- **接受零假设 (H0)**：不同类型的情感与股价变动的相关性没有显著差异。\n")
            f.write("- **拒绝备择假设 (H1)**：没有足够证据表明不同类型的情感对股价变动有不同的预测能力。\n\n")
            
            f.write("### 可能的解释\n\n")
            f.write("1. 样本量可能不足以检测到差异。\n")
            f.write("2. 所选时间范围内市场可能受到其他因素的主导影响。\n")
            f.write("3. 情感分类方法可能需要进一步优化以更好地捕捉不同类型的情感。\n")
            
        f.write("\n### 建议\n\n")
        f.write("1. 扩大数据样本，特别是增加历史数据以捕捉不同市场环境下的情感影响。\n")
        f.write("2. 优化情感分类算法，可能引入机器学习方法提高分类准确性。\n")
        f.write("3. 考虑不同时间窗口的分析，以验证不同类型情感的短期和长期影响。\n")
        f.write("4. 结合其他市场指标进行多因素分析，以控制外部变量的影响。\n")
        
    print(f"假设检验报告已生成: {output_path}")

def visualize_correlation_comparison(correlations, output_path):
    """
    可视化不同情感类别相关性比较
    
    Args:
        correlations: 相关性结果字典
        output_path: 输出文件路径
    """
    # 准备数据
    categories = ['fiscal', 'data_driven', 'opinion']
    sources = list(correlations.keys())
    
    # 创建数据框
    data = []
    for source in sources:
        for category in categories:
            if category in correlations[source]:
                corr = correlations[source][category]
                p_value = correlations[source].get(f"{category}_p_value", 1.0)
                significance = '*' if p_value < 0.05 else ''
                
                data.append({
                    'source': source,
                    'category': category,
                    'correlation': corr,
                    'p_value': p_value,
                    'significance': significance
                })
                
    df = pd.DataFrame(data)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制条形图
    ax = sns.barplot(x='category', y='correlation', hue='source', data=df, palette='Set2')
    
    # 添加显著性标记
    for i, row in df.iterrows():
        if row['significance']:
            x = list(categories).index(row['category'])
            if row['source'] == sources[0]:
                x -= 0.2
            else:
                x += 0.2
            y = row['correlation']
            y_offset = 0.02 if y >= 0 else -0.08
            ax.text(x, y + y_offset, row['significance'], ha='center', fontweight='bold')
    
    # 设置图形属性
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('不同情感类别与股价变动的相关性比较', fontsize=16)
    plt.xlabel('情感类别', fontsize=14)
    plt.ylabel('相关系数', fontsize=14)
    plt.ylim(-0.5, 0.5)  # 调整y轴范围
    plt.legend(title='数据源')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"相关性比较可视化已保存至: {output_path}")

def visualize_sentiment_time_window_comparison(analyzer, output_path):
    """
    可视化不同时间窗口下的情感相关性
    
    Args:
        analyzer: 数据分析器实例
        output_path: 输出文件路径
    """
    # 计算不同时间窗口的相关性
    time_windows = ['short_term', 'medium_term', 'long_term', 'all']
    window_results = {}
    
    for window in time_windows:
        window_results[window] = analyzer.calculate_correlations(window)
    
    # 准备数据
    categories = ['fiscal', 'data_driven', 'opinion']
    sources = ['news', 'reddit']
    
    # 创建多个子图
    fig, axes = plt.subplots(len(sources), 1, figsize=(14, 10), sharex=True)
    
    for i, source in enumerate(sources):
        # 提取数据
        data = []
        for window in time_windows:
            if source in window_results[window]:
                for category in categories:
                    if category in window_results[window][source]:
                        corr = window_results[window][source][category]
                        p_value = window_results[window][source].get(f"{category}_p_value", 1.0)
                        significance = '*' if p_value < 0.05 else ''
                        
                        data.append({
                            'window': window,
                            'category': category,
                            'correlation': corr,
                            'p_value': p_value,
                            'significance': significance
                        })
        
        df = pd.DataFrame(data)
        
        # 绘制条形图
        sns.barplot(x='window', y='correlation', hue='category', data=df, ax=axes[i], palette='Set3')
        
        # 添加显著性标记
        for _, row in df.iterrows():
            if row['significance']:
                x = list(time_windows).index(row['window'])
                category_index = list(categories).index(row['category'])
                offset = (category_index - 1) * 0.2
                y = row['correlation']
                y_offset = 0.02 if y >= 0 else -0.08
                axes[i].text(x + offset, y + y_offset, row['significance'], ha='center', fontweight='bold')
        
        # 设置图形属性
        axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[i].set_title(f'{source.capitalize()}情感在不同时间窗口的相关性', fontsize=14)
        axes[i].set_ylabel('相关系数', fontsize=12)
        axes[i].set_ylim(-0.5, 0.5)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend(title='情感类别')
    
    # 设置x轴标签
    axes[-1].set_xlabel('时间窗口', fontsize=14)
    window_labels = {
        'short_term': f'短期 ({TIME_WINDOWS["short_term"]["days"]}天)',
        'medium_term': f'中期 ({TIME_WINDOWS["medium_term"]["days"]}天)',
        'long_term': f'长期 ({TIME_WINDOWS["long_term"]["days"]}天)',
        'all': '全部'
    }
    axes[-1].set_xticklabels([window_labels[w] for w in time_windows])
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"时间窗口比较可视化已保存至: {output_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建数据分析器
    analyzer = DataAnalyzer()
    
    print("=" * 50)
    print("情感分类假设分析")
    print("=" * 50)
    print(f"分析时间范围: {args.start_date} 至今")
    print(f"时间窗口: {args.time_window}")
    print("-" * 50)
    
    try:
        # 使用DataLoader加载数据
        analyzer.load_and_process_data()
        
        # 计算相关性
        correlations = analyzer.calculate_correlations(args.time_window)
        
        # 测试相关性差异
        diff_tests = test_correlation_difference(correlations)
        
        # 生成假设检验报告
        report_path = os.path.join(PATHS['results']['reports'], 'hypothesis_test_report.md')
        generate_hypothesis_report(correlations, diff_tests, report_path)
        
        # 可视化相关性比较
        viz_path = os.path.join(PATHS['results']['figures'], 'correlation_comparison.png')
        visualize_correlation_comparison(correlations, viz_path)
        
        # 可视化不同时间窗口比较
        time_window_viz_path = os.path.join(PATHS['results']['figures'], 'time_window_comparison.png')
        visualize_sentiment_time_window_comparison(analyzer, time_window_viz_path)
        
        print("\n分析完成！")
        print(f"报告保存在: {report_path}")
        print(f"可视化结果保存在: {PATHS['results']['figures']}")
        
    except Exception as e:
        print(f"分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 