import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from collections import Counter
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# 确保有必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def load_data(file_path):
    """加载META的数据并返回处理后的DataFrame"""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    print(f"Data loaded successfully with shape {df.shape}")
    return df

def extract_titles_and_content(news_data_path):
    """从原始新闻数据中提取META相关的新闻标题和内容"""
    try:
        df = pd.read_csv(news_data_path)
        # 筛选与META相关的新闻
        meta_news = df[df['title'].str.contains('META|Meta|Facebook|FB|Zuckerberg', case=False, na=False)]
        print(f"Extracted {len(meta_news)} META-related news articles")
        return meta_news
    except Exception as e:
        print(f"Error loading news data: {str(e)}")
        return pd.DataFrame()

def analyze_sentiment_distribution(df, news_col, reddit_col):
    """分析NEWS和Reddit情感分布，识别差异"""
    # 提取情感列
    news_sentiment = df[news_col]
    reddit_sentiment = df[reddit_col]
    
    # 计算基本统计量
    news_stats = news_sentiment.describe()
    reddit_stats = reddit_sentiment.describe()
    
    print("\nNEWS情感统计:")
    print(news_stats)
    print("\nReddit情感统计:")
    print(reddit_stats)
    
    # 绘制情感分布对比图
    plt.figure(figsize=(12, 6))
    sns.histplot(news_sentiment, color='blue', alpha=0.5, label='NEWS情感')
    sns.histplot(reddit_sentiment, color='red', alpha=0.5, label='Reddit情感')
    plt.title('META - NEWS vs Reddit情感分布对比')
    plt.xlabel('情感得分')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('meta_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 按日期绘制情感趋势
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, news_sentiment, label='NEWS情感', color='blue', alpha=0.7)
    plt.plot(df.index, reddit_sentiment, label='Reddit情感', color='red', alpha=0.7)
    plt.title('META - NEWS vs Reddit情感趋势')
    plt.xlabel('日期')
    plt.ylabel('情感得分')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('meta_sentiment_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return news_stats, reddit_stats

def identify_divergence_periods(df, news_col, reddit_col, window=5):
    """识别情感严重分歧的时间段"""
    # 计算情感分数之间的差异
    df['sentiment_diff'] = df[news_col] - df[reddit_col]
    
    # 计算极端分歧的时间点（差异最大的点）
    extreme_diff = df.sort_values('sentiment_diff', ascending=False).head(window)
    extreme_neg_diff = df.sort_values('sentiment_diff').head(window)
    
    print("\n情感最大分歧时间点(NEWS > Reddit):")
    print(extreme_diff[['sentiment_diff', news_col, reddit_col]])
    
    print("\n情感最大分歧时间点(Reddit > NEWS):")
    print(extreme_neg_diff[['sentiment_diff', news_col, reddit_col]])
    
    # 绘制情感差异图
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['sentiment_diff'], color='purple', alpha=0.8)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('META - NEWS vs Reddit情感差异')
    plt.xlabel('日期')
    plt.ylabel('情感差异 (NEWS - Reddit)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 标记极端分歧点
    for idx in extreme_diff.index:
        plt.scatter(idx, df.loc[idx, 'sentiment_diff'], color='red', s=50, zorder=5)
    for idx in extreme_neg_diff.index:
        plt.scatter(idx, df.loc[idx, 'sentiment_diff'], color='blue', s=50, zorder=5)
    
    plt.tight_layout()
    plt.savefig('meta_sentiment_divergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return extreme_diff, extreme_neg_diff

def analyze_news_content(news_data, dates_list):
    """分析指定日期的新闻内容"""
    if news_data.empty:
        print("No news data available for analysis")
        return
    
    # 转换日期格式
    news_data['date'] = pd.to_datetime(news_data['datetime']).dt.date
    dates_list = [pd.to_datetime(date).date() for date in dates_list]
    
    # 筛选指定日期的新闻
    target_news = news_data[news_data['date'].isin(dates_list)]
    
    if target_news.empty:
        print("No news found for the specified dates")
        return
    
    print(f"\n找到 {len(target_news)} 条在目标日期的新闻")
    
    # 分析新闻标题和内容
    all_titles = ' '.join(target_news['title'].fillna(''))
    all_content = ' '.join(target_news['content'].fillna(''))
    
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    
    # 处理标题
    word_tokens = word_tokenize(all_titles.lower())
    filtered_title_text = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    title_word_freq = Counter(filtered_title_text)
    
    # 处理内容
    word_tokens = word_tokenize(all_content.lower())
    filtered_content_text = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    content_word_freq = Counter(filtered_content_text)
    
    print("\n标题中最常见的词:")
    for word, count in title_word_freq.most_common(15):
        print(f"{word}: {count}")
    
    print("\n内容中最常见的词:")
    for word, count in content_word_freq.most_common(15):
        print(f"{word}: {count}")
    
    # 创建词云
    title_wordcloud = WordCloud(width=800, height=400, background_color='white',
                               max_words=100, contour_width=3, contour_color='steelblue')
    title_wordcloud.generate_from_frequencies(title_word_freq)
    
    content_wordcloud = WordCloud(width=800, height=400, background_color='white',
                                 max_words=100, contour_width=3, contour_color='steelblue')
    content_wordcloud.generate_from_frequencies(content_word_freq)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(title_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('META新闻标题词云')
    plt.tight_layout()
    plt.savefig('meta_title_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(content_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('META新闻内容词云')
    plt.tight_layout()
    plt.savefig('meta_content_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 返回部分新闻标题和摘要
    return target_news[['date', 'title', 'content']].head(10)

def compare_quarterly_sentiment(df, news_col, reddit_col):
    """按季度比较情感变化"""
    # 添加季度信息
    df['quarter'] = df.index.to_period('Q')
    
    # 按季度分组并计算平均情感
    quarterly_sentiment = df.groupby('quarter').agg({
        news_col: 'mean',
        reddit_col: 'mean'
    })
    
    # 绘制季度情感对比图
    plt.figure(figsize=(14, 7))
    
    x = range(len(quarterly_sentiment))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], quarterly_sentiment[news_col], width, label='NEWS情感', alpha=0.7, color='blue')
    plt.bar([i + width/2 for i in x], quarterly_sentiment[reddit_col], width, label='Reddit情感', alpha=0.7, color='red')
    
    plt.xlabel('季度')
    plt.ylabel('平均情感得分')
    plt.title('META - 季度NEWS vs Reddit情感对比')
    plt.xticks(x, [str(q) for q in quarterly_sentiment.index])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('meta_quarterly_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return quarterly_sentiment

def analyze_news_reddit_relationship(df, news_col, reddit_col):
    """分析NEWS和Reddit情感之间的关系"""
    # 计算不同情感类别的日期数
    df['news_pos'] = df[news_col] > 0
    df['news_neg'] = df[news_col] < 0
    df['reddit_pos'] = df[reddit_col] > 0
    df['reddit_neg'] = df[reddit_col] < 0
    
    # 计算一致和不一致的情况
    df['sentiment_match'] = ((df['news_pos'] & df['reddit_pos']) | (df['news_neg'] & df['reddit_neg']))
    df['sentiment_opposite'] = ((df['news_pos'] & df['reddit_neg']) | (df['news_neg'] & df['reddit_pos']))
    
    match_count = df['sentiment_match'].sum()
    opposite_count = df['sentiment_opposite'].sum()
    total = len(df)
    
    print(f"\n情感方向一致的天数: {match_count} ({match_count/total*100:.1f}%)")
    print(f"情感方向相反的天数: {opposite_count} ({opposite_count/total*100:.1f}%)")
    
    # 区分四种情况
    case1 = ((df['news_pos']) & (df['reddit_pos'])).sum()  # 都是正面
    case2 = ((df['news_neg']) & (df['reddit_neg'])).sum()  # 都是负面
    case3 = ((df['news_pos']) & (df['reddit_neg'])).sum()  # 新闻正面，Reddit负面
    case4 = ((df['news_neg']) & (df['reddit_pos'])).sum()  # 新闻负面，Reddit正面
    
    # 绘制饼图
    plt.figure(figsize=(10, 7))
    plt.pie([case1, case2, case3, case4], 
            labels=['NEWS+/Reddit+', 'NEWS-/Reddit-', 'NEWS+/Reddit-', 'NEWS-/Reddit+'],
            autopct='%1.1f%%',
            colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'],
            explode=(0, 0, 0.1, 0.1))
    plt.title('META - NEWS vs Reddit情感方向分布')
    plt.tight_layout()
    plt.savefig('meta_sentiment_direction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    
    # 分四类点绘制
    pos_pos = df[(df['news_pos']) & (df['reddit_pos'])]
    neg_neg = df[(df['news_neg']) & (df['reddit_neg'])]
    pos_neg = df[(df['news_pos']) & (df['reddit_neg'])]
    neg_pos = df[(df['news_neg']) & (df['reddit_pos'])]
    
    plt.scatter(pos_pos[news_col], pos_pos[reddit_col], color='green', alpha=0.6, label='NEWS+/Reddit+')
    plt.scatter(neg_neg[news_col], neg_neg[reddit_col], color='red', alpha=0.6, label='NEWS-/Reddit-')
    plt.scatter(pos_neg[news_col], pos_neg[reddit_col], color='blue', alpha=0.6, label='NEWS+/Reddit-')
    plt.scatter(neg_pos[news_col], neg_pos[reddit_col], color='orange', alpha=0.6, label='NEWS-/Reddit+')
    
    # 添加拟合线
    z = np.polyfit(df[news_col], df[reddit_col], 1)
    p = np.poly1d(z)
    plt.plot(df[news_col], p(df[news_col]), "r--", alpha=0.8)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('NEWS情感得分')
    plt.ylabel('Reddit情感得分')
    plt.title('META - NEWS vs Reddit情感散点图')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('meta_sentiment_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'case1': case1, 'case2': case2, 'case3': case3, 'case4': case4,
        'match_count': match_count, 'opposite_count': opposite_count
    }

def main():
    print("开始分析META的新闻和Reddit情感负相关现象")
    
    # 创建输出目录
    import os
    os.makedirs('meta_analysis_results', exist_ok=True)
    
    # 1. 加载数据
    meta_data = load_data('dataset/20250315/merged/meta_merged_data.csv')
    
    # 定义关键列名
    news_col = 'news_meta_EMA0.02_scaled'  # 与脚本匹配的列名
    reddit_col = 'reddit_meta_EMA0.02_scaled'
    
    # 2. 提取新闻标题和内容
    # 假设原始新闻数据在以下路径
    news_data = extract_titles_and_content('dataset/raw_news_data.csv')
    
    # 3. 分析情感分布
    news_stats, reddit_stats = analyze_sentiment_distribution(meta_data, news_col, reddit_col)
    
    # 4. 识别分歧最大的时间段
    extreme_diff, extreme_neg_diff = identify_divergence_periods(meta_data, news_col, reddit_col)
    
    # 5. 获取分歧最大时间段的日期列表
    divergence_dates = list(extreme_diff.index) + list(extreme_neg_diff.index)
    
    # 6. 分析相关时间段的新闻内容
    if not news_data.empty:
        news_examples = analyze_news_content(news_data, divergence_dates)
        if news_examples is not None:
            news_examples.to_csv('meta_analysis_results/divergence_news_examples.csv', index=False)
    
    # 7. 分析季度变化
    quarterly_sentiment = compare_quarterly_sentiment(meta_data, news_col, reddit_col)
    quarterly_sentiment.to_csv('meta_analysis_results/quarterly_sentiment.csv')
    
    # 8. 分析NEWS和Reddit关系
    relationship_stats = analyze_news_reddit_relationship(meta_data, news_col, reddit_col)
    
    # 9. 输出汇总报告
    with open('meta_analysis_results/META_sentiment_analysis_report.txt', 'w') as f:
        f.write("META新闻与Reddit情感负相关分析报告\n")
        f.write("===============================\n\n")
        
        f.write("1. 数据概览\n")
        f.write(f"   分析期间: {meta_data.index.min().strftime('%Y-%m-%d')} 到 {meta_data.index.max().strftime('%Y-%m-%d')}\n")
        f.write(f"   样本数量: {len(meta_data)}\n\n")
        
        f.write("2. 情感统计\n")
        f.write("   NEWS情感:\n")
        for stat, value in news_stats.items():
            f.write(f"     {stat}: {value:.4f}\n")
        
        f.write("\n   Reddit情感:\n")
        for stat, value in reddit_stats.items():
            f.write(f"     {stat}: {value:.4f}\n")
        
        f.write("\n3. 情感方向一致性分析\n")
        f.write(f"   情感方向一致的天数: {relationship_stats['match_count']} ({relationship_stats['match_count']/len(meta_data)*100:.1f}%)\n")
        f.write(f"   情感方向相反的天数: {relationship_stats['opposite_count']} ({relationship_stats['opposite_count']/len(meta_data)*100:.1f}%)\n\n")
        
        f.write("4. 情感方向细分:\n")
        f.write(f"   NEWS+/Reddit+: {relationship_stats['case1']} ({relationship_stats['case1']/len(meta_data)*100:.1f}%)\n")
        f.write(f"   NEWS-/Reddit-: {relationship_stats['case2']} ({relationship_stats['case2']/len(meta_data)*100:.1f}%)\n")
        f.write(f"   NEWS+/Reddit-: {relationship_stats['case3']} ({relationship_stats['case3']/len(meta_data)*100:.1f}%)\n")
        f.write(f"   NEWS-/Reddit+: {relationship_stats['case4']} ({relationship_stats['case4']/len(meta_data)*100:.1f}%)\n\n")
        
        f.write("5. 负相关可能原因分析\n")
        
        # 根据分析结果编写原因
        if relationship_stats['case3'] + relationship_stats['case4'] > relationship_stats['case1'] + relationship_stats['case2']:
            f.write("   a) 新闻媒体与Reddit用户对META的情感态度存在显著对立\n")
        
        f.write("   b) 新闻媒体对META的报道可能更关注公司战略、财务表现和长期发展\n")
        f.write("   c) Reddit用户可能更关注产品体验、用户隐私和短期市场反应\n")
        f.write("   d) 在重大事件发生时，Reddit反应可能更快且更极端，而新闻媒体报道相对滞后且更平衡\n")
        
        f.write("\n6. 下一步研究建议\n")
        f.write("   a) 对特定重大事件期间的情感变化进行深入案例研究\n")
        f.write("   b) 分析新闻标题与内容的情感差异，以及不同新闻来源的报道倾向\n")
        f.write("   c) 研究Reddit不同子版块对META的情感差异\n")
        f.write("   d) 比较META与其他科技公司的情感相关性模式，识别是否存在行业趋势\n")
    
    print("分析完成，结果保存在meta_analysis_results目录下")

if __name__ == "__main__":
    main() 