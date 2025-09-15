import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import os
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# Set style for all plots
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

class KeywordImpactAnalyzer:
    """Analyzer for studying the impact of keywords on stock price movements"""
    
    def __init__(self):
        self.data = None
        self.keywords = {
            'market_sentiment': ['bull', 'bear', 'bullish', 'bearish'],
            'economic_indicators': ['inflation', 'unemployment', 'tariffs', 'recession', 'gdp'],
            'company_specific': ['earnings', 'revenue', 'profit', 'loss', 'guidance'],
            'market_events': ['crash', 'rally', 'correction', 'volatility'],
            'tech_specific': ['ai', 'autonomous', 'ev', 'battery', 'chip','deepseek']
        }
        self.output_dir = 'Hypo4_sentimen_keywords'
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['s', 't', 'm', 're', 'll', 'd', 've'])  # 添加额外的停用词
        
    def load_and_process_data(self):
        """Load and process the data"""
        print("Loading and processing data...")
        
        # Load news data
        print("\nLoading news data...")
        try:
            news_data = pd.read_pickle('dataset/news_sentiment.pkl')
            print(f"News data shape: {news_data.shape}")
            print(f"News data columns: {news_data.columns.tolist()}")
            print("\nNews data sample (first 3 rows):")
            print(news_data[['datetime', 'headline']].head(3))
        except Exception as e:
            print(f"Error loading news data: {str(e)}")
            exit(1)
        
        # Load Reddit data
        print("\nLoading Reddit data...")
        try:
            reddit_data = pd.read_pickle('dataset/reddit_sentiment.pkl')
            print(f"Reddit data shape: {reddit_data.shape}")
            print(f"Reddit data columns: {reddit_data.columns.tolist()}")
            print("\nReddit data sample (first 3 rows):")
            print(reddit_data[['datetime', 'title']].head(3))
        except Exception as e:
            print(f"Error loading Reddit data: {str(e)}")
            exit(1)
        
        # Load stock price data
        print("\nLoading stock price data...")
        try:
            stock_data = pd.read_pickle('dataset/6datasets-2024-2025/spy_compare.pkl')
            print(f"Stock data shape: {stock_data.shape}")
            print(f"Stock data columns: {stock_data.columns.tolist()}")
            print("Stock data first few rows:")
            print(stock_data.head())
        except Exception as e:
            print(f"Error loading stock data: {str(e)}")
            exit(1)
        
        # Process news data
        print("\nProcessing news data...")
        news_data['datetime'] = pd.to_datetime(news_data['datetime'])
        if news_data['datetime'].dt.tz is not None:
            news_data['datetime'] = news_data['datetime'].dt.tz_convert(None)
        news_data = news_data[news_data['datetime'] >= pd.Timestamp('2024-03-01')]
        print(f"News data after filtering: {len(news_data)} records")
        
        # Process Reddit data
        print("\nProcessing Reddit data...")
        reddit_data['datetime'] = pd.to_datetime(reddit_data['datetime'])
        if reddit_data['datetime'].dt.tz is not None:
            reddit_data['datetime'] = reddit_data['datetime'].dt.tz_convert(None)
        reddit_data = reddit_data[reddit_data['datetime'] >= pd.Timestamp('2024-03-01')]
        print(f"Reddit data after filtering: {len(reddit_data)} records")
        
        # Process stock data
        print("\nProcessing stock data...")
        stock_data = stock_data.copy()
        if stock_data.index.tz is None:
            stock_data.index = stock_data.index.tz_localize('UTC')
        stock_data.index = stock_data.index.tz_convert(None)
        stock_data = stock_data[stock_data.index >= pd.Timestamp('2024-03-01')]
        print(f"Stock data after filtering: {len(stock_data)} records")
        
        # Aggregate texts by date
        print("\nAggregating texts by date...")
        news_daily = news_data.groupby(news_data['datetime'].dt.date).agg({
            'headline': lambda x: ' '.join(set(x))  # 使用set去除可能的重复标题
        }).reset_index()
        news_daily['datetime'] = pd.to_datetime(news_daily['datetime'])
        
        reddit_daily = reddit_data.groupby(reddit_data['datetime'].dt.date).agg({
            'title': lambda x: ' '.join(set(x))  # 使用set去除可能的重复标题
        }).reset_index()
        reddit_daily['datetime'] = pd.to_datetime(reddit_daily['datetime'])
        
        # Process text data
        print("\nProcessing text data...")
        news_daily.rename(columns={'headline': 'full_text_news'}, inplace=True)
        reddit_daily.rename(columns={'title': 'full_text_reddit'}, inplace=True)
        
        # Merge datasets
        print("\nMerging datasets...")
        merged_data = pd.DataFrame(index=stock_data.index)
        merged_data['stock'] = stock_data['stock']
        merged_data = merged_data.reset_index()
        merged_data.rename(columns={'index': 'date'}, inplace=True)
        
        # 分别合并新闻和Reddit数据
        merged_data = pd.merge(merged_data, news_daily[['datetime', 'full_text_news']], 
                             left_on='date', right_on='datetime', how='left')
        merged_data = pd.merge(merged_data, reddit_daily[['datetime', 'full_text_reddit']], 
                             left_on='date', right_on='datetime', how='left')
        
        # 删除多余的datetime列
        merged_data.drop(['datetime_x', 'datetime_y', 'datetime'], axis=1, errors='ignore', inplace=True)
        
        # 数据验证
        print("\nData validation:")
        print("Total days in merged data:", len(merged_data))
        print("Days with news data:", merged_data['full_text_news'].notna().sum())
        print("Days with Reddit data:", merged_data['full_text_reddit'].notna().sum())
        print("\nSample comparison of news and Reddit texts:")
        sample_idx = merged_data.index[0]
        print(f"\nDay {merged_data['date'][sample_idx].strftime('%Y-%m-%d')}:")
        print("News headline:", merged_data['full_text_news'][sample_idx][:100] + "...")
        print("Reddit title:", merged_data['full_text_reddit'][sample_idx][:100] + "...")
        
        # Calculate daily returns
        print("\nCalculating daily returns...")
        merged_data['stock_return'] = merged_data['stock'].pct_change()
        
        # Handle missing values
        print("\nHandling missing values...")
        initial_rows = len(merged_data)
        merged_data = merged_data.dropna()
        removed_rows = initial_rows - len(merged_data)
        print(f"Removed {removed_rows} rows with missing values")
        
        print(f"\nFinal merged data shape: {merged_data.shape}")
        print(f"Date range: {merged_data['date'].min()} to {merged_data['date'].max()}\n")
        
        self.data = merged_data

    def calculate_keyword_frequencies(self, text, keyword_group):
        """Calculate frequency of keywords in text"""
        text = str(text).lower()
        frequencies = {}
        for keyword in self.keywords[keyword_group]:
            # Use word boundaries to match whole words only
            pattern = r'\b' + keyword + r'\b'
            frequencies[keyword] = len(re.findall(pattern, text))
        return frequencies
    
    def analyze_keyword_impact(self):
        """Analyze the impact of keywords on stock price movements"""
        results = {}
        
        # Analyze each keyword group
        for group_name in self.keywords.keys():
            print(f"\nAnalyzing {group_name} keywords...")
            
            # Calculate keyword frequencies for news and Reddit
            news_frequencies = pd.DataFrame(index=self.data.index)
            reddit_frequencies = pd.DataFrame(index=self.data.index)
            
            for keyword in self.keywords[group_name]:
                # Calculate frequencies for news
                news_frequencies[keyword] = self.data['full_text_news'].fillna('').astype(str).apply(
                    lambda x: len(re.findall(r'\b' + re.escape(keyword) + r'\b', x.lower()))
                )
                
                # Calculate frequencies for Reddit
                reddit_frequencies[keyword] = self.data['full_text_reddit'].fillna('').astype(str).apply(
                    lambda x: len(re.findall(r'\b' + re.escape(keyword) + r'\b', x.lower()))
                )
            
            # Calculate correlations with stock returns
            news_correlations = {}
            reddit_correlations = {}
            
            # Prepare stock returns data
            stock_returns = self.data['stock_return'].shift(-1)[:-1].fillna(0)
            
            for keyword in self.keywords[group_name]:
                # Calculate correlations with next day's stock returns for news
                news_series = news_frequencies[keyword][:-1]
                news_corr = stats.pearsonr(news_series.values, stock_returns.values)
                
                # Calculate correlations with next day's stock returns for Reddit
                reddit_series = reddit_frequencies[keyword][:-1]
                reddit_corr = stats.pearsonr(reddit_series.values, stock_returns.values)
                
                # Print debug information
                print(f"\nKeyword: {keyword}")
                print(f"News correlation: {news_corr[0]:.4f} (p-value: {news_corr[1]:.4f})")
                print(f"Reddit correlation: {reddit_corr[0]:.4f} (p-value: {reddit_corr[1]:.4f})")
                print(f"News frequencies: min={news_series.min()}, max={news_series.max()}, mean={news_series.mean():.2f}")
                print(f"Reddit frequencies: min={reddit_series.min()}, max={reddit_series.max()}, mean={reddit_series.mean():.2f}")
                
                news_correlations[keyword] = {
                    'correlation': news_corr[0],
                    'p_value': news_corr[1],
                    'frequency_stats': {
                        'min': news_series.min(),
                        'max': news_series.max(),
                        'mean': news_series.mean()
                    }
                }
                reddit_correlations[keyword] = {
                    'correlation': reddit_corr[0],
                    'p_value': reddit_corr[1],
                    'frequency_stats': {
                        'min': reddit_series.min(),
                        'max': reddit_series.max(),
                        'mean': reddit_series.mean()
                    }
                }
            
            # Store results
            results[group_name] = {
                'news_frequencies': news_frequencies,
                'reddit_frequencies': reddit_frequencies,
                'news_correlations': news_correlations,
                'reddit_correlations': reddit_correlations
            }
            
            # Generate visualizations
            self.plot_keyword_analysis(
                news_frequencies,
                reddit_frequencies,
                news_correlations,
                reddit_correlations,
                group_name
            )
        
        # Generate summary report
        self.generate_summary_report(results)
        return results
    
    def plot_keyword_analysis(self, news_freq, reddit_freq, news_corr, reddit_corr, group_name):
        """Generate visualizations for keyword analysis"""
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # First subplot: Keyword frequencies over time
        ax1 = plt.subplot(211)
        
        # Plot frequencies
        for keyword in news_freq.columns:
            ax1.plot(self.data['date'], news_freq[keyword], 
                    label=f'News - {keyword}', linestyle='-', alpha=0.7)
            ax1.plot(self.data['date'], reddit_freq[keyword],
                    label=f'Reddit - {keyword}', linestyle='--', alpha=0.7)
        
        ax1.set_title(f'Keyword Frequencies Over Time - {group_name}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Frequency')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Second subplot: Correlation heatmap
        ax2 = plt.subplot(212)
        
        # Prepare correlation data
        corr_data = []
        keywords = list(news_corr.keys())
        
        for keyword in keywords:
            corr_data.append([
                news_corr[keyword]['correlation'],
                reddit_corr[keyword]['correlation']
            ])
        
        corr_df = pd.DataFrame(
            corr_data,
            index=keywords,
            columns=['News', 'Reddit']
        )
        
        # Create heatmap
        sns.heatmap(corr_df, annot=True, cmap='RdYlBu', center=0,
                    vmin=-1, vmax=1, ax=ax2)
        ax2.set_title(f'Keyword-Stock Return Correlations - {group_name}')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'keyword_analysis_{group_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, results):
        """Generate comprehensive analysis report"""
        report_path = os.path.join(self.output_dir, 'keyword_impact_summary.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Keyword Impact Analysis Summary Report\n\n")
            
            # Overall statistics
            f.write("## 1. Overall Statistics\n\n")
            f.write(f"- Analysis Period: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}\n")
            f.write(f"- Total Days Analyzed: {len(self.data)}\n")
            f.write("- Keyword Groups Analyzed: " + ", ".join(self.keywords.keys()) + "\n\n")
            
            # Results by keyword group
            f.write("## 2. Results by Keyword Group\n\n")
            
            for group_name, group_results in results.items():
                f.write(f"### 2.{list(results.keys()).index(group_name) + 1}. {group_name}\n\n")
                
                # Create correlation comparison table
                f.write("#### Correlation Analysis\n\n")
                f.write("| Keyword | News Correlation | News P-value | Reddit Correlation | Reddit P-value |\n")
                f.write("|---------|-----------------|--------------|-------------------|----------------|\n")
                
                for keyword in self.keywords[group_name]:
                    news_corr = group_results['news_correlations'][keyword]
                    reddit_corr = group_results['reddit_correlations'][keyword]
                    
                    f.write(f"| {keyword} | {news_corr['correlation']:.4f} | {news_corr['p_value']:.4f} | ")
                    f.write(f"{reddit_corr['correlation']:.4f} | {reddit_corr['p_value']:.4f} |\n")
                
                # Calculate significant correlations
                significant_news = sum(1 for k in self.keywords[group_name]
                                    if group_results['news_correlations'][k]['p_value'] < 0.05)
                significant_reddit = sum(1 for k in self.keywords[group_name]
                                      if group_results['reddit_correlations'][k]['p_value'] < 0.05)
                
                f.write(f"\nSignificant correlations (p < 0.05):\n")
                f.write(f"- News: {significant_news} keywords\n")
                f.write(f"- Reddit: {significant_reddit} keywords\n\n")
            
            # Overall conclusions
            f.write("## 3. Overall Conclusions\n\n")
            
            # Calculate total significant correlations
            total_significant_news = sum(
                sum(1 for k in group
                    if results[group_name]['news_correlations'][k]['p_value'] < 0.05)
                for group_name, group in self.keywords.items()
            )
            total_significant_reddit = sum(
                sum(1 for k in group
                    if results[group_name]['reddit_correlations'][k]['p_value'] < 0.05)
                for group_name, group in self.keywords.items()
            )
            
            total_keywords = sum(len(group) for group in self.keywords.values())
            
            f.write("### 3.1 Hypothesis Testing\n\n")
            if total_significant_news > 0 or total_significant_reddit > 0:
                f.write("The null hypothesis (H0) is rejected:\n")
                f.write("- Significant correlations were found between keyword frequencies and stock price movements\n")
                f.write(f"- {total_significant_news}/{total_keywords} news keywords showed significant correlation\n")
                f.write(f"- {total_significant_reddit}/{total_keywords} Reddit keywords showed significant correlation\n\n")
            else:
                f.write("Failed to reject the null hypothesis (H0):\n")
                f.write("- No significant correlations were found between keyword frequencies and stock price movements\n\n")
            
            # Investment implications
            f.write("## 4. Investment Implications\n\n")
            f.write("Based on the analysis results, we recommend:\n\n")
            
            # Add specific recommendations based on the strongest correlations
            strong_correlations = []
            for group_name, group_results in results.items():
                for keyword in self.keywords[group_name]:
                    news_corr = abs(group_results['news_correlations'][keyword]['correlation'])
                    reddit_corr = abs(group_results['reddit_correlations'][keyword]['correlation'])
                    
                    if news_corr > 0.3 or reddit_corr > 0.3:
                        strong_correlations.append({
                            'keyword': keyword,
                            'group': group_name,
                            'news_corr': news_corr,
                            'reddit_corr': reddit_corr
                        })
            
            strong_correlations.sort(key=lambda x: max(x['news_corr'], x['reddit_corr']), reverse=True)
            
            for i, corr in enumerate(strong_correlations[:3], 1):
                source = "news media" if corr['news_corr'] > corr['reddit_corr'] else "Reddit"
                f.write(f"{i}. Monitor the frequency of '{corr['keyword']}' in {source} ")
                f.write(f"(correlation: {max(corr['news_corr'], corr['reddit_corr']):.4f})\n")
            
        print(f"Summary report saved to: {report_path}")

    def generate_wordcloud(self, text_data, title, output_filename):
        """生成词云图"""
        # 文本预处理
        text = ' '.join(text_data.fillna('').astype(str))
        
        # 分词和停用词过滤
        words = text.lower().split()
        words = [word for word in words if word.isalnum() and word not in self.stop_words 
                and len(word) > 1 and not word.isnumeric()]  # 添加额外的过滤条件
        
        # 统计词频
        word_freq = Counter(words)
        
        # 生成词云
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        
        # 绘制词云图
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, pad=20)
        
        # 保存图片
        plt.savefig(os.path.join(self.output_dir, output_filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 返回TOP10词频
        return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def analyze_top_words_impact(self, word_freq_news, word_freq_reddit):
        """分析TOP10词对股价的影响"""
        results = {
            'news': {},
            'reddit': {}
        }
        
        # 准备股票收益率数据
        stock_returns = self.data['stock_return'].shift(-1)[:-1].fillna(0)
        
        # 分析新闻TOP10词的影响
        for word, freq in word_freq_news.items():
            word_freq = self.data['full_text_news'].fillna('').astype(str).apply(
                lambda x: len(re.findall(r'\b' + re.escape(word) + r'\b', x.lower()))
            )[:-1]
            
            correlation = stats.pearsonr(word_freq.values, stock_returns.values)
            results['news'][word] = {
                'frequency': freq,
                'correlation': correlation[0],
                'p_value': correlation[1]
            }
        
        # 分析Reddit TOP10词的影响
        for word, freq in word_freq_reddit.items():
            word_freq = self.data['full_text_reddit'].fillna('').astype(str).apply(
                lambda x: len(re.findall(r'\b' + re.escape(word) + r'\b', x.lower()))
            )[:-1]
            
            correlation = stats.pearsonr(word_freq.values, stock_returns.values)
            results['reddit'][word] = {
                'frequency': freq,
                'correlation': correlation[0],
                'p_value': correlation[1]
            }
        
        return results
    
    def generate_top_words_report(self, top_words_results):
        """生成TOP10词影响分析报告"""
        report_path = os.path.join(self.output_dir, 'top_words_impact_summary.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# TOP 10 Words Impact Analysis Report\n\n")
            
            # 新闻TOP10词分析
            f.write("## 1. News TOP 10 Words Analysis\n\n")
            f.write("| Word | Frequency | Correlation | P-value |\n")
            f.write("|------|-----------|-------------|----------|\n")
            
            for word, stats in sorted(top_words_results['news'].items(), 
                                    key=lambda x: abs(x[1]['correlation']), 
                                    reverse=True):
                f.write(f"| {word} | {stats['frequency']} | {stats['correlation']:.4f} | {stats['p_value']:.4f} |\n")
            
            # Reddit TOP10词分析
            f.write("\n## 2. Reddit TOP 10 Words Analysis\n\n")
            f.write("| Word | Frequency | Correlation | P-value |\n")
            f.write("|------|-----------|-------------|----------|\n")
            
            for word, stats in sorted(top_words_results['reddit'].items(), 
                                    key=lambda x: abs(x[1]['correlation']), 
                                    reverse=True):
                f.write(f"| {word} | {stats['frequency']} | {stats['correlation']:.4f} | {stats['p_value']:.4f} |\n")
            
            # 显著相关性分析
            f.write("\n## 3. Significant Correlations (p < 0.05)\n\n")
            
            significant_news = [(w, s) for w, s in top_words_results['news'].items() 
                              if s['p_value'] < 0.05]
            significant_reddit = [(w, s) for w, s in top_words_results['reddit'].items() 
                                if s['p_value'] < 0.05]
            
            if significant_news:
                f.write("### News Words:\n")
                for word, stats in sorted(significant_news, key=lambda x: abs(x[1]['correlation']), 
                                        reverse=True):
                    f.write(f"- {word}: correlation = {stats['correlation']:.4f} (p = {stats['p_value']:.4f})\n")
            else:
                f.write("### News Words:\n- No significant correlations found\n")
            
            if significant_reddit:
                f.write("\n### Reddit Words:\n")
                for word, stats in sorted(significant_reddit, key=lambda x: abs(x[1]['correlation']), 
                                        reverse=True):
                    f.write(f"- {word}: correlation = {stats['correlation']:.4f} (p = {stats['p_value']:.4f})\n")
            else:
                f.write("\n### Reddit Words:\n- No significant correlations found\n")
        
        print(f"Top words impact report saved to: {report_path}")

    def analyze_recent_data(self, start_date='2024-03-01'):
        """分析指定开始日期至今的数据"""
        print(f"\n分析 {start_date} 至今的数据...")
        
        # 保存原始输出目录
        original_output_dir = self.output_dir
        # 修改输出目录，添加年份前缀
        self.output_dir = f'Hypo4_sentimen_keywords/2024_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            # 加载和处理数据
            self.load_and_process_data()
            
            # 过滤指定日期范围的数据
            self.data = self.data[self.data['date'] >= pd.Timestamp(start_date)]
            print(f"\n过滤后的数据范围：{self.data['date'].min()} 到 {self.data['date'].max()}")
            print(f"数据记录数：{len(self.data)}")
            
            # 生成词云和分析TOP词影响
            print("\n生成词云和分析TOP词影响...")
            word_freq_news = self.generate_wordcloud(
                self.data['full_text_news'],
                '2024新闻词云',
                '2024_news_wordcloud.png'
            )
            word_freq_reddit = self.generate_wordcloud(
                self.data['full_text_reddit'],
                '2024 Reddit词云',
                '2024_reddit_wordcloud.png'
            )
            
            # 分析TOP词的影响
            top_words_results = self.analyze_top_words_impact(word_freq_news, word_freq_reddit)
            self.generate_top_words_report(top_words_results)
            
            # 分析关键词影响
            results = self.analyze_keyword_impact()
            
            print(f"\n分析完成！结果已保存到 {self.output_dir} 目录")
            return results
            
        finally:
            # 恢复原始输出目录
            self.output_dir = original_output_dir

def main():
    # Create output directory if it doesn't exist
    output_dir = 'Hypo4_sentimen_keywords'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = KeywordImpactAnalyzer()
    
    # 分析2024年3月至今的数据
    print("\n分析2024年3月至今的数据...")
    analyzer.analyze_recent_data('2024-03-01')
    
    # 分析所有数据
    print("\n分析所有历史数据...")
    analyzer.load_and_process_data()
    
    # 生成词云和分析TOP词影响
    print("\n生成词云和分析TOP词影响...")
    word_freq_news = analyzer.generate_wordcloud(
        analyzer.data['full_text_news'],
        '新闻词云',
        'news_wordcloud.png'
    )
    word_freq_reddit = analyzer.generate_wordcloud(
        analyzer.data['full_text_reddit'],
        'Reddit词云',
        'reddit_wordcloud.png'
    )
    
    # 分析TOP词的影响
    top_words_results = analyzer.analyze_top_words_impact(word_freq_news, word_freq_reddit)
    analyzer.generate_top_words_report(top_words_results)
    
    # 分析关键词影响
    analyzer.analyze_keyword_impact()

if __name__ == '__main__':
    main() 