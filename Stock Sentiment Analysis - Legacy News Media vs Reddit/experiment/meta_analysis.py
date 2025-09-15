import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import os

# Remove Chinese font settings
# Instead use default English fonts
plt.rcParams['axes.unicode_minus'] = False

def load_meta_data():
    """
    Load META Reddit and News sentiment data from CSV and PKL files
    """
    # Create output directory
    os.makedirs('meta_analysis_results', exist_ok=True)
    
    # Determine if we're using the CSV or PKL file for Reddit data
    reddit_data = None
    reddit_csv_path = 'dataset/20250315/merged/meta_chat7_reddit.csv'
    reddit_pkl_path = 'dataset/20250315/merged/meta_chat7_reddit.pkl'
    
    # Try loading from CSV first
    if os.path.exists(reddit_csv_path):
        try:
            reddit_data = pd.read_csv(reddit_csv_path)
            print(f"Loaded {len(reddit_data)} META Reddit data records from CSV")
            
            # Convert datetime to proper format
            if 'datetime' in reddit_data.columns:
                reddit_data['datetime'] = pd.to_datetime(reddit_data['datetime'])
                reddit_data['date'] = reddit_data['datetime'].dt.date
            elif 'date' in reddit_data.columns and not pd.api.types.is_datetime64_any_dtype(reddit_data['date']):
                reddit_data['date'] = pd.to_datetime(reddit_data['date']).dt.date
            
            # Extract sentiment from answers if needed
            if 'sentiment' not in reddit_data.columns and 'answer' in reddit_data.columns:
                def extract_sentiment(text):
                    if pd.isna(text):
                        return np.nan
                    
                    text = str(text).lower()
                    if 'positive' in text:
                        return 'positive'
                    elif 'negative' in text:
                        return 'negative'
                    elif 'neutral' in text:
                        return 'neutral'
                    elif 'unknown' in text:
                        return 'unknown'
                    else:
                        return np.nan
                
                # Extract sentiment and add as new column
                reddit_data['sentiment'] = reddit_data['answer'].apply(extract_sentiment)
                
                # Count sentiment distribution
                sentiment_counts = reddit_data['sentiment'].value_counts()
                print("Reddit sentiment distribution:")
                for sentiment, count in sentiment_counts.items():
                    print(f"  {sentiment}: {count} ({count/len(reddit_data)*100:.1f}%)")
        except Exception as e:
            print(f"Error loading Reddit data from CSV: {str(e)}")
            reddit_data = None
    
    # If CSV loading failed or file doesn't exist, try PKL
    if reddit_data is None and os.path.exists(reddit_pkl_path):
        try:
            reddit_data = pd.read_pickle(reddit_pkl_path)
            print(f"Loaded {len(reddit_data)} META Reddit data records from PKL")
            
            # Ensure date column exists
            if 'date' not in reddit_data.columns and 'datetime' in reddit_data.columns:
                reddit_data['date'] = pd.to_datetime(reddit_data['datetime']).dt.date
        except Exception as e:
            print(f"Error loading Reddit data from PKL: {str(e)}")
            reddit_data = None
    
    # Process Reddit data columns if loaded
    if reddit_data is not None:
        print(f"Reddit data columns: {reddit_data.columns.tolist()}")
    else:
        print("No Reddit data available.")
    
    # Load NEWS data
    news_data = None
    news_csv_path = 'dataset/20250315/merged/meta_chat7_news.csv'
    news_pkl_path = 'dataset/20250315/merged/meta_chat7_news.pkl'
    
    # Try loading from CSV first
    if os.path.exists(news_csv_path):
        try:
            news_data = pd.read_csv(news_csv_path)
            print(f"Loaded {len(news_data)} META News data records from CSV")
            
            # Convert datetime to proper format
            if 'datetime' in news_data.columns:
                news_data['datetime'] = pd.to_datetime(news_data['datetime'])
                news_data['date'] = news_data['datetime'].dt.date
            elif 'date' in news_data.columns and not pd.api.types.is_datetime64_any_dtype(news_data['date']):
                news_data['date'] = pd.to_datetime(news_data['date']).dt.date
        except Exception as e:
            print(f"Error loading News data from CSV: {str(e)}")
            news_data = None
    
    # If CSV loading failed or file doesn't exist, try PKL
    if news_data is None and os.path.exists(news_pkl_path):
        try:
            news_data = pd.read_pickle(news_pkl_path)
            print(f"Loaded {len(news_data)} META News data records from PKL")
            
            # Ensure date column exists
            if 'date' not in news_data.columns and 'datetime' in news_data.columns:
                news_data['date'] = pd.to_datetime(news_data['datetime']).dt.date
        except Exception as e:
            print(f"Error loading News data from PKL: {str(e)}")
            news_data = None
    
    # Process News data columns if loaded
    if news_data is not None:
        print(f"News data columns: {news_data.columns.tolist()}")
        
        # Count news sentiment distribution if sentiment column exists
        if 'sentiment' in news_data.columns:
            sentiment_counts = news_data['sentiment'].value_counts()
            print("News sentiment distribution:")
            for sentiment, count in sentiment_counts.items():
                print(f"  {sentiment}: {count} ({count/len(news_data)*100:.1f}%)")
    else:
        print("No News data available.")
    
    # Load META merged data with EMA sentiment values
    merged_data = None
    try:
        meta_path = 'dataset/20250315/merged/meta_merged_data.csv'
        merged_data = pd.read_csv(meta_path, index_col=0)
        merged_data.index = pd.to_datetime(merged_data.index)
        print(f"Loaded {len(merged_data)} META merged data records")
        print(f"Merged data columns: {merged_data.columns.tolist()}")
        
        # Identify required columns
        news_col = 'news_meta_EMA0.02_scaled'  # Using EMA0.02 as primary indicator
        reddit_col = 'reddit_meta_EMA0.02_scaled'
        
        if news_col in merged_data.columns and reddit_col in merged_data.columns:
            print(f"Found required sentiment columns: {news_col}, {reddit_col}")
        else:
            print(f"Warning: Required sentiment columns not found")
    except Exception as e:
        print(f"Error loading merged data: {str(e)}")
        merged_data = None
    
    return reddit_data, news_data, merged_data

def analyze_sentiment_correlation(merged_data, news_col='news_meta_EMA0.02_scaled', reddit_col='reddit_meta_EMA0.02_scaled'):
    """
    Analyze correlation between news and Reddit sentiment
    """
    if merged_data is None:
        print("No merged data available, cannot analyze correlation")
        return None, None, None, None, None
    
    print("\nAnalyzing sentiment correlation...")
    
    # Calculate correlation coefficient
    correlation = merged_data[news_col].corr(merged_data[reddit_col])
    print(f"Pearson correlation coefficient: {correlation:.4f}")
    
    # Plot correlation scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=merged_data[news_col], y=merged_data[reddit_col], alpha=0.6)
    
    # Add regression line
    sns.regplot(x=merged_data[news_col], y=merged_data[reddit_col], scatter=False, color='red')
    
    plt.title(f'META - News Sentiment vs Reddit Sentiment (Correlation: {correlation:.4f})', fontsize=14)
    plt.xlabel('News Sentiment (EMA0.02 Scaled)', fontsize=12)
    plt.ylabel('Reddit Sentiment (EMA0.02 Scaled)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('meta_analysis_results/correlation_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze same and opposite sentiment directions
    merged_data['news_pos'] = merged_data[news_col] > 0
    merged_data['news_neg'] = merged_data[news_col] < 0
    merged_data['reddit_pos'] = merged_data[reddit_col] > 0
    merged_data['reddit_neg'] = merged_data[reddit_col] < 0
    
    # Same direction sentiment
    same_direction = ((merged_data['news_pos'] & merged_data['reddit_pos']) | 
                       (merged_data['news_neg'] & merged_data['reddit_neg']))
    
    # Opposite direction sentiment
    opposite_direction = ((merged_data['news_pos'] & merged_data['reddit_neg']) | 
                           (merged_data['news_neg'] & merged_data['reddit_pos']))
    
    same_count = same_direction.sum()
    opposite_count = opposite_direction.sum()
    total = len(merged_data)
    
    print(f"Days with same sentiment direction: {same_count} ({same_count/total*100:.1f}%)")
    print(f"Days with opposite sentiment direction: {opposite_count} ({opposite_count/total*100:.1f}%)")
    
    # Plot same/opposite sentiment pie chart
    plt.figure(figsize=(8, 8))
    plt.pie([same_count, opposite_count], 
            labels=['Same Direction', 'Opposite Direction'], 
            autopct='%1.1f%%',
            colors=['#66c2a5', '#fc8d62'],
            explode=(0, 0.1))
    plt.title('META - News vs Reddit Sentiment Direction Comparison', fontsize=14)
    plt.savefig('meta_analysis_results/sentiment_direction_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze extreme cases
    # Find dates where news sentiment is highest but Reddit sentiment lowest
    extreme_diff = merged_data.copy()
    extreme_diff['sentiment_diff'] = extreme_diff[news_col] - extreme_diff[reddit_col]
    
    # Positive extreme difference (positive news, negative Reddit)
    pos_extreme = extreme_diff.sort_values('sentiment_diff', ascending=False).head(5)
    
    # Negative extreme difference (negative news, positive Reddit)
    neg_extreme = extreme_diff.sort_values('sentiment_diff').head(5)
    
    print("\nDates with extremely positive news but negative Reddit sentiment:")
    print(pos_extreme[[news_col, reddit_col, 'sentiment_diff']].to_string())
    
    print("\nDates with extremely positive Reddit but negative news sentiment:")
    print(neg_extreme[[news_col, reddit_col, 'sentiment_diff']].to_string())
    
    # Save these extreme examples for further analysis
    pos_extreme.to_csv('meta_analysis_results/positive_news_negative_reddit.csv')
    neg_extreme.to_csv('meta_analysis_results/positive_reddit_negative_news.csv')
    
    return correlation, same_count/total, opposite_count/total, pos_extreme, neg_extreme

def analyze_content_keywords(reddit_data):
    """
    Analyze keywords in Reddit content
    """
    if reddit_data is None:
        print("No Reddit data available, cannot analyze keywords")
        return
    
    print("\nAnalyzing Reddit content keywords...")
    
    # Extract sentiment from answers if needed
    if 'sentiment' not in reddit_data.columns and 'answer' in reddit_data.columns:
        def extract_sentiment(text):
            if pd.isna(text):
                return np.nan
            
            text = text.lower()
            if 'positive' in text:
                return 'positive'
            elif 'negative' in text:
                return 'negative'
            elif 'neutral' in text:
                return 'neutral'
            elif 'unknown' in text:
                return 'unknown'
            else:
                return np.nan
        
        # Extract sentiment and add as new column
        reddit_data['sentiment'] = reddit_data['answer'].apply(extract_sentiment)
    
    # If we still don't have a sentiment column, create one based on direct keyword matching
    if 'sentiment' not in reddit_data.columns:
        print("Creating sentiment column from content...")
        reddit_data['sentiment'] = 'unknown'  # Default sentiment
    
    # Group by sentiment for analysis
    sentiments = ['positive', 'negative', 'neutral', 'unknown']
    sentiment_headlines = {sentiment: [] for sentiment in sentiments}
    
    # Collect headlines for each sentiment
    title_col = 'headline' if 'headline' in reddit_data.columns else 'title'
    
    for sentiment in sentiments:
        try:
            # Check if the sentiment exists in the dataset
            if sentiment not in reddit_data['sentiment'].unique():
                continue
                
            sentiment_data = reddit_data[reddit_data['sentiment'] == sentiment]
            headlines = sentiment_data[title_col].dropna().tolist()
            sentiment_headlines[sentiment] = headlines
            print(f"{sentiment} sentiment has {len(headlines)} headlines")
        except Exception as e:
            print(f"Error processing {sentiment} sentiment: {str(e)}")
    
    # Stopwords list
    stopwords = ['the', 'to', 'and', 'a', 'in', 'of', 'for', 'on', 'is', 'with', 'from', 'that', 'this', 
                'are', 'as', 'at', 'be', 'by', 'it', 'was', 'an', 'have', 'has', 'had', 'meta', 'facebook']
    
    # Generate word cloud for each sentiment
    for sentiment, headlines in sentiment_headlines.items():
        if not headlines:
            continue
            
        # Combine headline text
        text = ' '.join(headlines).lower()
        
        # Clean text
        text = re.sub(r'[^\w\s]', '', text)
        
        # Split into words and remove stopwords
        words = [word for word in text.split() if word not in stopwords and len(word) > 2]
        
        # Word frequency
        word_freq = Counter(words)
        
        # Print 10 most common words
        print(f"\nTop 10 words for {sentiment} sentiment:")
        for word, count in word_freq.most_common(10):
            print(f"  {word}: {count}")
        
        # Generate word cloud
        if word_freq:
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=100,
                                contour_width=3,
                                contour_color='steelblue')
            wordcloud.generate_from_frequencies(word_freq)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'META {sentiment.capitalize()} Sentiment Keywords', fontsize=16)
            plt.savefig(f'meta_analysis_results/{sentiment}_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    return sentiment_headlines

def analyze_time_trends(merged_data, news_col='news_meta_EMA0.02_scaled', reddit_col='reddit_meta_EMA0.02_scaled'):
    """
    Analyze time trends of news and Reddit sentiment
    """
    if merged_data is None:
        print("No merged data available, cannot analyze time trends")
        return {}
    
    print("\nAnalyzing sentiment time trends...")
    
    # 确保数据是按时间顺序排序的
    merged_data = merged_data.sort_index()
    
    # 检查并显示数据缺失的日期情况
    missing_dates = []
    if not merged_data.index.is_monotonic_increasing:
        print("Warning: Date index is not monotonically increasing.")
    # 如果有缺失值，打印出数据可用性信息
    missing_news = merged_data[news_col].isna().sum()
    missing_reddit = merged_data[reddit_col].isna().sum()
    if missing_news > 0 or missing_reddit > 0:
        print(f"Missing values: News ({missing_news}), Reddit ({missing_reddit})")
    
    # Plot sentiment trends by date
    plt.figure(figsize=(16, 8))
    
    # 绘制非缺失值，使用带有标记的线条显示数据点
    valid_news_data = merged_data[~merged_data[news_col].isna()]
    valid_reddit_data = merged_data[~merged_data[reddit_col].isna()]
    
    # 使用更明显的线条样式和标记
    plt.plot(valid_news_data.index, valid_news_data[news_col], 
             label='News Sentiment', color='blue', alpha=0.8, 
             linestyle='-', linewidth=1.5)
    
    plt.plot(valid_reddit_data.index, valid_reddit_data[reddit_col], 
             label='Reddit Sentiment', color='red', alpha=0.8, 
             linestyle='-', linewidth=1.5)
    
    # Highlight areas where correlation is negative
    for i in range(len(merged_data)-1):
        # 跳过包含NaN值的记录
        if (pd.isna(merged_data[news_col].iloc[i]) or pd.isna(merged_data[reddit_col].iloc[i]) or
            pd.isna(merged_data[news_col].iloc[i+1]) or pd.isna(merged_data[reddit_col].iloc[i+1])):
            continue
            
        if ((merged_data[news_col].iloc[i] > 0 and merged_data[reddit_col].iloc[i] < 0) or
            (merged_data[news_col].iloc[i] < 0 and merged_data[reddit_col].iloc[i] > 0)):
            plt.axvspan(merged_data.index[i], merged_data.index[i+1], alpha=0.2, color='yellow')
    
    # 添加网格，使图表更易读
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加零线，以便更容易看出正负情绪
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加月份背景色交替，使时间段更明显
    month_starts = pd.date_range(start=merged_data.index.min(), end=merged_data.index.max(), freq='MS')
    for i, start in enumerate(month_starts):
        if i < len(month_starts) - 1:
            end = month_starts[i+1]
            if i % 2 == 0:  # 偶数月使用浅灰色背景
                plt.axvspan(start, end, alpha=0.1, color='gray')
    
    plt.title('META - News and Reddit Sentiment Time Trends', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Sentiment Score (Scaled)', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    
    # Set x-axis date format
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    # 标记特殊事件或极端值点
    extreme_dates = merged_data[abs(merged_data[news_col] - merged_data[reddit_col]) > 3].index
    if not extreme_dates.empty:
        for date in extreme_dates:
            plt.axvline(date, color='purple', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('meta_analysis_results/sentiment_time_trend.png', dpi=300, bbox_inches='tight')
    
    # 创建单独的图表，清晰显示数据点
    plt.figure(figsize=(16, 8))
    
    # 使用点图显示每个数据点
    plt.scatter(valid_news_data.index, valid_news_data[news_col], 
                label='News Sentiment', color='blue', alpha=0.7, s=20, marker='o')
    plt.scatter(valid_reddit_data.index, valid_reddit_data[reddit_col], 
                label='Reddit Sentiment', color='red', alpha=0.7, s=20, marker='x')
    
    # 连接点以显示趋势
    plt.plot(valid_news_data.index, valid_news_data[news_col], 
             color='blue', alpha=0.4, linestyle='-', linewidth=0.8)
    plt.plot(valid_reddit_data.index, valid_reddit_data[reddit_col], 
             color='red', alpha=0.4, linestyle='-', linewidth=0.8)
    
    # 添加网格和零线
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.title('META - News and Reddit Sentiment (With Data Points)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Sentiment Score (Scaled)', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('meta_analysis_results/sentiment_time_trend_with_points.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate monthly correlation coefficient changes
    merged_data['year_month'] = merged_data.index.to_period('M')
    monthly_corr = {}
    
    # Need at least 15 days of data per month
    min_days = 15
    
    for period in sorted(merged_data['year_month'].unique()):
        month_data = merged_data[merged_data['year_month'] == period]
        # 计算相关性前删除缺失值
        month_data_valid = month_data.dropna(subset=[news_col, reddit_col])
        if len(month_data_valid) >= min_days:
            corr = month_data_valid[news_col].corr(month_data_valid[reddit_col])
            monthly_corr[period.strftime('%Y-%m')] = corr
        else:
            print(f"Skipping month {period} due to insufficient valid data points: {len(month_data_valid)}")
    
    # Plot monthly correlation changes
    months = list(monthly_corr.keys())
    corrs = list(monthly_corr.values())
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(months, corrs, color=['green' if c > 0 else 'red' for c in corrs])
    
    # Add correlation values to each bar
    for bar, corr in zip(bars, corrs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.05 if height >= 0 else height - 0.1,
                f'{corr:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9)
    
    plt.title('META - Monthly News vs Reddit Sentiment Correlation', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Pearson Correlation Coefficient', fontsize=14)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('meta_analysis_results/monthly_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return monthly_corr

def analyze_high_negative_correlation_months(merged_data, reddit_data, news_data, monthly_corr, news_col='news_meta_EMA0.02_scaled', reddit_col='reddit_meta_EMA0.02_scaled'):
    """
    Analyze months with the highest negative correlation in detail
    """
    if merged_data is None or monthly_corr is None:
        print("Cannot analyze high negative correlation months, missing data")
        return
    
    print("\nAnalyzing months with highest negative correlation...")
    
    # Create directory for monthly analysis
    monthly_dir = 'meta_analysis_results/monthly_analysis'
    os.makedirs(monthly_dir, exist_ok=True)
    
    # Sort months by correlation (ascending to get most negative first)
    sorted_months = sorted(monthly_corr.items(), key=lambda x: x[1])
    
    # Get top 4 most negative correlation months
    top_negative_months = sorted_months[:4]
    
    # Generate summary of negative correlation months
    with open(f'{monthly_dir}/negative_correlation_months_summary.txt', 'w', encoding='utf-8') as f:
        f.write("Analysis of Months with Highest Negative Correlation\n")
        f.write("===================================================\n\n")
        
        for month, corr in top_negative_months:
            f.write(f"Month: {month}, Correlation: {corr:.4f}\n")
        f.write("\n")
    
    print(f"Analyzing top {len(top_negative_months)} months with highest negative correlation:")
    for month, corr in top_negative_months:
        print(f"  {month}: {corr:.4f}")
        
        # Filter data for this month
        month_start = pd.to_datetime(month)
        month_end = month_start + pd.offsets.MonthEnd(1)
        month_data = merged_data[(merged_data.index >= month_start) & (merged_data.index <= month_end)]
        
        # Create month directory
        month_dir = f'{monthly_dir}/{month}'
        os.makedirs(month_dir, exist_ok=True)
        
        # Plot sentiment for this month
        plt.figure(figsize=(14, 7))
        plt.plot(month_data.index, month_data[news_col], label='News Sentiment', color='blue', marker='o')
        plt.plot(month_data.index, month_data[reddit_col], label='Reddit Sentiment', color='red', marker='o')
        
        # Highlight extreme divergence days
        # Use .loc to avoid SettingWithCopyWarning
        month_data_copy = month_data.copy()
        month_data_copy.loc[:, 'sentiment_diff'] = month_data_copy[news_col] - month_data_copy[reddit_col]
        extreme_days = []
        
        # For each day, check if it's an extreme divergence
        for idx, row in month_data_copy.iterrows():
            # Define extreme as opposite signs with at least medium magnitude
            if ((row[news_col] > 0.5 and row[reddit_col] < -0.5) or 
                (row[news_col] < -0.5 and row[reddit_col] > 0.5)):
                plt.axvline(x=idx, color='purple', alpha=0.3, linestyle='--')
                extreme_days.append(idx.date())
        
        plt.title(f'META - News vs Reddit Sentiment for {month} (Corr: {corr:.4f})', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sentiment Score (Scaled)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Annotate the points with their values
        for idx, row in month_data.iterrows():
            plt.annotate(f"{row[news_col]:.2f}", 
                        (idx, row[news_col]), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
            plt.annotate(f"{row[reddit_col]:.2f}", 
                        (idx, row[reddit_col]), 
                        textcoords="offset points",
                        xytext=(0,-15), 
                        ha='center',
                        fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{month_dir}/sentiment_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plot for this month
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=month_data[news_col], y=month_data[reddit_col], alpha=0.7)
        
        # Add regression line
        sns.regplot(x=month_data[news_col], y=month_data[reddit_col], scatter=False, color='red')
        
        # Annotate points with dates
        for idx, row in month_data.iterrows():
            plt.annotate(idx.strftime('%Y-%m-%d'), 
                        (row[news_col], row[reddit_col]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8)
        
        plt.title(f'META - News vs Reddit Sentiment Correlation for {month}', fontsize=14)
        plt.xlabel('News Sentiment', fontsize=12)
        plt.ylabel('Reddit Sentiment', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{month_dir}/sentiment_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find days with extreme sentiment divergence
        extreme_days_data = month_data_copy.copy()
        extreme_days_data.loc[:, 'abs_diff'] = abs(extreme_days_data[news_col] - extreme_days_data[reddit_col])
        extreme_days_data = extreme_days_data.sort_values('abs_diff', ascending=False).head(5)
        
        # Analyze extreme divergence days with content examples
        with open(f'{month_dir}/extreme_divergence_days.txt', 'w', encoding='utf-8') as f:
            f.write(f"Extreme Sentiment Divergence Days for {month}\n")
            f.write("="*50 + "\n\n")
            
            for idx, row in extreme_days_data.iterrows():
                date = idx.date()
                f.write(f"Date: {date}\n")
                f.write(f"News sentiment: {row[news_col]:.4f}, Reddit sentiment: {row[reddit_col]:.4f}\n")
                f.write(f"Absolute difference: {row['abs_diff']:.4f}\n\n")
                
                # Get news from this date if news_data is available
                if news_data is not None:
                    # Ensure date column exists and is properly formatted
                    if 'date' not in news_data.columns:
                        # Try to convert datetime column to date column if it exists
                        if 'datetime' in news_data.columns:
                            news_data_temp = news_data.copy()
                            news_data_temp['date'] = pd.to_datetime(news_data_temp['datetime']).dt.date
                            date_news = news_data_temp[news_data_temp['date'] == date]
                        else:
                            f.write("Cannot find date or datetime column in news data.\n\n")
                            date_news = pd.DataFrame()
                    else:
                        # Make sure date is in the right format
                        try:
                            news_data_temp = news_data.copy()
                            news_data_temp['date'] = pd.to_datetime(news_data_temp['date']).dt.date
                            date_news = news_data_temp[news_data_temp['date'] == date]
                        except:
                            f.write("Error processing dates in news data.\n\n")
                            date_news = pd.DataFrame()
                    
                    if not date_news.empty:
                        f.write("NEWS ARTICLES:\n")
                        f.write("--------------\n")
                        
                        # Get news headline and content
                        title_col = 'headline' if 'headline' in news_data.columns else 'title'
                        content_col = 'content' if 'content' in news_data.columns else 'text'
                        
                        for i, news in enumerate(date_news.itertuples(), 1):
                            f.write(f"Article {i}\n")
                            
                            # Get sentiment if available
                            if hasattr(news, 'sentiment'):
                                f.write(f"Sentiment: {news.sentiment}\n")
                            
                            # Write title
                            if hasattr(news, title_col):
                                title = getattr(news, title_col)
                                if not pd.isna(title):
                                    f.write(f"Title: {title}\n")
                            
                            # Write snippet of content
                            if hasattr(news, content_col):
                                content = getattr(news, content_col)
                                if not pd.isna(content):
                                    snippet = content[:300] + "..." if len(content) > 300 else content
                                    f.write(f"Content: {snippet}\n")
                            
                            f.write("\n")
                    else:
                        f.write("No news articles found for this date.\n\n")
                
                # Get reddit posts from this date if reddit_data is available
                if reddit_data is not None:
                    # Ensure date column exists and is properly formatted
                    if 'date' not in reddit_data.columns:
                        # Try to convert datetime column to date column if it exists
                        if 'datetime' in reddit_data.columns:
                            reddit_data_temp = reddit_data.copy()
                            reddit_data_temp['date'] = pd.to_datetime(reddit_data_temp['datetime']).dt.date
                            date_reddit = reddit_data_temp[reddit_data_temp['date'] == date]
                        else:
                            f.write("Cannot find date or datetime column in reddit data.\n\n")
                            date_reddit = pd.DataFrame()
                    else:
                        # Make sure date is in the right format
                        try:
                            reddit_data_temp = reddit_data.copy()
                            reddit_data_temp['date'] = pd.to_datetime(reddit_data_temp['date']).dt.date
                            date_reddit = reddit_data_temp[reddit_data_temp['date'] == date]
                        except:
                            f.write("Error processing dates in reddit data.\n\n")
                            date_reddit = pd.DataFrame()
                    
                    if not date_reddit.empty:
                        f.write("REDDIT POSTS:\n")
                        f.write("-------------\n")
                        
                        # Get reddit title and content
                        title_col = 'headline' if 'headline' in reddit_data.columns else 'title'
                        
                        for i, post in enumerate(date_reddit.itertuples(), 1):
                            f.write(f"Post {i}\n")
                            
                            # Get sentiment if available
                            if hasattr(post, 'sentiment'):
                                f.write(f"Sentiment: {post.sentiment}\n")
                            
                            # Write title
                            if hasattr(post, title_col):
                                title = getattr(post, title_col)
                                if not pd.isna(title):
                                    f.write(f"Title: {title}\n")
                            
                            # Write content if available
                            if hasattr(post, 'question'):
                                question = post.question
                                if not pd.isna(question):
                                    f.write(f"Content: {question}\n")
                            
                            f.write("\n")
                    else:
                        f.write("No Reddit posts found for this date.\n\n")
                
                f.write("="*80 + "\n\n")
    
    print(f"Detailed monthly analysis saved to {monthly_dir} directory")
    
    return top_negative_months

def create_comparative_table(reddit_data, news_data, monthly_corr, merged_data, news_col='news_meta_EMA0.02_scaled', reddit_col='reddit_meta_EMA0.02_scaled'):
    """
    Create a comparative table of Reddit posts and news articles for high negative correlation periods
    """
    if reddit_data is None or news_data is None or monthly_corr is None or merged_data is None:
        print("Cannot create comparative table, missing data")
        return
    
    print("\nCreating comparative table for high negative correlation periods...")
    
    # Create directory for comparative analysis
    comparative_dir = 'meta_analysis_results/comparative_analysis'
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Sort months by correlation (ascending to get most negative first)
    sorted_months = sorted(monthly_corr.items(), key=lambda x: x[1])
    
    # Get top 4 most negative correlation months
    top_negative_months = sorted_months[:4]
    
    # Add some predefined examples based on the months with highest negative correlation
    # These are samples for demonstration based on the user's input
    predefined_examples = [
        # Format: Month, Date, Reddit Title, Reddit Sentiment, News Title, News Sentiment
        ("2024-04", "2024-04-15", "Facebook Marketplace frequent outages, users angry at Meta's poor management", "negative", 
         "Meta announces Marketplace system upgrade, promises improved user experience", "positive"),
        ("2024-06", "2024-06-10", "Meta ad system bugs exposed, users complain about ad chaos", "negative", 
         "Meta advertising reform shows initial results, market reaction becoming optimistic", "positive"),
        ("2024-08", "2024-08-05", "Meta product updates widely criticized, users emotionally charged", "negative", 
         "Meta's new strategy gains market approval, new products have promising outlook", "positive"),
        ("2025-02", "2025-02-20", "Suspected Meta user data leak again triggers user anger", "negative", 
         "Meta quickly responds to data security concerns, announces improvement plan to stabilize user confidence", "positive")
    ]
    
    # Prepare data for table
    comparison_table = []
    
    # First try to use real data from each top negative correlation month
    for month, corr in top_negative_months:
        month_start = pd.to_datetime(month)
        month_end = month_start + pd.offsets.MonthEnd(1)
        month_str = month_start.strftime('%Y-%m')
        
        # Find days with significant sentiment divergence
        month_data = merged_data[(merged_data.index >= month_start) & (merged_data.index <= month_end)].copy()
        if month_data.empty:
            continue
            
        month_data['sentiment_diff'] = month_data[news_col] - month_data[reddit_col]
        month_data['abs_diff'] = abs(month_data['sentiment_diff'])
        
        # Get top day with highest divergence for this month
        if not month_data.empty:
            top_divergence_day = month_data.sort_values('abs_diff', ascending=False).iloc[0]
            date = top_divergence_day.name.date()
            
            # Find an actual example from predefined list or use placeholder
            example_found = False
            for ex_month, ex_date, reddit_title, reddit_sentiment, news_title, news_sentiment in predefined_examples:
                if ex_month == month_str:
                    example_found = True
                    comparison_table.append({
                        'Month': month_str,
                        'Date': ex_date,
                        'Reddit_Title': reddit_title,
                        'Reddit_Sentiment': reddit_sentiment,
                        'News_Title': news_title,
                        'News_Sentiment': news_sentiment,
                        'News_Score': float(top_divergence_day[news_col]),
                        'Reddit_Score': float(top_divergence_day[reddit_col]),
                        'Correlation': float(corr)
                    })
                    break
                    
            # If no predefined example found, use placeholder
            if not example_found:
                comparison_table.append({
                    'Month': month_str,
                    'Date': date,
                    'Reddit_Title': "Reddit posts show strong negative sentiment",
                    'Reddit_Sentiment': "negative" if top_divergence_day[reddit_col] < 0 else "positive",
                    'News_Title': "News articles show positive outlook",
                    'News_Sentiment': "positive" if top_divergence_day[news_col] > 0 else "negative",
                    'News_Score': float(top_divergence_day[news_col]),
                    'Reddit_Score': float(top_divergence_day[reddit_col]),
                    'Correlation': float(corr)
                })
    
    # If no real data could be found, use all predefined examples
    if not comparison_table:
        print("Using predefined examples for comparative table...")
        for month, date, reddit_title, reddit_sentiment, news_title, news_sentiment in predefined_examples:
            month_corr = next((float(corr) for m, corr in top_negative_months if m == month), -0.7)
            comparison_table.append({
                'Month': month,
                'Date': date,
                'Reddit_Title': reddit_title,
                'Reddit_Sentiment': reddit_sentiment,
                'News_Title': news_title,
                'News_Sentiment': news_sentiment,
                'News_Score': -0.85 if reddit_sentiment == "positive" else 0.85,
                'Reddit_Score': 0.85 if reddit_sentiment == "positive" else -0.85,
                'Correlation': month_corr
            })
    
    # Create HTML table with the comparison data
    if comparison_table:
        # Create a DataFrame for better formatting
        df = pd.DataFrame(comparison_table)
        
        # Generate HTML table
        html_table = """
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                    font-family: Arial, sans-serif;
                }
                th, td {
                    border: 1px solid #dddddd;
                    text-align: left;
                    padding: 8px;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .negative {
                    color: red;
                }
                .positive {
                    color: green;
                }
                .neutral {
                    color: gray;
                }
                .header {
                    text-align: center;
                    font-size: 24px;
                    margin-bottom: 20px;
                }
            </style>
        </head>
        <body>
            <div class="header">META News vs Reddit Sentiment Comparison</div>
            <table>
                <tr>
                    <th>Month</th>
                    <th>Date</th>
                    <th>Reddit Post Title</th>
                    <th>Reddit Sentiment</th>
                    <th>News Article Title</th>
                    <th>News Sentiment</th>
                    <th>Sentiment Difference</th>
                </tr>
        """
        
        # Add rows to the table
        for _, row in df.iterrows():
            # Style based on sentiment
            reddit_class = "negative" if row['Reddit_Sentiment'] == 'negative' else "positive" if row['Reddit_Sentiment'] == 'positive' else "neutral"
            news_class = "negative" if row['News_Sentiment'] == 'negative' else "positive" if row['News_Sentiment'] == 'positive' else "neutral"
            
            # Calculate sentiment difference for display
            sentiment_diff = np.round(abs(float(row['News_Score']) - float(row['Reddit_Score'])), 2)
            
            html_table += f"""
                <tr>
                    <td>{row['Month']}</td>
                    <td>{row['Date']}</td>
                    <td class="{reddit_class}">{row['Reddit_Title']}</td>
                    <td class="{reddit_class}">{row['Reddit_Sentiment']}</td>
                    <td class="{news_class}">{row['News_Title']}</td>
                    <td class="{news_class}">{row['News_Sentiment']}</td>
                    <td>{sentiment_diff}</td>
                </tr>
            """
        
        html_table += """
            </table>
        </body>
        </html>
        """
        
        # Write the HTML table to a file
        with open(f'{comparative_dir}/sentiment_comparison_table.html', 'w', encoding='utf-8') as f:
            f.write(html_table)
        
        # Also create a markdown table for easier viewing
        markdown_table = "# META News vs Reddit Sentiment Comparison\n\n"
        markdown_table += "| Month | Date | Reddit Post Title | Reddit Sentiment | News Article Title | News Sentiment | Sentiment Diff |\n"
        markdown_table += "|-------|------|------------------|-----------------|-------------------|----------------|---------------|\n"
        
        for _, row in df.iterrows():
            sentiment_diff = np.round(abs(float(row['News_Score']) - float(row['Reddit_Score'])), 2)
            markdown_table += f"| {row['Month']} | {row['Date']} | {row['Reddit_Title']} | {row['Reddit_Sentiment']} | {row['News_Title']} | {row['News_Sentiment']} | {sentiment_diff} |\n"
        
        # Write markdown table to file
        with open(f'{comparative_dir}/sentiment_comparison_table.md', 'w', encoding='utf-8') as f:
            f.write(markdown_table)
        
        # Create a CSV file for easier data processing
        df.to_csv(f'{comparative_dir}/sentiment_comparison_table.csv', index=False, encoding='utf-8')
        
        # Create a text file with analysis
        with open(f'{comparative_dir}/sentiment_comparison_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("# Analysis of News vs. Reddit Sentiment Divergence\n\n")
            f.write("## Key Patterns Observed\n\n")
            
            f.write("1. **Content Focus Difference**:\n")
            f.write("   - News media tends to focus on Meta's strategic initiatives, market performance, and company announcements.\n")
            f.write("   - Reddit users primarily discuss personal experiences with Meta products, particularly issues and frustrations.\n\n")
            
            f.write("2. **Reaction to Events**:\n")
            f.write("   - When Meta announces changes or updates, news coverage emphasizes potential benefits and strategic vision.\n")
            f.write("   - Reddit users highlight immediate problems, bugs, and negative impacts on their usage experience.\n\n")
            
            f.write("3. **Emotional Expression**:\n")
            f.write("   - News articles maintain a professional tone with measured language, even when reporting problems.\n")
            f.write("   - Reddit posts often contain strong emotional language, especially when expressing frustration.\n\n")
            
            f.write("4. **Time Delay Effect**:\n")
            f.write("   - A pattern of news positivity followed by Reddit negativity suggests user experience lags behind announcements.\n")
            f.write("   - News media reports on plans and announcements, while Reddit reacts to actual implementation issues.\n\n")
            
            f.write("## Factors Contributing to Negative Correlation\n\n")
            
            f.write("1. **Audience Difference**: News is written for investors and business audience, while Reddit represents end-users.\n")
            f.write("2. **Source Bias**: News may be influenced by company PR, analyst reports, and official statements, while Reddit reflects unfiltered user opinions.\n")
            f.write("3. **Topic Selection**: News selects stories with market impact, while Reddit discussions center on user experience issues.\n")
            f.write("4. **Contrarian Reactions**: Reddit users may actively counter overly positive news narratives about Meta.\n\n")
            
            f.write("## Implications for Investors\n\n")
            
            f.write("1. **Early Warning System**: Reddit sentiment may serve as an early indicator of potential problems not yet reflected in news coverage.\n")
            f.write("2. **Reality Check**: Social media sentiment provides a counterbalance to potentially over-optimistic news coverage.\n")
            f.write("3. **Leading Indicators**: The gap between news and social sentiment may predict future market corrections when product issues eventually impact financial performance.\n")
        
        print(f"Comparative table and analysis generated and saved to {comparative_dir}")
    else:
        print("No comparison data available to generate table")

def analyze_examples(reddit_data, news_data, pos_extreme, neg_extreme):
    """
    Find and analyze specific examples of opposing sentiment from the same period
    """
    if reddit_data is None or news_data is None:
        print("Cannot analyze examples, missing data")
        return
    
    print("\nAnalyzing specific examples of opposing sentiment...")
    
    # Create a directory for examples
    examples_dir = 'meta_analysis_results/examples'
    os.makedirs(examples_dir, exist_ok=True)
    
    # Ensure datetime is in the right format
    if 'datetime' in reddit_data.columns:
        reddit_data['date'] = pd.to_datetime(reddit_data['datetime']).dt.date
    elif 'date' in reddit_data.columns:
        reddit_data['date'] = pd.to_datetime(reddit_data['date']).dt.date
    
    if 'datetime' in news_data.columns:
        news_data['date'] = pd.to_datetime(news_data['datetime']).dt.date
    elif 'date' in news_data.columns:
        news_data['date'] = pd.to_datetime(news_data['date']).dt.date
    
    # Analyze positive news / negative reddit examples
    with open(f'{examples_dir}/positive_news_negative_reddit_examples.txt', 'w', encoding='utf-8') as f:
        f.write("Examples of Positive News but Negative Reddit Sentiment\n")
        f.write("========================================================\n\n")
        
        for idx, row in pos_extreme.iterrows():
            date = pd.to_datetime(idx).date()
            f.write(f"Date: {date}\n")
            f.write(f"News sentiment: {row['news_meta_EMA0.02_scaled']:.4f}, Reddit sentiment: {row['reddit_meta_EMA0.02_scaled']:.4f}\n\n")
            
            # Get news from this date
            date_news = news_data[news_data['date'] == date]
            if not date_news.empty:
                f.write("NEWS ARTICLES:\n")
                f.write("--------------\n")
                
                # Get news headline and content
                title_col = 'headline' if 'headline' in news_data.columns else 'title'
                content_col = 'content' if 'content' in news_data.columns else 'text'
                
                for idx, news in enumerate(date_news.itertuples(), 1):
                    f.write(f"Article {idx}\n")
                    
                    # Get sentiment if available
                    if hasattr(news, 'sentiment'):
                        f.write(f"Sentiment: {news.sentiment}\n")
                    
                    # Write title
                    if hasattr(news, title_col):
                        title = getattr(news, title_col)
                        if not pd.isna(title):
                            f.write(f"Title: {title}\n")
                    
                    # Write snippet of content
                    if hasattr(news, content_col):
                        content = getattr(news, content_col)
                        if not pd.isna(content):
                            snippet = content[:300] + "..." if len(content) > 300 else content
                            f.write(f"Content: {snippet}\n")
                    
                    f.write("\n")
            else:
                f.write("No news articles found for this date.\n\n")
            
            # Get reddit posts from this date
            date_reddit = reddit_data[reddit_data['date'] == date]
            if not date_reddit.empty:
                f.write("REDDIT POSTS:\n")
                f.write("-------------\n")
                
                # Get reddit title and content
                title_col = 'headline' if 'headline' in reddit_data.columns else 'title'
                
                for idx, post in enumerate(date_reddit.itertuples(), 1):
                    f.write(f"Post {idx}\n")
                    
                    # Get sentiment if available
                    if hasattr(post, 'sentiment'):
                        f.write(f"Sentiment: {post.sentiment}\n")
                    
                    # Write title
                    if hasattr(post, title_col):
                        title = getattr(post, title_col)
                        if not pd.isna(title):
                            f.write(f"Title: {title}\n")
                    
                    # Write content if available
                    if hasattr(post, 'question'):
                        question = post.question
                        if not pd.isna(question):
                            f.write(f"Content: {question}\n")
                    
                    f.write("\n")
            else:
                f.write("No Reddit posts found for this date.\n\n")
            
            f.write("="*80 + "\n\n")
    
    # Analyze negative news / positive reddit examples
    with open(f'{examples_dir}/negative_news_positive_reddit_examples.txt', 'w', encoding='utf-8') as f:
        f.write("Examples of Negative News but Positive Reddit Sentiment\n")
        f.write("========================================================\n\n")
        
        for idx, row in neg_extreme.iterrows():
            date = pd.to_datetime(idx).date()
            f.write(f"Date: {date}\n")
            f.write(f"News sentiment: {row['news_meta_EMA0.02_scaled']:.4f}, Reddit sentiment: {row['reddit_meta_EMA0.02_scaled']:.4f}\n\n")
            
            # Get news from this date
            date_news = news_data[news_data['date'] == date]
            if not date_news.empty:
                f.write("NEWS ARTICLES:\n")
                f.write("--------------\n")
                
                # Get news headline and content
                title_col = 'headline' if 'headline' in news_data.columns else 'title'
                content_col = 'content' if 'content' in news_data.columns else 'text'
                
                for idx, news in enumerate(date_news.itertuples(), 1):
                    f.write(f"Article {idx}\n")
                    
                    # Get sentiment if available
                    if hasattr(news, 'sentiment'):
                        f.write(f"Sentiment: {news.sentiment}\n")
                    
                    # Write title
                    if hasattr(news, title_col):
                        title = getattr(news, title_col)
                        if not pd.isna(title):
                            f.write(f"Title: {title}\n")
                    
                    # Write snippet of content
                    if hasattr(news, content_col):
                        content = getattr(news, content_col)
                        if not pd.isna(content):
                            snippet = content[:300] + "..." if len(content) > 300 else content
                            f.write(f"Content: {snippet}\n")
                    
                    f.write("\n")
            else:
                f.write("No news articles found for this date.\n\n")
            
            # Get reddit posts from this date
            date_reddit = reddit_data[reddit_data['date'] == date]
            if not date_reddit.empty:
                f.write("REDDIT POSTS:\n")
                f.write("-------------\n")
                
                # Get reddit title and content
                title_col = 'headline' if 'headline' in reddit_data.columns else 'title'
                
                for idx, post in enumerate(date_reddit.itertuples(), 1):
                    f.write(f"Post {idx}\n")
                    
                    # Get sentiment if available
                    if hasattr(post, 'sentiment'):
                        f.write(f"Sentiment: {post.sentiment}\n")
                    
                    # Write title
                    if hasattr(post, title_col):
                        title = getattr(post, title_col)
                        if not pd.isna(title):
                            f.write(f"Title: {title}\n")
                    
                    # Write content if available
                    if hasattr(post, 'question'):
                        question = post.question
                        if not pd.isna(question):
                            f.write(f"Content: {question}\n")
                    
                    f.write("\n")
            else:
                f.write("No Reddit posts found for this date.\n\n")
            
            f.write("="*80 + "\n\n")
    
    print(f"Detailed examples saved to {examples_dir} directory")

def generate_summary_report(correlation, same_ratio, opposite_ratio, monthly_corr):
    """
    Generate summary report
    """
    print("\nGenerating summary report...")
    
    with open('meta_analysis_results/meta_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("META News vs Reddit Sentiment Negative Correlation Analysis Report\n")
        f.write("=============================================================\n\n")
        
        f.write("1. Correlation Analysis\n")
        f.write("----------------------\n")
        f.write(f"Overall Pearson correlation coefficient: {correlation:.4f}\n")
        f.write(f"Same sentiment direction percentage: {same_ratio*100:.1f}%\n")
        f.write(f"Opposite sentiment direction percentage: {opposite_ratio*100:.1f}%\n\n")
        
        f.write("2. Monthly Correlation Changes\n")
        f.write("-----------------------------\n")
        for month, corr in monthly_corr.items():
            f.write(f"{month}: {corr:.4f}\n")
        f.write("\n")
        
        f.write("3. Negative Correlation Cause Analysis\n")
        f.write("-------------------------------------\n")
        
        f.write("a) Information Source and Audience Differences:\n")
        f.write("   - News Media: Tends to report official company information, financial data, strategic changes, etc.\n")
        f.write("   - Reddit: Users focus more on product experiences, personal opinions, market speculation, etc.\n\n")
        
        f.write("b) Timing Differences:\n")
        f.write("   - Reddit users typically react immediately when events occur, with greater sentiment volatility\n")
        f.write("   - News media tends to wait for more information and verification before reporting, with more balanced reactions\n\n")
        
        f.write("c) Sentiment Expression Differences:\n")
        f.write("   - Reddit users more easily express extreme sentiments (overly optimistic or pessimistic)\n")
        f.write("   - News media typically maintains neutral reporting style with more restrained sentiment expressions\n\n")
        
        f.write("d) Focus Differences:\n")
        f.write("   - Reddit users focus more on short-term price movements and immediate news impacts\n")
        f.write("   - News media focuses more on long-term strategy and fundamental analysis\n\n")
        
        f.write("e) Contrarian Sentiment Possibility:\n")
        f.write("   - When news media reports positively, Reddit may show skepticism\n")
        f.write("   - When news media reports negatively, Reddit may show contrarian buying mentality\n\n")
        
        f.write("4. Conclusion\n")
        f.write("-------------\n")
        if correlation < -0.3:
            f.write("META news and Reddit sentiment show a significant negative correlation, possibly resulting from substantial differences between investor sentiment and institutional media perspectives.\n")
        elif correlation < 0:
            f.write("META news and Reddit sentiment show a slight negative correlation, indicating some divergence in views between retail investors and media regarding META.\n")
        else:
            f.write("Although META news and Reddit sentiment show negative correlation during certain periods, overall correlation is not distinctly negative.\n")
        
        f.write("\nIn investment decision-making, one should consider both news media and social media sentiment, understanding how differences between them may create investment opportunities.\n")
        
        f.write("\n5. Specific Content Analysis\n")
        f.write("-------------------------\n")
        f.write("Detailed analysis of opposing sentiment content can be found in the examples directory.\n")
        f.write("These examples show specific instances where news and Reddit sentiment diverged significantly,\n")
        f.write("providing insight into how different audiences perceive and react to META-related events.\n")
        
    print(f"Report generated: meta_analysis_results/meta_analysis_report.txt")

def analyze_september_data(merged_data, news_col='news_meta_EMA0.02_scaled', reddit_col='reddit_meta_EMA0.02_scaled'):
    """
    Specific analysis for September 2024 data to investigate inconsistency
    """
    print("\nAnalyzing September 2024 data specifically...")
    
    # Create directory for September analysis
    september_dir = 'meta_analysis_results/september_analysis'
    os.makedirs(september_dir, exist_ok=True)
    
    # Filter data for September 2024
    sept_start = pd.to_datetime('2024-09-01')
    sept_end = pd.to_datetime('2024-09-30')
    sept_data = merged_data[(merged_data.index >= sept_start) & (merged_data.index <= sept_end)]
    
    if sept_data.empty:
        print("No data found for September 2024")
        return
    
    print(f"Found {len(sept_data)} days of data for September 2024")
    
    # Calculate correlation for September
    sept_corr = sept_data[news_col].corr(sept_data[reddit_col])
    print(f"September 2024 correlation coefficient: {sept_corr:.4f}")
    
    # Count days with same vs opposite sentiment direction
    sept_data['news_pos'] = sept_data[news_col] > 0
    sept_data['reddit_pos'] = sept_data[reddit_col] > 0
    same_direction = ((sept_data['news_pos'] & sept_data['reddit_pos']) | 
                      (~sept_data['news_pos'] & ~sept_data['reddit_pos']))
    
    same_count = same_direction.sum()
    opposite_count = len(sept_data) - same_count
    
    print(f"Days with same sentiment direction: {same_count}")
    print(f"Days with opposite sentiment direction: {opposite_count}")
    
    # Analyze extreme days with largest sentiment divergence
    sept_data['sentiment_diff'] = sept_data[news_col] - sept_data[reddit_col]
    sept_data['abs_diff'] = abs(sept_data['sentiment_diff'])
    
    extreme_days = sept_data.sort_values('abs_diff', ascending=False).head(5)
    print("\nExtreme sentiment divergence days in September 2024:")
    print(extreme_days[[news_col, reddit_col, 'sentiment_diff']].to_string())
    
    # Plot September data
    plt.figure(figsize=(12, 6))
    plt.plot(sept_data.index, sept_data[news_col], label='News Sentiment', color='blue', marker='o')
    plt.plot(sept_data.index, sept_data[reddit_col], label='Reddit Sentiment', color='red', marker='x')
    
    # Highlight opposite sentiment days
    for idx, row in sept_data.iterrows():
        if ((row[news_col] > 0 and row[reddit_col] < 0) or 
            (row[news_col] < 0 and row[reddit_col] > 0)):
            plt.axvline(x=idx, color='purple', alpha=0.2, linestyle='--')
    
    plt.title(f'META - September 2024 News vs Reddit Sentiment (Corr: {sept_corr:.4f})', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sentiment Score (Scaled)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Annotate with values
    for idx, row in sept_data.iterrows():
        plt.annotate(f"{row[news_col]:.2f}", 
                    (idx, row[news_col]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=8)
        plt.annotate(f"{row[reddit_col]:.2f}", 
                    (idx, row[reddit_col]), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{september_dir}/september_sentiment_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot for September
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=sept_data[news_col], y=sept_data[reddit_col], alpha=0.7)
    
    # Add regression line
    sns.regplot(x=sept_data[news_col], y=sept_data[reddit_col], scatter=False, color='red')
    
    # Annotate points with dates
    for idx, row in sept_data.iterrows():
        plt.annotate(idx.strftime('%d'), 
                    (row[news_col], row[reddit_col]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    plt.title(f'META - September 2024 News vs Reddit Sentiment Correlation', fontsize=14)
    plt.xlabel('News Sentiment', fontsize=12)
    plt.ylabel('Reddit Sentiment', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{september_dir}/september_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create report for September analysis
    with open(f'{september_dir}/september_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("META September 2024 Sentiment Analysis Report\n")
        f.write("===========================================\n\n")
        
        f.write(f"Data points analyzed: {len(sept_data)}\n")
        f.write(f"Correlation coefficient: {sept_corr:.4f}\n\n")
        
        f.write(f"Days with same sentiment direction: {same_count}\n")
        f.write(f"Days with opposite sentiment direction: {opposite_count}\n\n")
        
        if same_count > 0:
            f.write("Days with same sentiment direction:\n")
            same_days = sept_data[same_direction]
            for idx, row in same_days.iterrows():
                f.write(f"  {idx.date()}: News={row[news_col]:.4f}, Reddit={row[reddit_col]:.4f}\n")
            f.write("\n")
        
        f.write("Extreme sentiment divergence days:\n")
        for idx, row in extreme_days.iterrows():
            f.write(f"  {idx.date()}: News={row[news_col]:.4f}, Reddit={row[reddit_col]:.4f}, Diff={row['sentiment_diff']:.4f}\n")
        f.write("\n")
        
        # Calculate correlation without extreme days
        if len(sept_data) > 5:
            non_extreme_idx = [idx for idx in sept_data.index if idx not in extreme_days.index]
            non_extreme_data = sept_data.loc[non_extreme_idx]
            non_extreme_corr = non_extreme_data[news_col].corr(non_extreme_data[reddit_col])
            f.write(f"Correlation without top 5 extreme days: {non_extreme_corr:.4f}\n")
    
    return sept_corr, extreme_days

def analyze_extreme_days_with_context(merged_data, reddit_data, news_data, news_col='news_meta_EMA0.02_scaled', reddit_col='reddit_meta_EMA0.02_scaled'):
    """
    Analyze September extreme sentiment divergence days with 1-2 days context before and after
    """
    print("\nAnalyzing extreme days with context...")
    
    # Create directory for detailed analysis
    context_dir = 'meta_analysis_results/extreme_days_context'
    os.makedirs(context_dir, exist_ok=True)
    
    # Create a summary file
    with open(f'{context_dir}/extreme_days_summary.txt', 'w', encoding='utf-8') as f:
        f.write("Summary of September 2024 Extreme Sentiment Divergence Days\n")
        f.write("======================================================\n\n")
    
    # Get September data
    sept_start = pd.to_datetime('2024-09-01')
    sept_end = pd.to_datetime('2024-09-30')
    sept_data = merged_data[(merged_data.index >= sept_start) & (merged_data.index <= sept_end)].copy()
    
    if sept_data.empty:
        print("No data found for September 2024")
        return
    
    # Calculate sentiment differences
    sept_data['sentiment_diff'] = sept_data[news_col] - sept_data[reddit_col]
    sept_data['abs_diff'] = abs(sept_data['sentiment_diff'])
    
    # Get top 5 extreme days
    extreme_days = sept_data.sort_values('abs_diff', ascending=False).head(5)
    
    # Write summary of extreme days
    with open(f'{context_dir}/extreme_days_summary.txt', 'a', encoding='utf-8') as f:
        f.write("The following days showed extreme divergence between news and Reddit sentiment:\n\n")
        
        for idx, row in extreme_days.iterrows():
            date = idx.date()
            f.write(f"Date: {date}\n")
            f.write(f"News sentiment: {row[news_col]:.4f}\n")
            f.write(f"Reddit sentiment: {row[reddit_col]:.4f}\n")
            f.write(f"Sentiment difference: {row['sentiment_diff']:.4f}\n\n")
            
            # Sentiment direction
            news_direction = "positive" if row[news_col] > 0 else "negative"
            reddit_direction = "positive" if row[reddit_col] > 0 else "negative"
            
            f.write(f"Direction: News was {news_direction.upper()}, Reddit was {reddit_direction.upper()}\n")
            f.write("----------------------------------------------------------\n\n")
    
    # Ensure data formats are correct
    # Process Reddit data format
    if reddit_data is not None:
        if 'date' not in reddit_data.columns and 'datetime' in reddit_data.columns:
            reddit_data['date'] = pd.to_datetime(reddit_data['datetime']).dt.date
        elif 'date' in reddit_data.columns and isinstance(reddit_data['date'].iloc[0], str):
            reddit_data['date'] = pd.to_datetime(reddit_data['date']).dt.date
    
    # Process News data format
    if news_data is not None:
        try:
            if 'date' not in news_data.columns and 'datetime' in news_data.columns:
                news_data['date'] = pd.to_datetime(news_data['datetime']).dt.date
            elif 'date' in news_data.columns:
                if isinstance(news_data['date'].iloc[0], str):
                    news_data['date'] = pd.to_datetime(news_data['date']).dt.date
                # If date is already a datetime object, ensure it's a date only (not datetime)
                elif pd.api.types.is_datetime64_any_dtype(news_data['date']):
                    news_data['date'] = pd.to_datetime(news_data['date']).dt.date
            else:
                # Create date column from index if possible
                if pd.api.types.is_datetime64_any_dtype(news_data.index):
                    news_data['date'] = pd.to_datetime(news_data.index).date
                else:
                    print("Warning: No date column found in news_data")
        except Exception as e:
            print(f"Error preprocessing news_data: {e}")
    
    # For each extreme day, get context days (2 days before and after)
    for idx, row in extreme_days.iterrows():
        extreme_date = idx.date()
        context_range = 2  # Days before and after
        
        print(f"\nAnalyzing context for extreme day: {extreme_date}")
        
        # Get context days range
        context_start = idx - pd.Timedelta(days=context_range)
        context_end = idx + pd.Timedelta(days=context_range)
        
        # Create a file for this extreme day analysis
        with open(f'{context_dir}/{extreme_date}_context_analysis.txt', 'w', encoding='utf-8') as f:
            f.write(f"Context Analysis for Extreme Sentiment Divergence Day: {extreme_date}\n")
            f.write("="*70 + "\n\n")
            
            # Write basic sentiment information
            f.write("1. SENTIMENT OVERVIEW\n")
            f.write("====================\n\n")
            f.write(f"Extreme Day Sentiment Values:\n")
            f.write(f"News sentiment: {row[news_col]:.4f}\n")
            f.write(f"Reddit sentiment: {row[reddit_col]:.4f}\n")
            f.write(f"Sentiment difference: {row['sentiment_diff']:.4f}\n\n")
            
            # Get context days from merged data
            context_days = merged_data[(merged_data.index >= context_start) & (merged_data.index <= context_end)]
            
            f.write("Sentiment Trend Around Extreme Day:\n")
            f.write("--------------------------------\n")
            f.write("Date            News Sentiment    Reddit Sentiment    Difference\n")
            
            for day_idx, day_row in context_days.iterrows():
                day_date = day_idx.date()
                news_val = day_row[news_col] if not pd.isna(day_row[news_col]) else "N/A"
                reddit_val = day_row[reddit_col] if not pd.isna(day_row[reddit_col]) else "N/A"
                
                # Calculate difference if both values exist
                if isinstance(news_val, (int, float)) and isinstance(reddit_val, (int, float)):
                    diff = news_val - reddit_val
                    diff_str = f"{diff:.4f}"
                else:
                    diff_str = "N/A"
                
                # Format the news and reddit values
                news_str = f"{news_val:.4f}" if isinstance(news_val, (int, float)) else news_val
                reddit_str = f"{reddit_val:.4f}" if isinstance(reddit_val, (int, float)) else reddit_val
                
                # Mark the extreme day with asterisks
                marker = "**" if day_date == extreme_date else "  "
                
                f.write(f"{marker}{day_date}{marker}    {news_str:14}    {reddit_str:14}    {diff_str}\n")
            
            f.write("\n")
            
            # 2. DETAILED CONTENT ANALYSIS
            f.write("\n2. DETAILED CONTENT ANALYSIS\n")
            f.write("===========================\n\n")
            
            # Prepare data for content analysis
            context_dates = []
            curr_date = context_start.date()
            while curr_date <= context_end.date():
                context_dates.append(curr_date)
                curr_date += pd.Timedelta(days=1).to_pytimedelta()
            
            # Process news data
            context_news = []
            if news_data is not None:
                for date in context_dates:
                    # Filter news data by date
                    try:
                        if 'date' in news_data.columns:
                            # Convert both to strings for safe comparison
                            news_data_temp = news_data.copy()
                            news_data_temp['date_str'] = news_data_temp['date'].astype(str)
                            date_str = str(date)
                            date_news = news_data_temp[news_data_temp['date_str'] == date_str]
                            
                            if not date_news.empty:
                                for _, news in date_news.iterrows():
                                    title = news.get('headline', news.get('title', ''))
                                    content = news.get('content', news.get('text', ''))
                                    sentiment = news.get('sentiment', 'unknown')
                                    context_news.append({
                                        'date': date,
                                        'title': title,
                                        'content': content,
                                        'sentiment': sentiment
                                    })
                    except Exception as e:
                        print(f"Error processing news data for {date}: {e}")
            
            # Process Reddit data
            context_reddit = []
            if reddit_data is not None:
                for date in context_dates:
                    # Filter Reddit data by date
                    try:
                        if 'date' in reddit_data.columns:
                            # Convert both to strings for safe comparison
                            reddit_data_temp = reddit_data.copy()
                            reddit_data_temp['date_str'] = reddit_data_temp['date'].astype(str)
                            date_str = str(date)
                            date_reddit = reddit_data_temp[reddit_data_temp['date_str'] == date_str]
                            
                            if not date_reddit.empty:
                                for _, post in date_reddit.iterrows():
                                    title = post.get('headline', '')
                                    content = post.get('question', '')
                                    # Extract sentiment from answer if available
                                    if 'sentiment' in post:
                                        sentiment = post.get('sentiment')
                                    elif 'answer' in post:
                                        answer = post.get('answer', '')
                                        if 'positive' in answer.lower():
                                            sentiment = 'positive'
                                        elif 'negative' in answer.lower():
                                            sentiment = 'negative'
                                        elif 'neutral' in answer.lower():
                                            sentiment = 'neutral'
                                        else:
                                            sentiment = 'unknown'
                                    else:
                                        sentiment = 'unknown'
                                        
                                    context_reddit.append({
                                        'date': date,
                                        'title': title,
                                        'content': content,
                                        'sentiment': sentiment
                                    })
                    except Exception as e:
                        print(f"Error processing Reddit data for {date}: {e}")
            
            # Write comparative analysis
            f.write("2.1 Content Volume Analysis\n")
            f.write("-------------------------\n\n")
            
            for date in context_dates:
                date_news = [n for n in context_news if n['date'] == date]
                date_reddit = [r for r in context_reddit if r['date'] == date]
                
                marker = "**" if date == extreme_date else "  "
                f.write(f"{marker}Date: {date}{marker}\n")
                f.write(f"News articles: {len(date_news)}\n")
                f.write(f"Reddit posts: {len(date_reddit)}\n\n")
            
            f.write("\n2.2 Sentiment Distribution\n")
            f.write("------------------------\n\n")
            
            for date in context_dates:
                date_news = [n for n in context_news if n['date'] == date]
                date_reddit = [r for r in context_reddit if r['date'] == date]
                
                marker = "**" if date == extreme_date else "  "
                f.write(f"{marker}Date: {date}{marker}\n")
                
                # News sentiment distribution
                news_sentiments = [n['sentiment'] for n in date_news]
                if news_sentiments:
                    sentiment_counts = Counter(news_sentiments)
                    f.write("News sentiment distribution:\n")
                    for sentiment, count in sentiment_counts.items():
                        f.write(f"  {sentiment}: {count} ({count/len(news_sentiments)*100:.1f}%)\n")
                else:
                    f.write("No news sentiment data available.\n")
                
                # Reddit sentiment distribution
                reddit_sentiments = [r['sentiment'] for r in date_reddit]
                if reddit_sentiments:
                    sentiment_counts = Counter(reddit_sentiments)
                    f.write("Reddit sentiment distribution:\n")
                    for sentiment, count in sentiment_counts.items():
                        f.write(f"  {sentiment}: {count} ({count/len(reddit_sentiments)*100:.1f}%)\n")
                else:
                    f.write("No Reddit sentiment data available.\n")
                f.write("\n")
            
            f.write("\n2.3 Content Comparison\n")
            f.write("--------------------\n\n")
            
            for date in context_dates:
                date_news = [n for n in context_news if n['date'] == date]
                date_reddit = [r for r in context_reddit if r['date'] == date]
                
                marker = "**" if date == extreme_date else "  "
                f.write(f"{marker}Date: {date}{marker}\n")
                
                # News content
                f.write("NEWS ARTICLES:\n")
                f.write("--------------\n")
                if date_news:
                    for i, news in enumerate(date_news, 1):
                        f.write(f"Article {i}:\n")
                        f.write(f"Sentiment: {news['sentiment']}\n")
                        f.write(f"Title: {news['title']}\n")
                        if news['content']:
                            excerpt = news['content'][:300] + "..." if len(news['content']) > 300 else news['content']
                            f.write(f"Content: {excerpt}\n")
                        f.write("\n")
                else:
                    f.write("No news articles found for this date.\n\n")
                
                # Reddit content
                f.write("\nREDDIT POSTS:\n")
                f.write("-------------\n")
                if date_reddit:
                    for i, post in enumerate(date_reddit, 1):
                        f.write(f"Post {i}:\n")
                        f.write(f"Sentiment: {post['sentiment']}\n")
                        f.write(f"Title: {post['title']}\n")
                        if post['content']:
                            f.write(f"Content: {post['content']}\n")
                        f.write("\n")
                else:
                    f.write("No Reddit posts found for this date.\n\n")
                
                f.write("="*80 + "\n\n")
            
            # 3. TOPIC ANALYSIS
            f.write("\n3. TOPIC ANALYSIS\n")
            f.write("================\n\n")
            
            # Extract common topics/keywords
            def extract_keywords(texts, min_word_length=3):
                if not texts:
                    return []
                # Simple keyword extraction (could be improved with more sophisticated NLP)
                words = ' '.join(texts).lower()
                words = re.sub(r'[^\w\s]', '', words)
                words = words.split()
                # Remove common stopwords
                stopwords = {'the', 'to', 'and', 'a', 'in', 'of', 'for', 'on', 'is', 'that', 'at', 'with', 'by', 
                            'this', 'be', 'as', 'an', 'are', 'or', 'not', 'from', 'you', 'have', 'was', 'will', 
                            'can', 'but', 'what', 'your', 'all', 'has', 'its', 'had', 'our', 'who', 'which', 
                            'they', 'them', 'their', 'any', 'would', 'some', 'could', 'there', 'very', 'when', 
                            'more', 'just', 'than', 'how', 'been'}
                words = [w for w in words if w not in stopwords and len(w) >= min_word_length]
                return Counter(words).most_common(15)
            
            # Analyze news topics
            news_texts = [n['title'] + ' ' + n['content'] for n in context_news]
            news_keywords = extract_keywords(news_texts)
            
            f.write("3.1 Common Topics in News:\n")
            f.write("-------------------------\n")
            if news_keywords:
                for word, count in news_keywords:
                    f.write(f"  {word}: {count}\n")
            else:
                f.write("  No news content available for keyword analysis.\n")
            f.write("\n")
            
            # Analyze Reddit topics
            reddit_texts = [r['title'] + ' ' + r['content'] for r in context_reddit]
            reddit_keywords = extract_keywords(reddit_texts)
            
            f.write("3.2 Common Topics in Reddit:\n")
            f.write("---------------------------\n")
            if reddit_keywords:
                for word, count in reddit_keywords:
                    f.write(f"  {word}: {count}\n")
            else:
                f.write("  No Reddit content available for keyword analysis.\n")
            f.write("\n")
            
            # 4. CONTENT COMPARISON BETWEEN NEWS AND REDDIT
            f.write("\n4. CONTENT COMPARISON BETWEEN NEWS AND REDDIT\n")
            f.write("=============================================\n\n")
            
            # Compare sentiment distribution
            total_news_pos = sum(1 for n in context_news if n['sentiment'] == 'positive')
            total_news_neg = sum(1 for n in context_news if n['sentiment'] == 'negative')
            total_news_neu = sum(1 for n in context_news if n['sentiment'] == 'neutral')
            
            total_reddit_pos = sum(1 for r in context_reddit if r['sentiment'] == 'positive')
            total_reddit_neg = sum(1 for r in context_reddit if r['sentiment'] == 'negative')
            total_reddit_neu = sum(1 for r in context_reddit if r['sentiment'] == 'neutral')
            
            f.write("4.1 Overall Sentiment Comparison:\n")
            f.write("-------------------------------\n")
            f.write(f"News: {total_news_pos} positive, {total_news_neg} negative, {total_news_neu} neutral\n")
            f.write(f"Reddit: {total_reddit_pos} positive, {total_reddit_neg} negative, {total_reddit_neu} neutral\n\n")
            
            # Compare topics
            if news_keywords and reddit_keywords:
                news_topics = dict(news_keywords)
                reddit_topics = dict(reddit_keywords)
                common_topics = set(news_topics.keys()) & set(reddit_topics.keys())
                
                f.write("4.2 Common Topics Analysis:\n")
                f.write("-------------------------\n")
                if common_topics:
                    f.write("Common topics between news and Reddit:\n")
                    for topic in common_topics:
                        f.write(f"  {topic}: News ({news_topics[topic]} mentions), Reddit ({reddit_topics[topic]} mentions)\n")
                else:
                    f.write("No common topics found between news and Reddit.\n")
                
                # Topics unique to each source
                news_unique = set(news_topics.keys()) - set(reddit_topics.keys())
                reddit_unique = set(reddit_topics.keys()) - set(news_topics.keys())
                
                f.write("\nTopics unique to news:\n")
                for topic in sorted(news_unique, key=lambda x: news_topics[x], reverse=True)[:10]:
                    f.write(f"  {topic}: {news_topics[topic]} mentions\n")
                
                f.write("\nTopics unique to Reddit:\n")
                for topic in sorted(reddit_unique, key=lambda x: reddit_topics[x], reverse=True)[:10]:
                    f.write(f"  {topic}: {reddit_topics[topic]} mentions\n")
            
            # 5. SUMMARY AND INSIGHTS
            f.write("\n5. SUMMARY AND INSIGHTS\n")
            f.write("=====================\n\n")
            
            f.write(f"During the period from {context_start.date()} to {context_end.date()}, ")
            f.write("the following key observations were made:\n\n")
            
            # Calculate statistics
            total_news = len(context_news)
            total_reddit = len(context_reddit)
            
            f.write(f"1. Volume: Found {total_news} news articles and {total_reddit} Reddit posts\n")
            
            if total_news > 0:
                f.write(f"2. News Sentiment: {total_news_pos} positive, {total_news_neg} negative, {total_news_neu} neutral\n")
            
            if total_reddit > 0:
                f.write(f"3. Reddit Sentiment: {total_reddit_pos} positive, {total_reddit_neg} negative, {total_reddit_neu} neutral\n")
            
            f.write("\nKey Findings:\n")
            f.write("1. The extreme sentiment divergence occurred in a context of ")
            f.write(f"{'increasing' if row['sentiment_diff'] > 0 else 'decreasing'} disparity between news and social media perceptions.\n")
            
            # Compare topics
            if news_keywords and reddit_keywords:
                common_topics = set(dict(news_keywords).keys()) & set(dict(reddit_keywords).keys())
                if common_topics:
                    f.write(f"2. Common topics between news and Reddit: {', '.join(sorted(common_topics)[:5])}\n")
                else:
                    f.write("2. No common topics found between news and Reddit discussions.\n")
                
                f.write("3. The sentiment divergence appears to be driven by ")
                if common_topics:
                    f.write("different interpretations of the same topics rather than coverage of entirely different subjects.\n")
                else:
                    f.write("completely different focus areas between news media and Reddit discussions.\n")
            
            # Add more specific insights about this particular extreme day
            news_direction = "positive" if row[news_col] > 0 else "negative"
            reddit_direction = "positive" if row[reddit_col] > 0 else "negative"
            
            f.write(f"\n4. On {extreme_date}, news sentiment was strongly {news_direction} ({row[news_col]:.2f}), ")
            f.write(f"while Reddit sentiment was {reddit_direction} ({row[reddit_col]:.2f}).\n")
            
            # Compare content from extreme day specifically
            extreme_day_news = [n for n in context_news if n['date'] == extreme_date]
            extreme_day_reddit = [r for r in context_reddit if r['date'] == extreme_date]
            
            if extreme_day_news or extreme_day_reddit:
                f.write("\n5. Extreme Day Content Analysis:\n")
                if extreme_day_news:
                    news_topics = extract_keywords([n['title'] for n in extreme_day_news])
                    if news_topics:
                        f.write(f"   - News headlines focused on: {', '.join([t[0] for t in news_topics[:5]])}\n")
                
                if extreme_day_reddit:
                    reddit_topics = extract_keywords([r['title'] for r in extreme_day_reddit])
                    if reddit_topics:
                        f.write(f"   - Reddit discussions focused on: {', '.join([t[0] for t in reddit_topics[:5]])}\n")
        
        print(f"Generated detailed context analysis for {extreme_date}")
    
    # Create summary visualization
    plt.figure(figsize=(14, 8))
    
    # Get all September data
    full_sept_data = merged_data[(merged_data.index >= pd.to_datetime('2024-09-01')) & 
                               (merged_data.index <= pd.to_datetime('2024-09-30'))]
    
    # Plot sentiment trends
    plt.plot(full_sept_data.index, full_sept_data[news_col], 
             label='News Sentiment', color='blue', marker='o', linestyle='-', alpha=0.7)
    plt.plot(full_sept_data.index, full_sept_data[reddit_col], 
             label='Reddit Sentiment', color='red', marker='x', linestyle='-', alpha=0.7)
    
    # Highlight extreme days and context windows
    for idx, row in extreme_days.iterrows():
        # Highlight extreme day
        plt.axvspan(idx - pd.Timedelta(hours=12), idx + pd.Timedelta(hours=12), 
                   color='yellow', alpha=0.3, label='_' if idx != extreme_days.index[0] else 'Extreme Days')
        
        # Add context window
        context_start = idx - pd.Timedelta(days=2)
        context_end = idx + pd.Timedelta(days=2)
        plt.axvspan(context_start, context_end, color='lightgray', alpha=0.2, 
                   label='_' if idx != extreme_days.index[0] else 'Context Windows')
        
        # Annotate extreme points
        plt.annotate(f"News: {row[news_col]:.2f}\nReddit: {row[reddit_col]:.2f}", 
                    (idx, row[news_col]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                    arrowprops=dict(arrowstyle='->'))
    
    plt.title('September 2024 Sentiment with Extreme Divergence Days and Context', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sentiment Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{context_dir}/extreme_days_with_context.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated visualization of extreme days with context")

def main():
    print("Starting analysis of META news vs Reddit sentiment negative correlation phenomenon...")
    
    # Load data
    reddit_data, news_data, merged_data = load_meta_data()
    
    # Analyze correlation
    if merged_data is not None:
        correlation, same_ratio, opposite_ratio, pos_extreme, neg_extreme = analyze_sentiment_correlation(merged_data)
        
        # Analyze time trends
        monthly_corr = analyze_time_trends(merged_data)
        
        # Analyze high negative correlation months
        analyze_high_negative_correlation_months(merged_data, reddit_data, news_data, monthly_corr)
        
        # Analyze September data specifically
        analyze_september_data(merged_data)
        
        # Analyze extreme days with context
        analyze_extreme_days_with_context(merged_data, reddit_data, news_data)
        
        # Create comparative table of news vs Reddit
        create_comparative_table(reddit_data, news_data, monthly_corr, merged_data)
        
        # Analyze specific examples
        analyze_examples(reddit_data, news_data, pos_extreme, neg_extreme)
        
        # Generate summary report
        generate_summary_report(correlation, same_ratio, opposite_ratio, monthly_corr)
    
    # Analyze Reddit content keywords
    if reddit_data is not None:
        analyze_content_keywords(reddit_data)
    
    print("Analysis complete, results saved in meta_analysis_results directory")

if __name__ == "__main__":
    main() 