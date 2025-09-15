import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
import sys
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.signal import correlate

# Set style for all plots
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'  # A more commonly available font
plt.rcParams['font.size'] = 10

def calculate_ema(series, span=2):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def load_data(file_path):
    """加载并验证预处理数据"""
    print(f"Loading data from {file_path}")
    try:
        df = pd.read_pickle(file_path)
        print("Data loaded successfully")
        print("DataFrame shape:", df.shape)
        print("First few rows:\n", df.head())
        
        # 验证必要的列
        required_columns = ['news', 'reddit', 'stock']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 处理缺失值
        missing_count = df.isnull().sum()
        if missing_count.sum() > 0:
            print("Missing values detected:")
            for col in df.columns:
                if missing_count[col] > 0:
                    print(f"  {col}: {missing_count[col]} missing values")
            
            # 对于少量缺失值，可以使用前向填充方法
            if missing_count.max() < len(df) * 0.1:  # 如果缺失值少于10%
                print("Filling missing values using forward fill method")
                df = df.fillna(method='ffill').fillna(method='bfill')  # 先前向填充，再后向填充剩余的
            else:
                print("Too many missing values, dropping rows with any missing values")
                df = df.dropna()
            
            print(f"After handling missing values, shape: {df.shape}")
        
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
        return None

def calculate_correlation(news_sentiment, reddit_sentiment):
    """Calculate correlation between news and reddit sentiment"""
    try:
        # Remove any NaN values
        mask = ~(np.isnan(news_sentiment) | np.isnan(reddit_sentiment))
        news_clean = news_sentiment[mask]
        reddit_clean = reddit_sentiment[mask]
        
        if len(news_clean) < 2:
            raise ValueError("Not enough valid data points for correlation analysis")
            
        correlation, p_value = stats.pearsonr(news_clean, reddit_clean)
        print(f"Correlation: {correlation:.4f}, P-value: {p_value:.4f}")
        return correlation, p_value
    except Exception as e:
        print(f"Error calculating correlation: {str(e)}")
        raise

def calculate_cross_correlation(news_sentiment, reddit_sentiment, max_lag=30):
    """Calculate cross-correlation between news and reddit sentiment"""
    try:
        # Remove any NaN values
        mask = ~(np.isnan(news_sentiment) | np.isnan(reddit_sentiment))
        news_clean = news_sentiment[mask]
        reddit_clean = reddit_sentiment[mask]
        
        # Calculate cross-correlation
        cross_corr = correlate(news_clean, reddit_clean, mode='full')
        
        # Create lags array with correct length
        lags = np.arange(-(len(cross_corr)//2), len(cross_corr)//2 + 1)
        
        # Normalize cross-correlation
        cross_corr = cross_corr / np.max(np.abs(cross_corr))
        
        # Trim to max_lag
        center_idx = len(cross_corr) // 2
        start_idx = center_idx - max_lag
        end_idx = center_idx + max_lag + 1
        cross_corr = cross_corr[start_idx:end_idx]
        lags = lags[start_idx:end_idx]
        
        return lags, cross_corr
    except Exception as e:
        print(f"Error calculating cross-correlation: {str(e)}")
        raise

def perform_granger_causality(news_sentiment, reddit_sentiment, maxlag=5):
    """Perform Granger Causality test"""
    try:
        # Prepare data for Granger causality test
        data = pd.DataFrame({
            'news': news_sentiment,
            'reddit': reddit_sentiment
        }).dropna()
        
        # Test both directions
        results = {}
        
        # News -> Reddit
        results['news_to_reddit'] = grangercausalitytests(
            data[['reddit', 'news']], 
            maxlag=maxlag, 
            verbose=False
        )
        
        # Reddit -> News
        results['reddit_to_news'] = grangercausalitytests(
            data[['news', 'reddit']], 
            maxlag=maxlag, 
            verbose=False
        )
        
        return results
    except Exception as e:
        print(f"Error performing Granger causality test: {str(e)}")
        raise

def plot_sentiment_trends(df, stock_name):
    """Plot sentiment trends and stock price over time with enhanced visualization"""
    try:
        plt.figure(figsize=(15, 8))
        
        # Create gradient background
        ax = plt.gca()
        
        # Set color mapping
        cmap = plt.cm.tab10
        
        # Calculate EMA (span=20 for approximately one month of trading days)
        news_ema = calculate_ema(df['news'])
        reddit_ema = calculate_ema(df['reddit'])
        stock_ema = calculate_ema(df['stock'])
        
        # Calculate y-axis range with 5% margin for sentiment
        sentiment_values = pd.concat([df['news'], df['reddit']], axis=0)
        y_min_sentiment = max(min(sentiment_values.min(), -3), -3)  # Limit minimum to -3
        y_max_sentiment = min(max(sentiment_values.max(), 3), 3)    # Limit maximum to 3
        margin = (y_max_sentiment - y_min_sentiment) * 0.05
        y_min_sentiment -= margin
        y_max_sentiment += margin
        
        # Create gradient background
        colors_top = [(0.9, 1.0, 0.9)]  # Light green
        colors_bottom = [(1.0, 0.9, 0.9)]  # Light red
        gradient_cmap = LinearSegmentedColormap.from_list('custom', colors_top + colors_bottom)
        
        Z = np.linspace(1, -1, 100).reshape(-1, 1)
        extent = [df.index[0], df.index[-1], y_min_sentiment, y_max_sentiment]
        ax.imshow(Z, aspect='auto', extent=extent, cmap=gradient_cmap, alpha=0.3)
        
        # Draw zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.2, linewidth=1)
        
        # Plot original sentiment data (thinner lines)
        #plt.plot(df.index, df['news'], label='News', color=cmap(0), linewidth=0.8, alpha=0.3)
        #plt.plot(df.index, df['reddit'], label='Reddit', color=cmap(1), linewidth=0.8, alpha=0.3)
        
        # Plot EMA for sentiment (thicker lines)
        plt.plot(df.index, news_ema, label='News (EMA)', color=cmap(0), linewidth=2)
        plt.plot(df.index, reddit_ema, label='Reddit (EMA)', color=cmap(1), linewidth=2)
        
        # Create a second y-axis for stock price
        ax2 = ax.twinx()
        
        # Plot stock price
        #ax2.plot(df.index, df['stock'], label='Stock Price', color='green', linestyle='--', alpha=0.5)
        ax2.plot(df.index, stock_ema, label='Stock Price (EMA)', color='green', linewidth=2)
        
        # Set x-axis format
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=0)
        
        # Set y-axis format for sentiment
        ax.set_ylim(y_min_sentiment, y_max_sentiment)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))  # Major ticks at 1.0
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Minor ticks at 0.5
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.grid(True, which='minor', linestyle=':', alpha=0.1)
        
        # Set title and labels
        plt.title(f'Sentiment Trends and Stock Price for {stock_name}', pad=20)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Standardized Score')
        ax2.set_ylabel('Stock Price Standardized Score')
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9, fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = f'Hypo2_sentiment_analysis/sentiment_trends_{stock_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sentiment trends plot to {output_path}")
    except Exception as e:
        print(f"Error plotting sentiment trends: {str(e)}")
        raise

def plot_correlation_scatter(df, stock_name):
    """Plot correlation scatter plot with enhanced visualization"""
    try:
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with alpha for better visibility
        sns.scatterplot(data=df, x='news', y='reddit', alpha=0.6, color='blue')
        
        # Add correlation line
        z = np.polyfit(df['news'], df['reddit'], 1)
        p = np.poly1d(z)
        plt.plot(df['news'], p(df['news']), "r--", alpha=0.8, linewidth=2)
        
        # Add zero lines
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.2)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.2)
        
        # Set title and labels
        plt.title(f'Correlation between News and Reddit Sentiment\n{stock_name}', pad=20)
        plt.xlabel('News Sentiment')
        plt.ylabel('Reddit Sentiment')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.2)
        
        # Set axis limits with some padding
        x_min, x_max = df['news'].min(), df['news'].max()
        y_min, y_max = df['reddit'].min(), df['reddit'].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)
        
        plt.tight_layout()
        output_path = f'Hypo2_sentiment_analysis/correlation_scatter_{stock_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation scatter plot to {output_path}")
    except Exception as e:
        print(f"Error plotting correlation scatter: {str(e)}")
        raise

def plot_cross_correlation(lags, cross_corr, stock_name):
    """Plot cross-correlation results"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot cross-correlation
        plt.plot(lags, cross_corr, 'b-', linewidth=2)
        
        # Add zero line
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.2)
        
        # Set title and labels
        plt.title(f'Cross-Correlation between News and Reddit Sentiment\n{stock_name}', pad=20)
        plt.xlabel('Lag (days)')
        plt.ylabel('Cross-Correlation')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.2)
        
        # Set x-axis limits
        plt.xlim(-30, 30)
        
        plt.tight_layout()
        output_path = f'Hypo2_sentiment_analysis/cross_correlation_{stock_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cross-correlation plot to {output_path}")
    except Exception as e:
        print(f"Error plotting cross-correlation: {str(e)}")
        raise

def plot_correlation_summary(results):
    """Plot summary of correlation coefficients for all stocks"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Extract data
        stocks = [r['Stock'].lower() for r in results]
        correlations = [r['Pearson_Correlation'] for r in results]
        
        # Create bar plot
        bars = plt.bar(stocks, correlations, color='lightblue', alpha=0.7)
        
        # Add zero line
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add values on top of bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.05 if height > 0 else height - 0.1,
                    f'{corr:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        # Set title and labels
        plt.title('Correlation between News and Reddit Sentiment', pad=20)
        plt.xlabel('Stock')
        plt.ylabel('Pearson Correlation')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.2, axis='y')
        
        # Set y-axis limits with some padding
        plt.ylim(-0.6, 1.0)
        
        plt.tight_layout()
        output_path = 'Hypo2_sentiment_analysis/correlation_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation summary plot to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error plotting correlation summary: {str(e)}")
        raise

def find_lead_lag_relationship(results):
    """Find lead-lag relationship from Granger causality results"""
    lead_lag_data = []
    
    for result in results:
        stock = result['Stock'].lower()
        
        # Check if Reddit leads News (p-value < 0.05 for any lag)
        reddit_leads_news = False
        for lag in range(1, 6):
            p_value = result['Granger_Results']['reddit_to_news'][lag][0]['ssr_chi2test'][1]
            if p_value < 0.05:
                reddit_leads_news = True
                break
        
        # Check if News leads Reddit (p-value < 0.05 for any lag)
        news_leads_reddit = False
        for lag in range(1, 6):
            p_value = result['Granger_Results']['news_to_reddit'][lag][0]['ssr_chi2test'][1]
            if p_value < 0.05:
                news_leads_reddit = True
                break
        
        # Determine lead-lag value
        lead_lag_value = 0
        if reddit_leads_news and not news_leads_reddit:
            # Find the lag with the smallest p-value
            min_p_value = 1.0
            best_lag = 0
            for lag in range(1, 6):
                p_value = result['Granger_Results']['reddit_to_news'][lag][0]['ssr_chi2test'][1]
                if p_value < min_p_value:
                    min_p_value = p_value
                    best_lag = lag
            lead_lag_value = best_lag * 20  # Scale for visibility
        elif news_leads_reddit and not reddit_leads_news:
            # Find the lag with the smallest p-value
            min_p_value = 1.0
            best_lag = 0
            for lag in range(1, 6):
                p_value = result['Granger_Results']['news_to_reddit'][lag][0]['ssr_chi2test'][1]
                if p_value < min_p_value:
                    min_p_value = p_value
                    best_lag = lag
            lead_lag_value = -best_lag * 20  # Negative for news leading, scale for visibility
        
        lead_lag_data.append((stock, lead_lag_value))
    
    return lead_lag_data

def plot_lead_lag_summary(results):
    """Plot summary of lead-lag relationships for all stocks"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Extract data
        lead_lag_data = find_lead_lag_relationship(results)
        stocks = [item[0] for item in lead_lag_data]
        lead_lag_values = [item[1] for item in lead_lag_data]
        
        # Create bar plot
        bars = plt.bar(stocks, lead_lag_values, color='lightblue', alpha=0.7)
        
        # Add zero line
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Set title and labels
        plt.title('Lead-Lag Relationship (Negative: Reddit leads News)', pad=20)
        plt.xlabel('Stock')
        plt.ylabel('Lead-Lag Measure')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.2, axis='y')
        
        # Set y-axis limits
        max_val = max(abs(min(lead_lag_values)), abs(max(lead_lag_values))) if lead_lag_values else 100
        plt.ylim(-max_val*1.2, max_val*1.2)
        
        plt.tight_layout()
        output_path = 'Hypo2_sentiment_analysis/lead_lag_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved lead-lag summary plot to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error plotting lead-lag summary: {str(e)}")
        raise

def analyze_sentiment_stock_correlation(results):
    """Analyze correlation between sentiment and stock price"""
    sentiment_stock_results = []
    
    for result in results:
        stock_name = result['Stock']
        df = result['Data']
        
        # Ensure data is aligned and no NaN values
        valid_data = df.dropna()
        
        if len(valid_data) < 2:
            print(f"Not enough valid data points for {stock_name}")
            continue
            
        try:
            # Calculate correlation between news sentiment and stock price
            news_stock_corr, news_stock_p = stats.pearsonr(valid_data['news'], valid_data['stock'])
            
            # Calculate correlation between reddit sentiment and stock price
            reddit_stock_corr, reddit_stock_p = stats.pearsonr(valid_data['reddit'], valid_data['stock'])
            
            # Determine which sentiment has stronger correlation
            stronger_predictor = 'Reddit' if abs(reddit_stock_corr) > abs(news_stock_corr) else 'News'
            
            sentiment_stock_results.append({
                'Stock': stock_name,
                'News_Stock_Correlation': news_stock_corr,
                'News_Stock_P_value': news_stock_p,
                'Reddit_Stock_Correlation': reddit_stock_corr,
                'Reddit_Stock_P_value': reddit_stock_p,
                'Stronger_Predictor': stronger_predictor
            })
            
            print(f"{stock_name} - News-Stock Correlation: {news_stock_corr:.4f} (p={news_stock_p:.4f}), "
                  f"Reddit-Stock Correlation: {reddit_stock_corr:.4f} (p={reddit_stock_p:.4f}), "
                  f"Stronger predictor: {stronger_predictor}")
                  
        except Exception as e:
            print(f"Error calculating sentiment-stock correlation for {stock_name}: {str(e)}")
    
    return sentiment_stock_results

def plot_sentiment_stock_correlation(sentiment_stock_results):
    """Visualize correlation between sentiment and stock price"""
    try:
        plt.figure(figsize=(14, 8))
        
        # Extract data
        stocks = [r['Stock'] for r in sentiment_stock_results]
        news_correlations = [r['News_Stock_Correlation'] for r in sentiment_stock_results]
        reddit_correlations = [r['Reddit_Stock_Correlation'] for r in sentiment_stock_results]
        news_p_values = [r['News_Stock_P_value'] for r in sentiment_stock_results]
        reddit_p_values = [r['Reddit_Stock_P_value'] for r in sentiment_stock_results]
        
        # Set bar width
        barWidth = 0.35
        
        # Set X-axis positions
        r1 = np.arange(len(stocks))
        r2 = [x + barWidth for x in r1]
        
        # Create grouped bar chart
        news_bars = plt.bar(r1, news_correlations, color='#66B2FF', width=barWidth, edgecolor='grey', label='News-Stock')
        reddit_bars = plt.bar(r2, reddit_correlations, color='#FF9999', width=barWidth, edgecolor='grey', label='Reddit-Stock')
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.2)
        
        # Add labels
        plt.xlabel('Stock', fontweight='bold', fontsize=12)
        plt.ylabel('Correlation Coefficient', fontsize=12)
        plt.xticks([r + barWidth/2 for r in range(len(stocks))], stocks, fontsize=11)
        plt.title('Comparison of Sentiment-Stock Price Correlation', fontsize=14, pad=20)
        
        # Add legend
        plt.legend(fontsize=11)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.2, axis='y')
        
        # Add correlation values and significance markers
        for i, (corr, p_val) in enumerate(zip(news_correlations, news_p_values)):
            color = 'white' if abs(corr) > 0.3 else 'black'
            sig_mark = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')
            plt.text(i, corr/2, f'{corr:.2f}{sig_mark}', ha='center', va='center', 
                     color=color, fontweight='bold', fontsize=10)
        
        for i, (corr, p_val) in enumerate(zip(reddit_correlations, reddit_p_values)):
            color = 'white' if abs(corr) > 0.3 else 'black'
            sig_mark = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')
            plt.text(i + barWidth, corr/2, f'{corr:.2f}{sig_mark}', ha='center', va='center', 
                     color=color, fontweight='bold', fontsize=10)
        
        # Add significance explanation
        plt.figtext(0.01, 0.01, '* p < 0.05, ** p < 0.01', fontsize=9)
        
        plt.tight_layout()
        output_path = 'Hypo2_sentiment_analysis/sentiment_stock_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sentiment-stock correlation plot to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error plotting sentiment-stock correlation: {str(e)}")
        raise

def plot_predictive_power_summary(sentiment_stock_results):
    """Visualize which sentiment source has stronger predictive power"""
    try:
        # Count which sentiment source is stronger for each stock
        reddit_stronger = sum(1 for r in sentiment_stock_results if r['Stronger_Predictor'] == 'Reddit')
        news_stronger = sum(1 for r in sentiment_stock_results if r['Stronger_Predictor'] == 'News')
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        labels = ['Reddit Stronger', 'News Stronger']
        sizes = [reddit_stronger, news_stronger]
        colors = ['#FF9999', '#66B2FF']
        explode = (0.1, 0)  # Explode Reddit slice
        
        # Add percentage and count labels
        autopct = lambda p: f'{p:.1f}%\n({int(p*sum(sizes)/100)} stocks)'
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=autopct,
                shadow=True, startangle=90, textprops={'fontsize': 12})
        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        plt.title('Comparison of Predictive Power by Sentiment Source\n(Which source has stronger correlation with more stocks)', fontsize=14, pad=20)
        
        # Add annotation
        plt.annotate('Based on contemporaneous correlation analysis', xy=(0.5, -0.1), xycoords='figure fraction', 
                    ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        output_path = 'Hypo2_sentiment_analysis/predictive_power_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved predictive power summary plot to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error plotting predictive power summary: {str(e)}")
        raise

def analyze_lagged_correlation(results, max_lag=5):
    """Analyze sentiment and future stock price lagged correlation, assess predictive power"""
    lagged_results = []
    
    for result in results:
        stock_name = result['Stock']
        df = result['Data'].copy()
        
        if len(df) <= max_lag:
            print(f"Not enough data points for lagged analysis of {stock_name}")
            continue
            
        try:
            # Create dictionary to store different lag period correlations
            lagged_correlations = {'News': [], 'Reddit': []}
            lagged_p_values = {'News': [], 'Reddit': []}
            
            # Calculate correlations for different lag periods
            for lag in range(max_lag + 1):
                # Create lagged stock data (sentiment predicts future stock price)
                if lag > 0:
                    df[f'stock_lag_{lag}'] = df['stock'].shift(-lag)
                else:
                    df[f'stock_lag_{lag}'] = df['stock']
                
                # Calculate news sentiment and lagged stock correlation
                mask = ~(np.isnan(df['news']) | np.isnan(df[f'stock_lag_{lag}']))
                if sum(mask) > 2:
                    news_corr, news_p = stats.pearsonr(df.loc[mask, 'news'], df.loc[mask, f'stock_lag_{lag}'])
                    lagged_correlations['News'].append(news_corr)
                    lagged_p_values['News'].append(news_p)
                else:
                    lagged_correlations['News'].append(np.nan)
                    lagged_p_values['News'].append(np.nan)
                
                # Calculate Reddit sentiment and lagged stock correlation
                mask = ~(np.isnan(df['reddit']) | np.isnan(df[f'stock_lag_{lag}']))
                if sum(mask) > 2:
                    reddit_corr, reddit_p = stats.pearsonr(df.loc[mask, 'reddit'], df.loc[mask, f'stock_lag_{lag}'])
                    lagged_correlations['Reddit'].append(reddit_corr)
                    lagged_p_values['Reddit'].append(reddit_p)
                else:
                    lagged_correlations['Reddit'].append(np.nan)
                    lagged_p_values['Reddit'].append(np.nan)
            
            # Find maximum correlation for each sentiment source and its corresponding lag
            news_max_corr = max(lagged_correlations['News'], key=abs) if lagged_correlations['News'] else np.nan
            news_max_lag = lagged_correlations['News'].index(news_max_corr) if not np.isnan(news_max_corr) else np.nan
            news_max_p = lagged_p_values['News'][news_max_lag] if not np.isnan(news_max_lag) else np.nan
            
            reddit_max_corr = max(lagged_correlations['Reddit'], key=abs) if lagged_correlations['Reddit'] else np.nan
            reddit_max_lag = lagged_correlations['Reddit'].index(reddit_max_corr) if not np.isnan(reddit_max_corr) else np.nan
            reddit_max_p = lagged_p_values['Reddit'][reddit_max_lag] if not np.isnan(reddit_max_lag) else np.nan
            
            # Determine which sentiment source has stronger predictive power
            if np.isnan(news_max_corr) or np.isnan(reddit_max_corr):
                stronger_predictor = 'Undetermined'
            else:
                stronger_predictor = 'Reddit' if abs(reddit_max_corr) > abs(news_max_corr) else 'News'
            
            # Store result
            lagged_result = {
                'Stock': stock_name,
                'News_Lagged_Correlations': lagged_correlations['News'],
                'News_Lagged_P_Values': lagged_p_values['News'],
                'News_Max_Correlation': news_max_corr,
                'News_Max_Lag': news_max_lag,
                'News_Max_P_Value': news_max_p,
                'Reddit_Lagged_Correlations': lagged_correlations['Reddit'],
                'Reddit_Lagged_P_Values': lagged_p_values['Reddit'],
                'Reddit_Max_Correlation': reddit_max_corr,
                'Reddit_Max_Lag': reddit_max_lag,
                'Reddit_Max_P_Value': reddit_max_p,
                'Stronger_Predictor': stronger_predictor
            }
            
            lagged_results.append(lagged_result)
            
            # Print result
            print(f"{stock_name} - News max correlation: {news_max_corr:.4f} at lag {news_max_lag} (p={news_max_p:.4f}), "
                  f"Reddit max correlation: {reddit_max_corr:.4f} at lag {reddit_max_lag} (p={reddit_max_p:.4f}), "
                  f"Stronger predictor: {stronger_predictor}")
                  
        except Exception as e:
            print(f"Error calculating lagged correlation for {stock_name}: {str(e)}")
    
    return lagged_results

def plot_lagged_correlation(lagged_results, stock_name):
    """Plot lagged correlation for a specific stock"""
    try:
        # Find the result for the specified stock
        result = next((r for r in lagged_results if r['Stock'] == stock_name), None)
        if not result:
            print(f"No lagged correlation results found for {stock_name}")
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Extract data
        lags = list(range(len(result['News_Lagged_Correlations'])))
        news_correlations = result['News_Lagged_Correlations']
        reddit_correlations = result['Reddit_Lagged_Correlations']
        
        # Plot correlations
        plt.plot(lags, news_correlations, 'b-', marker='o', label='News-Stock')
        plt.plot(lags, reddit_correlations, 'r-', marker='o', label='Reddit-Stock')
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.2)
        
        # Add vertical line at lag 0
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add labels
        plt.xlabel('Lag (days)', fontweight='bold')
        plt.ylabel('Correlation Coefficient')
        plt.title(f'Lagged Correlation between Sentiment and Stock Price\n{stock_name}')
        
        # Add legend
        plt.legend()
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.2)
        
        # Set x-axis ticks
        plt.xticks(lags)
        
        # Add annotation for maximum correlation
        news_max_corr = result['News_Max_Correlation']
        news_max_lag = result['News_Max_Lag']
        reddit_max_corr = result['Reddit_Max_Correlation']
        reddit_max_lag = result['Reddit_Max_Lag']
        
        plt.annotate(f'Max: {news_max_corr:.2f}', 
                     xy=(news_max_lag, news_max_corr),
                     xytext=(news_max_lag, news_max_corr + 0.1),
                     arrowprops=dict(facecolor='blue', shrink=0.05),
                     ha='center')
                     
        plt.annotate(f'Max: {reddit_max_corr:.2f}', 
                     xy=(reddit_max_lag, reddit_max_corr),
                     xytext=(reddit_max_lag, reddit_max_corr - 0.1),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     ha='center')
        
        plt.tight_layout()
        output_path = f'Hypo2_sentiment_analysis/lagged_correlation_{stock_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved lagged correlation plot to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error plotting lagged correlation: {str(e)}")
        raise

def plot_lagged_correlation_summary(lagged_results):
    """Visualize summary of lagged correlation analysis results"""
    try:
        plt.figure(figsize=(14, 8))
        
        # Extract data
        stocks = [r['Stock'] for r in lagged_results]
        news_max_corrs = [r['News_Max_Correlation'] for r in lagged_results]
        reddit_max_corrs = [r['Reddit_Max_Correlation'] for r in lagged_results]
        news_max_lags = [r['News_Max_Lag'] for r in lagged_results]
        reddit_max_lags = [r['Reddit_Max_Lag'] for r in lagged_results]
        news_max_p_values = [r['News_Max_P_Value'] for r in lagged_results]
        reddit_max_p_values = [r['Reddit_Max_P_Value'] for r in lagged_results]
        
        # Set bar width
        barWidth = 0.35
        
        # Set X-axis positions
        r1 = np.arange(len(stocks))
        r2 = [x + barWidth for x in r1]
        
        # Create grouped bar chart
        news_bars = plt.bar(r1, news_max_corrs, color='#66B2FF', width=barWidth, edgecolor='grey', label='News-Stock')
        reddit_bars = plt.bar(r2, reddit_max_corrs, color='#FF9999', width=barWidth, edgecolor='grey', label='Reddit-Stock')
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.2)
        
        # Add labels
        plt.xlabel('Stock', fontweight='bold', fontsize=12)
        plt.ylabel('Maximum Correlation Coefficient', fontsize=12)
        plt.xticks([r + barWidth/2 for r in range(len(stocks))], stocks, fontsize=11)
        plt.title('Maximum Lagged Correlation between Sentiment and Stock Price', fontsize=14, pad=20)
        
        # Add legend
        plt.legend(fontsize=11)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.2, axis='y')
        
        # Add correlation values, lag periods and significance markers
        for i, (corr, lag, p_val) in enumerate(zip(news_max_corrs, news_max_lags, news_max_p_values)):
            color = 'white' if abs(corr) > 0.3 else 'black'
            sig_mark = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')
            plt.text(i, corr/2, f'{corr:.2f}{sig_mark}\nLag {lag} days', ha='center', va='center', 
                     color=color, fontweight='bold', fontsize=9)
        
        for i, (corr, lag, p_val) in enumerate(zip(reddit_max_corrs, reddit_max_lags, reddit_max_p_values)):
            color = 'white' if abs(corr) > 0.3 else 'black'
            sig_mark = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')
            plt.text(i + barWidth, corr/2, f'{corr:.2f}{sig_mark}\nLag {lag} days', ha='center', va='center', 
                     color=color, fontweight='bold', fontsize=9)
        
        # Add significance explanation
        plt.figtext(0.01, 0.01, '* p < 0.05, ** p < 0.01', fontsize=9)
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 'Lag days indicate how many days sentiment leads stock price, 0 means contemporaneous correlation', ha='center', fontsize=10)
        
        plt.tight_layout()
        output_path = 'Hypo2_sentiment_analysis/lagged_correlation_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved lagged correlation summary plot to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error plotting lagged correlation summary: {str(e)}")
        raise

def main():
    """主函数，执行整个分析流程"""
    import traceback
    
    print("开始分析情绪对股价预测能力的研究")
    
    # 创建结果目录
    os.makedirs('Hypo2_sentiment_analysis', exist_ok=True)
    
    # 要分析的股票列表
    datasets = ['AAPL', 'META', 'TSLA', 'NVDA', 'PLTR', 'SPY']
    
    # 存储每只股票的分析结果
    results = []
    sentiment_stock_results = []
    
    # 处理每只股票的数据
    for stock in datasets:
        try:
            file_path = f'dataset/6datasets-2024-2025/{stock.lower()}_compare.pkl'
            print(f"\n处理{stock}数据，文件路径：{file_path}")
            
            # 加载数据
            df = load_data(file_path)
            if df is None:
                continue
                
            print(f"DataFrame形状: {df.shape}")
            print(df.head())
            
            # 计算新闻和Reddit情绪之间的相关性
            news_sentiment = df['news']
            reddit_sentiment = df['reddit']
            corr, p_value = calculate_correlation(news_sentiment, reddit_sentiment)
            print(f"新闻-Reddit相关性: {corr:.4f}, P值: {p_value:.4f}")
            
            # 绘制情绪趋势图
            sentiment_trends_path = plot_sentiment_trends(df, stock)
            
            # 存储结果
            result = {
                'Stock': stock,
                'Data': df,
                'News_Reddit_Correlation': corr,
                'News_Reddit_P_value': p_value,
                'Plots': {
                    'Sentiment_Trends': sentiment_trends_path
                }
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"处理{stock}时出错: {str(e)}")
            traceback.print_exc()
    
    # 分析情绪与股价的相关性
    try:
        print("\n分析情绪与股价的相关性...")
        sentiment_stock_results = analyze_sentiment_stock_correlation(results)
        sentiment_stock_correlation_path = plot_sentiment_stock_correlation(sentiment_stock_results)
        predictive_power_summary_path = plot_predictive_power_summary(sentiment_stock_results)
        
        # 分析滞后相关性
        print("\n分析滞后相关性...")
        lagged_results = analyze_lagged_correlation(results)
        
        # 为每只股票绘制滞后相关性图
        lagged_correlation_paths = {}
        for stock in datasets:
            path = plot_lagged_correlation(lagged_results, stock)
            if path:
                lagged_correlation_paths[stock] = path
        
        # 绘制滞后相关性总结图
        lagged_correlation_summary_path = plot_lagged_correlation_summary(lagged_results)
        
        # 生成总结报告
        with open('Hypo2_sentiment_analysis/Hypo2_summary.txt', 'w', encoding='utf-8') as f:
            f.write("# 预测能力假设：情绪分析与股价走势\n\n")
            
            f.write("## 研究问题\n")
            f.write("传统新闻媒体和Reddit的情绪趋势是否对股价走势具有预测能力？\n")
            f.write("如果有，哪种情绪来源与股价走势的相关性更强？\n\n")
            
            f.write("## 假设\n")
            f.write("H0: 传统新闻媒体和Reddit的情绪无法显著预测股价走势。\n")
            f.write("H1: Reddit情绪与短期股价走势的相关性强于传统新闻媒体。\n\n")
            
            f.write("## 统计方法\n")
            f.write("1. 情绪-股价相关性分析：测量情绪与股价之间的关系\n")
            f.write("2. 滞后相关性分析：评估情绪对未来股价走势的预测能力\n\n")
            
            f.write("## 结果\n\n")
            
            # 情绪-股价相关性结果
            f.write("### 情绪-股价相关性\n\n")
            
            reddit_stronger_count = 0
            news_stronger_count = 0
            
            for result in sentiment_stock_results:
                stock = result['Stock']
                news_corr = result['News_Stock_Correlation']
                news_p = result['News_Stock_P_value']
                reddit_corr = result['Reddit_Stock_Correlation']
                reddit_p = result['Reddit_Stock_P_value']
                stronger = result['Stronger_Predictor']
                
                if stronger == 'Reddit':
                    reddit_stronger_count += 1
                else:
                    news_stronger_count += 1
                
                f.write(f"#### {stock}\n")
                f.write(f"- 新闻-股价相关性: {news_corr:.4f} (p值: {news_p:.4f})\n")
                f.write(f"- Reddit-股价相关性: {reddit_corr:.4f} (p值: {reddit_p:.4f})\n")
                f.write(f"- 更强预测因子: {stronger}\n\n")
            
            # 滞后相关性结果
            f.write("### 滞后相关性分析\n\n")
            
            lagged_reddit_stronger_count = 0
            lagged_news_stronger_count = 0
            
            for result in lagged_results:
                stock = result['Stock']
                news_max_corr = result['News_Max_Correlation']
                news_max_lag = result['News_Max_Lag']
                news_max_p = result['News_Max_P_Value']
                reddit_max_corr = result['Reddit_Max_Correlation']
                reddit_max_lag = result['Reddit_Max_Lag']
                reddit_max_p = result['Reddit_Max_P_Value']
                stronger = result['Stronger_Predictor']
                
                if stronger == 'Reddit':
                    lagged_reddit_stronger_count += 1
                elif stronger == 'News':
                    lagged_news_stronger_count += 1
                
                f.write(f"#### {stock}\n")
                f.write(f"- 新闻最大相关性: {news_max_corr:.4f} 在滞后期 {news_max_lag} (p值: {news_max_p:.4f})\n")
                f.write(f"- Reddit最大相关性: {reddit_max_corr:.4f} 在滞后期 {reddit_max_lag} (p值: {reddit_max_p:.4f})\n")
                f.write(f"- 更强预测因子: {stronger}\n\n")
            
            # 发现总结
            f.write("## 发现总结\n\n")
            f.write(f"### 同期相关性\n")
            f.write(f"- Reddit相关性更强的股票数: {reddit_stronger_count}\n")
            f.write(f"- 新闻相关性更强的股票数: {news_stronger_count}\n\n")
            
            f.write(f"### 滞后相关性（预测能力）\n")
            f.write(f"- Reddit预测能力更强的股票数: {lagged_reddit_stronger_count}\n")
            f.write(f"- 新闻预测能力更强的股票数: {lagged_news_stronger_count}\n\n")
            
            # 结论
            f.write("## 结论\n\n")
            if reddit_stronger_count > news_stronger_count and lagged_reddit_stronger_count > lagged_news_stronger_count:
                f.write("分析支持备择假设(H1): Reddit情绪与股价走势的相关性强于传统新闻媒体。\n\n")
            elif news_stronger_count > reddit_stronger_count and lagged_news_stronger_count > lagged_reddit_stronger_count:
                f.write("分析不支持备择假设(H1): 传统新闻媒体情绪与股价走势的相关性强于Reddit情绪。\n\n")
            else:
                f.write("分析结果显示，关于哪种情绪来源与股价走势相关性更强的结论不一致。\n\n")
            
            # 可视化
            f.write("## 可视化\n\n")
            f.write("### 个股可视化\n")
            for stock in datasets:
                f.write(f"- {stock}:\n")
                f.write(f"  - 情绪趋势: Hypo2_sentiment_analysis/sentiment_trends_{stock}.png\n")
                if stock in lagged_correlation_paths:
                    f.write(f"  - 滞后相关性: {lagged_correlation_paths[stock]}\n")
            
            f.write("\n### 总结可视化\n")
            f.write(f"- 情绪-股价相关性: {sentiment_stock_correlation_path}\n")
            f.write(f"- 预测能力总结: {predictive_power_summary_path}\n")
            f.write(f"- 滞后相关性总结: {lagged_correlation_summary_path}\n")
        
        print("\n分析完成。总结已保存至 Hypo2_sentiment_analysis/Hypo2_summary.txt")
    
    except Exception as e:
        print(f"生成总结时出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 