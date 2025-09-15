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
    """Load and validate the preprocessed data"""
    print(f"Loading data from {file_path}")
    try:
        df = pd.read_pickle(file_path)
        print("Data loaded successfully")
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        print("First few rows:\n", df.head())
        
        # Validate required columns
        required_columns = ['news', 'reddit', 'stock']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
        raise

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
    """Plot sentiment trends over time with enhanced visualization"""
    try:
        plt.figure(figsize=(15, 8))
        
        # Create gradient background
        ax = plt.gca()
        
        # Set color mapping
        cmap = plt.cm.tab10
        
        # Calculate EMA (span=20 for approximately one month of trading days)
        news_ema = calculate_ema(df['news'])
        reddit_ema = calculate_ema(df['reddit'])
        
        # Calculate y-axis range with 5% margin
        all_values = pd.concat([df['news'], df['reddit']], axis=0)
        y_min = max(min(all_values.min(), -3), -3)  # Limit minimum to -3
        y_max = min(max(all_values.max(), 3), 3)    # Limit maximum to 3
        margin = (y_max - y_min) * 0.05
        y_min -= margin
        y_max += margin
        
        # Create gradient background
        colors_top = [(0.9, 1.0, 0.9)]  # Light green
        colors_bottom = [(1.0, 0.9, 0.9)]  # Light red
        gradient_cmap = LinearSegmentedColormap.from_list('custom', colors_top + colors_bottom)
        
        Z = np.linspace(1, -1, 100).reshape(-1, 1)
        extent = [df.index[0], df.index[-1], y_min, y_max]
        ax.imshow(Z, aspect='auto', extent=extent, cmap=gradient_cmap, alpha=0.3)
        
        # Draw zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.2, linewidth=1)
        
        # Plot original data (thinner lines)
        #plt.plot(df.index, df['news'], label='News', color=cmap(0), linewidth=0.8, alpha=0.6)
        #plt.plot(df.index, df['reddit'], label='Reddit', color=cmap(1), linewidth=0.8, alpha=0.6)
        
        # Plot EMA (thicker lines)
        plt.plot(df.index, news_ema, label='News (EMA)', color=cmap(0), linewidth=1.5)
        plt.plot(df.index, reddit_ema, label='Reddit (EMA)', color=cmap(1), linewidth=1.5)
        
        # Set x-axis format
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=0)
        
        # Set y-axis format
        plt.ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))  # Major ticks at 1.0
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Minor ticks at 0.5
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.grid(True, which='minor', linestyle=':', alpha=0.1)
        
        # Set title and labels
        plt.title(f'Sentiment Trends for {stock_name}', pad=20)
        plt.xlabel('Date')
        plt.ylabel('Standardized Score')
        
        # Adjust legend
        plt.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = f'Hypo1_sentiment_analysis/sentiment_trends_{stock_name}.png'
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
        output_path = f'Hypo1_sentiment_analysis/correlation_scatter_{stock_name}.png'
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
        output_path = f'Hypo1_sentiment_analysis/cross_correlation_{stock_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cross-correlation plot to {output_path}")
    except Exception as e:
        print(f"Error plotting cross-correlation: {str(e)}")
        raise

def main():
    print("Starting sentiment analysis")
    
    # Create results directory
    os.makedirs('Hypo1_sentiment_analysis', exist_ok=True)
    
    # List of datasets to analyze
    datasets = ['aapl', 'meta', 'tsla', 'nvda', 'pltr', 'spy']
    results = []
    
    # Process each stock
    for stock in datasets:
        print(f"\nProcessing {stock.upper()}")
        try:
            # Load and process data
            file_path = f'dataset/6datasets-2024-2025/{stock}_compare.pkl'
            df = load_data(file_path)
            
            # Calculate Pearson correlation
            correlation, p_value = calculate_correlation(df['news'], df['reddit'])
            
            # Calculate cross-correlation
            lags, cross_corr = calculate_cross_correlation(df['news'], df['reddit'])
            
            # Perform Granger causality test
            granger_results = perform_granger_causality(df['news'], df['reddit'])
            
            # Generate plots
            plot_sentiment_trends(df, stock.upper())
            plot_correlation_scatter(df, stock.upper())
            plot_cross_correlation(lags, cross_corr, stock.upper())
            
            # Store results
            results.append({
                'Stock': stock.upper(),
                'Pearson_Correlation': correlation,
                'P_value': p_value,
                'Sample_Size': len(df),
                'Cross_Correlation': cross_corr,
                'Granger_Results': granger_results,
                'Conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
            })
            
        except Exception as e:
            print(f"Failed to process {stock.upper()}: {str(e)}")
            continue
    
    # Generate summary report
    if results:
        print("\nGenerating summary report")
        with open('Hypo1_sentiment_analysis/Hypo1_summary.txt', 'w') as f:
            f.write('Hypothesis 1: Sentiment Correlation Analysis\n')
            f.write('==========================================\n\n')
            
            f.write('Research Question:\n')
            f.write('Do sentiment trends between legacy news media and Reddit show significant correlation?\n\n')
            
            f.write('Null Hypothesis (H0): Sentiment trends between legacy news media and Reddit does not have any significant correlation.\n')
            f.write('Alternative Hypothesis (H1): Sentiment trends between legacy news media and Reddit have a significant correlation.\n\n')
            
            f.write('Statistical Methods:\n')
            f.write('-------------------\n')
            f.write('1. Pearson Correlation Analysis: To measure the linear relationship between news and Reddit sentiment\n')
            f.write('2. Cross-Correlation Analysis: To identify lead-lag relationships between the two sentiment series\n')
            f.write('3. Granger Causality Tests: To determine if one sentiment series helps predict the other\n\n')
            
            f.write('Results:\n')
            f.write('--------\n')
            
            # Sort results by correlation strength
            results.sort(key=lambda x: abs(x['Pearson_Correlation']), reverse=True)
            
            for result in results:
                f.write(f"\nStock: {result['Stock']}\n")
                f.write(f"Sample Size: {result['Sample_Size']}\n")
                f.write(f"Pearson Correlation Coefficient: {result['Pearson_Correlation']:.4f}\n")
                f.write(f"P-value: {result['P_value']:.4f}\n")
                
                # Add cross-correlation results
                max_cross_corr = np.max(np.abs(result['Cross_Correlation']))
                f.write(f"Maximum Cross-Correlation: {max_cross_corr:.4f}\n")
                
                # Add Granger causality results
                f.write("\nGranger Causality Results:\n")
                for direction, granger_result in result['Granger_Results'].items():
                    f.write(f"{direction}:\n")
                    for lag in range(1, 6):
                        p_value = granger_result[lag][0]['ssr_chi2test'][1]
                        f.write(f"  Lag {lag}: p-value = {p_value:.4f}\n")
                
                f.write(f"Conclusion: {result['Conclusion']}\n")
                f.write('-' * 50 + '\n')
            
            # Overall summary
            significant_correlations = sum(1 for r in results if r['P_value'] < 0.05)
            f.write(f"\nOverall Summary:\n")
            f.write(f"Total stocks analyzed: {len(results)}\n")
            f.write(f"Stocks with significant correlation: {significant_correlations}\n")
            
            f.write('\nVisualization Files:\n')
            f.write('-------------------\n')
            for stock in [r['Stock'] for r in results]:
                f.write(f"1. sentiment_trends_{stock}.png - Time series plot of sentiment trends\n")
                f.write(f"2. correlation_scatter_{stock}.png - Correlation scatter plot\n")
                f.write(f"3. cross_correlation_{stock}.png - Cross-correlation analysis plot\n")
    
    print("Analysis completed")

if __name__ == "__main__":
    main() 