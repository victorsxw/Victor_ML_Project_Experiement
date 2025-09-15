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

def load_data(file_path, stock_name):
    """Load and validate the preprocessed data"""
    print(f"Loading data from {file_path}")
    try:
        # Try to load the CSV file
        df = pd.read_csv(file_path, index_col=0)
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        print("Data loaded successfully")
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        print("First few rows:\n", df.head())
        
        # Extract the required columns using the new format
        news_col = f"news_{stock_name}_SMA1_scaled"
        reddit_col = f"reddit_{stock_name}_SMA1_scaled"
        stock_col = f"stock_{stock_name}_SMA1_scaled"
        
        # Check if required columns exist
        missing_columns = []
        for col in [news_col, reddit_col, stock_col]:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"Warning: Missing columns for {stock_name}: {missing_columns}")
            print("Available columns:", df.columns.tolist())
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create a new DataFrame with the renamed columns
        result_df = pd.DataFrame({
            'news': df[news_col],
            'reddit': df[reddit_col],
            'stock': df[stock_col]
        }, index=df.index)
        
        # Check for NaN values and report
        nan_counts = result_df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"Warning: NaN values detected in {stock_name} data:")
            print(nan_counts)
            print(f"Rows with NaN: {result_df.isna().any(axis=1).sum()} out of {len(result_df)}")
            
            # Fill NaN values with 0 or drop rows with NaN
            # Here we choose to drop rows with NaN
            result_df = result_df.dropna()
            print(f"After dropping NaN rows: {len(result_df)} rows remaining")
            
            # If too few rows remain, raise an error
            if len(result_df) < 30:  # Set a minimum threshold
                raise ValueError(f"Too few valid data points ({len(result_df)}) after removing NaN values")
        
        print("Extracted columns:")
        print(f"  news: {news_col}")
        print(f"  reddit: {reddit_col}")
        print(f"  stock: {stock_col}")
        
        return result_df
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
        
        # Check if we have enough data
        if len(data) < maxlag + 2:
            print(f"Warning: Not enough data points for Granger causality test (need at least {maxlag + 2}, have {len(data)})")
            return {"news_to_reddit": None, "reddit_to_news": None}
        
        # Test both directions
        results = {}
        
        try:
            # News -> Reddit
            results['news_to_reddit'] = grangercausalitytests(
                data[['reddit', 'news']], 
                maxlag=maxlag, 
                verbose=False
            )
        except Exception as e:
            print(f"Error in news_to_reddit Granger test: {str(e)}")
            results['news_to_reddit'] = None
            
        try:
            # Reddit -> News
            results['reddit_to_news'] = grangercausalitytests(
                data[['news', 'reddit']], 
                maxlag=maxlag, 
                verbose=False
            )
        except Exception as e:
            print(f"Error in reddit_to_news Granger test: {str(e)}")
            results['reddit_to_news'] = None
            
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
        
        # Calculate EMA (span=2 for approximately one month of trading days)
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

def plot_correlation_summary(results):
    """Plot summary of correlations between news and Reddit sentiment for all stocks"""
    try:
        plt.figure(figsize=(15, 8))
        
        # Extract correlation values and stock names
        stocks = [r['Stock'].lower() for r in results]
        correlations = [r['Pearson_Correlation'] for r in results]
        
        # Create bar plot
        bars = plt.bar(stocks, correlations, color='lightblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top')
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        
        # Customize plot
        plt.title('Correlation between News and Reddit Sentiment', fontsize=14)
        plt.xlabel('Stock', fontsize=12)
        plt.ylabel('Pearson Correlation', fontsize=12)
        plt.ylim(-0.6, 1.0)  # Set y-axis limits to match the example
        plt.grid(True, linestyle='--', alpha=0.2)
        
        # Save plot
        output_path = 'Hypo1_sentiment_analysis/correlation_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation summary plot to {output_path}")
    except Exception as e:
        print(f"Error plotting correlation summary: {str(e)}")
        raise

def plot_lead_lag_summary(results):
    """Plot summary of lead-lag relationships between news and Reddit sentiment"""
    try:
        plt.figure(figsize=(15, 8))
        
        # Extract lead-lag measures
        stocks = [r['Stock'].lower() for r in results]
        lead_lag_measures = []
        
        for result in results:
            granger_results = result['Granger_Results']
            lead_lag_measure = 0
            
            # If Reddit leads News (reddit_to_news) and p-value is significant
            if granger_results['reddit_to_news'] is not None:
                p_value = granger_results['reddit_to_news'][1][0]['ssr_chi2test'][1]
                if p_value < 0.05:
                    # Stronger measure for more significant results (lower p-value)
                    lead_lag_measure = 40 * (1 - p_value)
            
            lead_lag_measures.append(lead_lag_measure)
        
        # Create bar plot
        bars = plt.bar(stocks, lead_lag_measures, color='lightblue')
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        
        # Customize plot
        plt.title('Lead-Lag Relationship (Negative: Reddit leads News)', fontsize=14)
        plt.xlabel('Stock', fontsize=12)
        plt.ylabel('Lead-Lag Measure', fontsize=12)
        plt.ylim(-50, 50)  # Set y-axis limits to match the example
        plt.grid(True, linestyle='--', alpha=0.2)
        
        # Save plot
        output_path = 'Hypo1_sentiment_analysis/lead_lag_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved lead-lag summary plot to {output_path}")
    except Exception as e:
        print(f"Error plotting lead-lag summary: {str(e)}")
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
            file_path = f'dataset/20250315/merged/{stock}_merged_data.csv'
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
            
            # Check file size to ensure it's not empty or corrupted
            file_size = os.path.getsize(file_path)
            if file_size < 1000:  # Arbitrarily small size
                print(f"Warning: File {file_path} is too small ({file_size} bytes). Skipping.")
                continue
                
            df = load_data(file_path, stock)
            
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
            result_entry = {
                'Stock': stock.upper(),
                'Pearson_Correlation': correlation,
                'P_value': p_value,
                'Sample_Size': len(df),
                'Cross_Correlation': cross_corr,
                'Granger_Results': granger_results,
                'Conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
            }
            results.append(result_entry)
            print(f"Successfully processed {stock.upper()}")
            
        except Exception as e:
            print(f"Failed to process {stock.upper()}: {str(e)}")
            # Print detailed error information
            import traceback
            print(f"Detailed error for {stock.upper()}:")
            traceback.print_exc()
            continue
    
    # Generate summary report
    if results:
        # Generate summary plots
        plot_correlation_summary(results)
        plot_lead_lag_summary(results)
        
        print("\nGenerating summary report")
        with open('Hypo1_sentiment_analysis/Hypo1_summary.txt', 'w') as f:
            f.write('Hypothesis 1: Sentiment Correlation Analysis\n')
            f.write('==========================================\n\n')
            
            f.write('Research Question:\n')
            f.write('Do sentiment trends between legacy news media and Reddit show significant correlation? Does Reddit sentiment lead news media sentiment?\n\n')
            
            f.write('Null Hypothesis (H0): Sentiment trends between legacy news media and Reddit does not have any significant correlation.\n')
            f.write('Alternative Hypothesis (H1): Sentiment trends between legacy news media and Reddit have a significant correlation. Also, Reddit sentiment has a faster respond and leads the legacy news media sentiment.\n\n')
            
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
                granger_results = result['Granger_Results']
                
                if granger_results['news_to_reddit'] is None and granger_results['reddit_to_news'] is None:
                    f.write("  Granger causality test could not be performed due to data issues.\n")
                else:
                    for direction, granger_result in granger_results.items():
                        if granger_result is None:
                            f.write(f"{direction}: Could not perform test\n")
                            continue
                            
                        f.write(f"{direction}:\n")
                        for lag in range(1, 6):
                            p_value = granger_result[lag][0]['ssr_chi2test'][1]
                            f.write(f"  Lag {lag}: p-value = {p_value:.4f}\n")
                            
                        # Check if Reddit leads News with any significant lag
                        if direction == 'reddit_to_news':
                            significant_lags = [lag for lag in range(1, 6) if granger_result[lag][0]['ssr_chi2test'][1] < 0.05]
                            if significant_lags:
                                f.write(f"  Significant lags where Reddit leads News: {', '.join(map(str, significant_lags))}\n")
                
                f.write(f"Conclusion: {result['Conclusion']}\n")
                f.write('-' * 50 + '\n')
            
            # Overall summary
            significant_correlations = sum(1 for r in results if r['P_value'] < 0.05)
            reddit_leading_stocks = []
            for r in results:
                if r['Granger_Results']['reddit_to_news'] is not None:
                    for lag in range(1, 6):
                        if r['Granger_Results']['reddit_to_news'][lag][0]['ssr_chi2test'][1] < 0.05:
                            reddit_leading_stocks.append(r['Stock'])
                            break
            
            f.write(f"\nOverall Summary:\n")
            f.write(f"Total stocks analyzed: {len(results)}\n")
            f.write(f"Stocks with significant correlation: {significant_correlations}\n")
            f.write(f"Stocks where Reddit sentiment leads News sentiment: {len(reddit_leading_stocks)}\n")
            if reddit_leading_stocks:
                f.write(f"  - {', '.join(reddit_leading_stocks)}\n")
            
            # List stocks that could not be analyzed
            missing_stocks = set(s.upper() for s in datasets) - set(r['Stock'] for r in results)
            if missing_stocks:
                f.write(f"\nStocks that could not be analyzed due to data issues: {', '.join(missing_stocks)}\n")
            
            f.write('\nVisualization Files:\n')
            f.write('-------------------\n')
            f.write('1. correlation_summary.png - Summary of correlation coefficients for all stocks\n')
            f.write('2. lead_lag_summary.png - Summary of lead-lag relationships for all stocks\n\n')
            
            for stock in [r['Stock'] for r in results]:
                f.write(f"Stock-specific visualizations for {stock}:\n")
                f.write(f"- sentiment_trends_{stock}.png - Time series plot of sentiment trends\n")
                f.write(f"- correlation_scatter_{stock}.png - Correlation scatter plot\n")
                f.write(f"- cross_correlation_{stock}.png - Cross-correlation analysis plot\n\n")
            
            # Hypothesis testing conclusion
            f.write('\nConclusion:\n')
            f.write('-----------\n')
            if significant_correlations >= len(results) / 2:
                f.write(f"The analysis provides evidence to reject the null hypothesis for {significant_correlations} out of {len(results)} stocks.\n")
                f.write("There is a significant correlation between sentiment trends in legacy news media and Reddit for the majority of the stocks analyzed.\n\n")
            else:
                f.write(f"The analysis provides evidence to reject the null hypothesis for only {significant_correlations} out of {len(results)} stocks.\n")
                f.write("The evidence is insufficient to fully reject the null hypothesis across all stocks.\n\n")
                
            if reddit_leading_stocks:
                f.write(f"Furthermore, for {len(reddit_leading_stocks)} stocks ({', '.join(reddit_leading_stocks)}), ")
                f.write("the Granger causality tests indicate that Reddit sentiment leads news media sentiment, ")
                f.write("supporting the second part of the alternative hypothesis.\n\n")
            else:
                f.write("However, the analysis does not provide sufficient evidence that Reddit sentiment consistently leads news media sentiment across the stocks analyzed.\n\n")
            
            f.write("These findings suggest that while there is often alignment between institutional and retail investor sentiment as reflected in news media and social media, ")
            f.write("the leading relationship between these platforms varies by stock and is not consistent across the market.\n")
        
        # Generate detailed report in HTML format for better presentation
        with open('Hypo1_sentiment_analysis/Hypo1_detailed_report.html', 'w') as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html lang="en">\n')
            f.write('<head>\n')
            f.write('    <meta charset="UTF-8">\n')
            f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
            f.write('    <title>Sentiment Correlation Analysis Report</title>\n')
            f.write('    <style>\n')
            f.write('        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0 auto; max-width: 1000px; padding: 20px; }\n')
            f.write('        h1, h2, h3 { color: #2c3e50; }\n')
            f.write('        h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }\n')
            f.write('        h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }\n')
            f.write('        .summary-box { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 5px; padding: 15px; margin: 20px 0; }\n')
            f.write('        table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n')
            f.write('        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
            f.write('        th { background-color: #f2f2f2; }\n')
            f.write('        tr:nth-child(even) { background-color: #f9f9f9; }\n')
            f.write('        .image-container { display: flex; justify-content: center; margin: 20px 0; }\n')
            f.write('        .image-container img { max-width: 90%; }\n')
            f.write('        .footer { margin-top: 40px; text-align: center; font-size: 0.8em; color: #7f8c8d; }\n')
            f.write('    </style>\n')
            f.write('</head>\n')
            f.write('<body>\n')
            
            f.write('    <h1>Hypothesis 1: Sentiment Correlation Analysis</h1>\n')
            
            f.write('    <h2>Research Question</h2>\n')
            f.write('    <p>Do sentiment trends between legacy news media and Reddit show significant correlation? Does Reddit sentiment lead news media sentiment?</p>\n')
            
            f.write('    <h2>Hypotheses</h2>\n')
            f.write('    <p><strong>Null Hypothesis (H0):</strong> Sentiment trends between legacy news media and Reddit does not have any significant correlation.</p>\n')
            f.write('    <p><strong>Alternative Hypothesis (H1):</strong> Sentiment trends between legacy news media and Reddit have a significant correlation. Also, Reddit sentiment has a faster respond and leads the legacy news media sentiment.</p>\n')
            
            f.write('    <h2>Statistical Methods</h2>\n')
            f.write('    <ol>\n')
            f.write('        <li><strong>Pearson Correlation Analysis:</strong> To measure the linear relationship between news and Reddit sentiment</li>\n')
            f.write('        <li><strong>Cross-Correlation Analysis:</strong> To identify lead-lag relationships between the two sentiment series</li>\n')
            f.write('        <li><strong>Granger Causality Tests:</strong> To determine if one sentiment series helps predict the other</li>\n')
            f.write('    </ol>\n')
            
            f.write('    <h2>Summary Results</h2>\n')
            f.write('    <div class="image-container">\n')
            f.write('        <img src="correlation_summary.png" alt="Correlation Summary">\n')
            f.write('    </div>\n')
            f.write('    <div class="image-container">\n')
            f.write('        <img src="lead_lag_summary.png" alt="Lead-Lag Summary">\n')
            f.write('    </div>\n')
            
            f.write('    <div class="summary-box">\n')
            f.write('        <h3>Key Findings:</h3>\n')
            f.write('        <ul>\n')
            f.write(f'            <li>Total stocks analyzed: {len(results)}</li>\n')
            f.write(f'            <li>Stocks with significant correlation: {significant_correlations}</li>\n')
            f.write(f'            <li>Stocks where Reddit sentiment leads News sentiment: {len(reddit_leading_stocks)}{" (" + ", ".join(reddit_leading_stocks) + ")" if reddit_leading_stocks else ""}</li>\n')
            if missing_stocks:
                f.write(f'            <li>Stocks that could not be analyzed due to data issues: {", ".join(missing_stocks)}</li>\n')
            f.write('        </ul>\n')
            f.write('    </div>\n')
            
            f.write('    <h2>Detailed Results by Stock</h2>\n')
            f.write('    <table>\n')
            f.write('        <tr>\n')
            f.write('            <th>Stock</th>\n')
            f.write('            <th>Sample Size</th>\n')
            f.write('            <th>Pearson Correlation</th>\n')
            f.write('            <th>P-value</th>\n')
            f.write('            <th>Reddit Leads News?</th>\n')
            f.write('            <th>Conclusion</th>\n')
            f.write('        </tr>\n')
            
            for result in results:
                reddit_leads = False
                if result['Granger_Results']['reddit_to_news'] is not None:
                    for lag in range(1, 6):
                        if result['Granger_Results']['reddit_to_news'][lag][0]['ssr_chi2test'][1] < 0.05:
                            reddit_leads = True
                            break
                
                f.write('        <tr>\n')
                f.write(f'            <td>{result["Stock"]}</td>\n')
                f.write(f'            <td>{result["Sample_Size"]}</td>\n')
                f.write(f'            <td>{result["Pearson_Correlation"]:.4f}</td>\n')
                f.write(f'            <td>{result["P_value"]:.4f}</td>\n')
                f.write(f'            <td>{"Yes" if reddit_leads else "No"}</td>\n')
                f.write(f'            <td>{result["Conclusion"]}</td>\n')
                f.write('        </tr>\n')
            
            f.write('    </table>\n')
            
            f.write('    <h2>Individual Stock Analysis</h2>\n')
            
            for result in results:
                f.write(f'    <h3>{result["Stock"]}</h3>\n')
                f.write('    <div class="image-container">\n')
                f.write(f'        <img src="sentiment_trends_{result["Stock"]}.png" alt="Sentiment Trends for {result["Stock"]}">\n')
                f.write('    </div>\n')
                f.write('    <div class="image-container">\n')
                f.write(f'        <img src="correlation_scatter_{result["Stock"]}.png" alt="Correlation Scatter for {result["Stock"]}">\n')
                f.write('    </div>\n')
                f.write('    <div class="image-container">\n')
                f.write(f'        <img src="cross_correlation_{result["Stock"]}.png" alt="Cross-Correlation for {result["Stock"]}">\n')
                f.write('    </div>\n')
                
                f.write('    <h4>Granger Causality Results:</h4>\n')
                f.write('    <table>\n')
                f.write('        <tr>\n')
                f.write('            <th>Direction</th>\n')
                f.write('            <th>Lag 1</th>\n')
                f.write('            <th>Lag 2</th>\n')
                f.write('            <th>Lag 3</th>\n')
                f.write('            <th>Lag 4</th>\n')
                f.write('            <th>Lag 5</th>\n')
                f.write('        </tr>\n')
                
                granger_results = result['Granger_Results']
                for direction, granger_result in granger_results.items():
                    if granger_result is None:
                        continue
                    
                    f.write('        <tr>\n')
                    f.write(f'            <td>{direction}</td>\n')
                    
                    for lag in range(1, 6):
                        p_value = granger_result[lag][0]['ssr_chi2test'][1]
                        cell_style = 'style="background-color: #d4edda;"' if p_value < 0.05 else ''
                        f.write(f'            <td {cell_style}>{p_value:.4f}</td>\n')
                    
                    f.write('        </tr>\n')
                
                f.write('    </table>\n')
            
            # Conclusion
            f.write('    <h2>Conclusion</h2>\n')
            if significant_correlations >= len(results) / 2:
                f.write(f'    <p>The analysis provides evidence to reject the null hypothesis for {significant_correlations} out of {len(results)} stocks. ')
                f.write('    There is a significant correlation between sentiment trends in legacy news media and Reddit for the majority of the stocks analyzed.</p>\n')
            else:
                f.write(f'    <p>The analysis provides evidence to reject the null hypothesis for only {significant_correlations} out of {len(results)} stocks. ')
                f.write('    The evidence is insufficient to fully reject the null hypothesis across all stocks.</p>\n')
                
            if reddit_leading_stocks:
                f.write(f'    <p>Furthermore, for {len(reddit_leading_stocks)} stocks ({", ".join(reddit_leading_stocks)}), ')
                f.write('    the Granger causality tests indicate that Reddit sentiment leads news media sentiment, ')
                f.write('    supporting the second part of the alternative hypothesis.</p>\n')
            else:
                f.write('    <p>However, the analysis does not provide sufficient evidence that Reddit sentiment consistently leads news media sentiment across the stocks analyzed.</p>\n')
            
            f.write('    <p>These findings suggest that while there is often alignment between institutional and retail investor sentiment as reflected in news media and social media, ')
            f.write('    the leading relationship between these platforms varies by stock and is not consistent across the market.</p>\n')
            
            f.write('    <div class="footer">\n')
            f.write(f'        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
            f.write('    </div>\n')
            
            f.write('</body>\n')
            f.write('</html>\n')
            
        print("Generated summary report and detailed HTML report")
    
    print("Analysis completed")

if __name__ == "__main__":
    main() 