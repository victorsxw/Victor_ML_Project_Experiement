import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import os
from datetime import datetime
import traceback

# Set style for all plots
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def load_data(file_path, stock_name):
    """Load and validate the preprocessed data"""
    print(f"Loading data for {stock_name} from {file_path}")
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path, index_col=0)
        print(f"Loaded data shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        
        df.index = pd.to_datetime(df.index)
        
        # Extract the required columns
        news_col = f"news_{stock_name}_EMA0.02_scaled"
        reddit_col = f"reddit_{stock_name}_EMA0.02_scaled"
        stock_col = f"stock_{stock_name}_EMA0.02_scaled"
        
        # Verify columns exist
        missing_cols = []
        for col in [news_col, reddit_col, stock_col]:
            if col not in df.columns:
                missing_cols.append(col)
        if missing_cols:
            raise KeyError(f"Missing columns: {missing_cols}")
        
        result_df = pd.DataFrame({
            'news': df[news_col],
            'reddit': df[reddit_col],
            'stock': df[stock_col]
        }, index=df.index)
        
        # Check for NaN values
        nan_counts = result_df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"Warning: Found NaN values:\n{nan_counts}")
        
        clean_df = result_df.dropna()
        print(f"Final data shape after cleaning: {clean_df.shape}")
        
        return clean_df
        
    except Exception as e:
        print(f"Error loading data for {stock_name}:")
        traceback.print_exc()
        raise

def calculate_correlations(df):
    """Calculate correlations between sentiment and stock movements"""
    try:
        correlations = {
            'news': stats.pearsonr(df['news'], df['stock']),
            'reddit': stats.pearsonr(df['reddit'], df['stock'])
        }
        return correlations
    except Exception as e:
        print("Error calculating correlations:")
        traceback.print_exc()
        raise

def calculate_lagged_correlations(df, max_lag=5):
    """Calculate lagged correlations between sentiment and stock movements"""
    try:
        lagged_corr = {'news': [], 'reddit': []}
        
        for lag in range(max_lag + 1):
            news_corr = stats.pearsonr(
                df['news'].shift(lag).dropna(),
                df['stock'].dropna()[:len(df)-lag]
            )[0]
            reddit_corr = stats.pearsonr(
                df['reddit'].shift(lag).dropna(),
                df['stock'].dropna()[:len(df)-lag]
            )[0]
            
            lagged_corr['news'].append(news_corr)
            lagged_corr['reddit'].append(reddit_corr)
        
        return pd.DataFrame(lagged_corr, index=range(max_lag + 1))
    except Exception as e:
        print("Error calculating lagged correlations:")
        traceback.print_exc()
        raise

def build_prediction_model(X, y):
    """Build and evaluate prediction model"""
    try:
        model = LinearRegression()
        
        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = {
            'r2': [],
            'mse': [],
            'direction_accuracy': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train.reshape(-1, 1), y_train)
            y_pred = model.predict(X_test.reshape(-1, 1))
            
            scores['r2'].append(r2_score(y_test, y_pred))
            scores['mse'].append(mean_squared_error(y_test, y_pred))
            scores['direction_accuracy'].append(
                accuracy_score((y_test > 0), (y_pred > 0))
            )
        
        return {
            'r2': np.mean(scores['r2']),
            'mse': np.mean(scores['mse']),
            'direction_accuracy': np.mean(scores['direction_accuracy']),
            'coefficient': model.coef_[0]
        }
    except Exception as e:
        print("Error building prediction model:")
        traceback.print_exc()
        raise

def plot_correlation_heatmap(correlations_dict, output_dir):
    """Plot correlation heatmap for all stocks"""
    try:
        # Extract correlation values
        data = {
            'News': [corr['news'][0] for corr in correlations_dict.values()],
            'Reddit': [corr['reddit'][0] for corr in correlations_dict.values()]
        }
        corr_data = pd.DataFrame(data, index=correlations_dict.keys())
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_data, annot=True, cmap='RdYlBu', center=0, fmt='.4f')
        plt.title('Sentiment-Stock Price Correlation Heatmap')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation heatmap to {output_path}")
        
    except Exception as e:
        print("Error plotting correlation heatmap:")
        traceback.print_exc()

def plot_lagged_correlations(lagged_corr_dict, output_dir):
    """Plot lagged correlations for all stocks"""
    try:
        fig = plt.figure(figsize=(15, 10))
        
        for i, (stock_name, df) in enumerate(lagged_corr_dict.items(), 1):
            plt.subplot(2, 3, i)
            df.plot(marker='o')
            plt.title(f'{stock_name.upper()}')
            plt.xlabel('Lag (days)')
            plt.ylabel('Correlation')
            plt.grid(True)
            plt.legend(['News', 'Reddit'])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'lagged_correlations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved lagged correlations plot to {output_path}")
        
    except Exception as e:
        print("Error plotting lagged correlations:")
        traceback.print_exc()

def plot_prediction_comparison(prediction_results, output_dir):
    """Plot prediction results comparison"""
    try:
        metrics = ['r2', 'direction_accuracy']
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, metric in enumerate(metrics):
            data = {
                'News': [results['news'][metric] for results in prediction_results.values()],
                'Reddit': [results['reddit'][metric] for results in prediction_results.values()]
            }
            df = pd.DataFrame(data, index=prediction_results.keys())
            
            df.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel('Score')
            axes[i].grid(True)
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'prediction_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved prediction comparison plot to {output_path}")
        
    except Exception as e:
        print("Error plotting prediction comparison:")
        traceback.print_exc()

def plot_sentiment_and_stock_trends(df, stock_name, output_dir):
    """Plot sentiment trends and stock price together"""
    try:
        plt.figure(figsize=(15, 8))
        
        # Create gradient background
        ax = plt.gca()
        
        # Calculate y-axis range with 5% margin
        all_values = pd.concat([df['news'], df['reddit'], df['stock']], axis=0)
        y_min = min(all_values.min(), -3)
        y_max = max(all_values.max(), 3)
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
        
        # Plot the three lines with different styles
        plt.plot(df.index, df['news'], label='News', color='blue', linewidth=1.5)
        plt.plot(df.index, df['reddit'], label='Reddit', color='orange', linewidth=1.5)
        plt.plot(df.index, df['stock'], label='Stock Price', color='green', linewidth=1.5)
        
        # Set x-axis format
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Set y-axis format
        plt.ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.grid(True, which='minor', linestyle=':', alpha=0.1)
        
        # Set title and labels
        plt.title(f'Sentiment and Stock Price Trends for {stock_name.upper()}', pad=20)
        plt.xlabel('Date')
        plt.ylabel('Standardized Score')
        
        # Adjust legend
        plt.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'sentiment_stock_trends_{stock_name.upper()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sentiment and stock trends plot to {output_path}")
        
    except Exception as e:
        print(f"Error plotting sentiment and stock trends for {stock_name}:")
        traceback.print_exc()

def plot_correlation_scatter_matrix(df, stock_name, output_dir):
    """Plot correlation scatter matrix between news, reddit and stock price"""
    try:
        # Create scatter matrix
        fig = plt.figure(figsize=(12, 12))
        
        # Define variables and their colors
        variables = ['news', 'reddit', 'stock']
        colors = ['blue', 'orange', 'green']
        
        # Create 3x3 subplot grid
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                plt.subplot(3, 3, i*3 + j + 1)
                
                if var1 == var2:
                    # Histogram on diagonal
                    plt.hist(df[var1], bins=30, color=colors[i], alpha=0.6)
                    plt.title(f'{var1.capitalize()} Distribution')
                else:
                    # Scatter plot with regression line
                    plt.scatter(df[var2], df[var1], alpha=0.5, color=colors[i])
                    
                    # Add regression line
                    z = np.polyfit(df[var2], df[var1], 1)
                    p = np.poly1d(z)
                    plt.plot(df[var2], p(df[var2]), "r--", alpha=0.8)
                    
                    # Add correlation coefficient
                    corr = df[var1].corr(df[var2])
                    plt.title(f'Corr: {corr:.2f}')
                
                if i == 2:  # Bottom row
                    plt.xlabel(var2.capitalize())
                if j == 0:  # Leftmost column
                    plt.ylabel(var1.capitalize())
        
        plt.suptitle(f'Correlation Matrix for {stock_name.upper()}', y=1.02, fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'correlation_matrix_{stock_name.upper()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation matrix plot to {output_path}")
        
    except Exception as e:
        print(f"Error plotting correlation matrix for {stock_name}:")
        traceback.print_exc()

def plot_rolling_correlations(df, stock_name, output_dir, window=30):
    """Plot rolling correlations between sentiments and stock price"""
    try:
        plt.figure(figsize=(15, 6))
        
        # Calculate rolling correlations
        roll_corr_news = df['news'].rolling(window=window).corr(df['stock'])
        roll_corr_reddit = df['reddit'].rolling(window=window).corr(df['stock'])
        
        # Plot rolling correlations
        plt.plot(df.index[window-1:], roll_corr_news[window-1:], 
                label='News-Stock Correlation', color='blue', linewidth=1.5)
        plt.plot(df.index[window-1:], roll_corr_reddit[window-1:], 
                label='Reddit-Stock Correlation', color='orange', linewidth=1.5)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.2)
        
        # Set title and labels
        plt.title(f'{window}-Day Rolling Correlations for {stock_name.upper()}')
        plt.xlabel('Date')
        plt.ylabel('Correlation Coefficient')
        
        # Customize x-axis
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'rolling_correlations_{stock_name.upper()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved rolling correlations plot to {output_path}")
        
    except Exception as e:
        print(f"Error plotting rolling correlations for {stock_name}:")
        traceback.print_exc()

def main():
    print("Starting Hypothesis 2 analysis...")
    
    # Create output directory
    output_dir = 'Hypo2_analysis'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # List of stocks to analyze
    stocks = ['aapl', 'meta', 'tsla', 'nvda', 'pltr', 'spy']
    
    # Store results
    correlations = {}
    lagged_correlations = {}
    prediction_results = {}
    
    # Process each stock
    for stock in stocks:
        print(f"\nAnalyzing {stock.upper()}")
        try:
            # Load data
            file_path = f'dataset/20250315/merged/{stock}_merged_data.csv'
            df = load_data(file_path, stock)
            
            # Generate new visualizations
            print(f"Generating visualizations for {stock}")
            plot_sentiment_and_stock_trends(df, stock, output_dir)
            plot_correlation_scatter_matrix(df, stock, output_dir)
            plot_rolling_correlations(df, stock, output_dir)
            
            # Calculate correlations
            print(f"Calculating correlations for {stock}")
            correlations[stock] = calculate_correlations(df)
            lagged_correlations[stock] = calculate_lagged_correlations(df)
            
            # Build prediction models
            print(f"Building prediction models for {stock}")
            prediction_results[stock] = {
                'news': build_prediction_model(
                    df['news'].values, df['stock'].values
                ),
                'reddit': build_prediction_model(
                    df['reddit'].values, df['stock'].values
                )
            }
            
            print(f"Successfully processed {stock}")
            
        except Exception as e:
            print(f"Failed to process {stock}:")
            traceback.print_exc()
            continue
    
    # Generate original visualizations
    print("\nGenerating summary visualizations...")
    plot_correlation_heatmap(correlations, output_dir)
    plot_lagged_correlations(lagged_correlations, output_dir)
    plot_prediction_comparison(prediction_results, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    report_path = os.path.join(output_dir, 'analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# Hypothesis 2: Predictive Power Analysis Report\n\n')
        
        f.write('## Research Question\n')
        f.write('Does sentiment from different sources (news media vs Reddit) have different predictive power for stock price movements?\n\n')
        
        f.write('## Hypotheses\n')
        f.write('- H0: Sentiment from legacy news media and Reddit does not significantly predict stock price movements.\n')
        f.write('- H1: Reddit sentiment has a stronger correlation with short-term stock price movements than legacy news media.\n\n')
        
        f.write('## Results Summary\n\n')
        
        # Correlation summary
        f.write('### Correlation Analysis\n')
        for stock in stocks:
            if stock in correlations:
                news_corr, news_p = correlations[stock]['news']
                reddit_corr, reddit_p = correlations[stock]['reddit']
                f.write(f'\n#### {stock.upper()}\n')
                f.write(f'- News Correlation: {news_corr:.4f} (p={news_p:.4f})\n')
                f.write(f'- Reddit Correlation: {reddit_corr:.4f} (p={reddit_p:.4f})\n')
        
        # Prediction results summary
        f.write('\n### Prediction Results\n')
        for stock in stocks:
            if stock in prediction_results:
                f.write(f'\n#### {stock.upper()}\n')
                f.write('\nNews Sentiment Model:\n')
                f.write(f'- R² Score: {prediction_results[stock]["news"]["r2"]:.4f}\n')
                f.write(f'- Direction Accuracy: {prediction_results[stock]["news"]["direction_accuracy"]:.4f}\n')
                f.write(f'- MSE: {prediction_results[stock]["news"]["mse"]:.4f}\n')
                
                f.write('\nReddit Sentiment Model:\n')
                f.write(f'- R² Score: {prediction_results[stock]["reddit"]["r2"]:.4f}\n')
                f.write(f'- Direction Accuracy: {prediction_results[stock]["reddit"]["direction_accuracy"]:.4f}\n')
                f.write(f'- MSE: {prediction_results[stock]["reddit"]["mse"]:.4f}\n')
        
        # Overall conclusion
        f.write('\n## Conclusion\n')
        
        # Calculate average metrics
        avg_news_r2 = np.mean([res['news']['r2'] for res in prediction_results.values()])
        avg_reddit_r2 = np.mean([res['reddit']['r2'] for res in prediction_results.values()])
        avg_news_acc = np.mean([res['news']['direction_accuracy'] for res in prediction_results.values()])
        avg_reddit_acc = np.mean([res['reddit']['direction_accuracy'] for res in prediction_results.values()])
        
        f.write(f'\nAverage Performance Metrics:\n')
        f.write(f'- News Sentiment: R² = {avg_news_r2:.4f}, Direction Accuracy = {avg_news_acc:.4f}\n')
        f.write(f'- Reddit Sentiment: R² = {avg_reddit_r2:.4f}, Direction Accuracy = {avg_reddit_acc:.4f}\n\n')
        
        # Statistical significance test
        r2_diff = [res['reddit']['r2'] - res['news']['r2'] for res in prediction_results.values()]
        acc_diff = [res['reddit']['direction_accuracy'] - res['news']['direction_accuracy'] 
                   for res in prediction_results.values()]
        
        r2_ttest = stats.ttest_1samp(r2_diff, 0)
        acc_ttest = stats.ttest_1samp(acc_diff, 0)
        
        f.write('Statistical Significance Tests:\n')
        f.write(f'- R² Difference (Reddit - News): t={r2_ttest.statistic:.4f}, p={r2_ttest.pvalue:.4f}\n')
        f.write(f'- Direction Accuracy Difference: t={acc_ttest.statistic:.4f}, p={acc_ttest.pvalue:.4f}\n\n')
        
        # Test conclusion
        if r2_ttest.pvalue < 0.05 or acc_ttest.pvalue < 0.05:
            if avg_reddit_r2 > avg_news_r2 and avg_reddit_acc > avg_news_acc:
                f.write('The analysis supports H1: Reddit sentiment shows significantly stronger predictive power for stock price movements.\n')
            else:
                f.write('The analysis supports rejecting H0, but the direction is opposite to H1: News sentiment shows stronger predictive power.\n')
        else:
            f.write('The analysis does not provide sufficient evidence to reject H0: The predictive power difference between news and Reddit sentiment is not statistically significant.\n')
        
        f.write('\n## Visualizations\n')
        f.write('1. sentiment_stock_trends_{stock_name.upper()}.png - Sentiment and stock price trends\n')
        f.write('2. correlation_matrix_{stock_name.upper()}.png - Correlation scatter matrix\n')
        f.write('3. rolling_correlations_{stock_name.upper()}.png - Rolling correlations\n')
        f.write('4. correlation_heatmap.png - Heatmap showing correlations between sentiment and stock prices\n')
        f.write('5. lagged_correlations.png - Time-lagged correlation analysis\n')
        f.write('6. prediction_comparison.png - Comparison of prediction performance metrics\n')
    
    print(f"Analysis completed. Results saved to {output_dir}/")

if __name__ == "__main__":
    main() 