import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    confusion_matrix, f1_score, classification_report
)
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import grangercausalitytests
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

def perform_granger_causality(df, max_lag=5):
    """
    Perform Granger causality tests between sentiment and stock price
    
    Parameters:
        df: DataFrame containing sentiment and stock price data
        max_lag: Maximum number of lags to test
    Returns:
        Dictionary containing test results
    """
    try:
        results = {
            'news_to_stock': {},
            'stock_to_news': {},
            'reddit_to_stock': {},
            'stock_to_reddit': {}
        }
        
        # Prepare data pairs for testing
        data_pairs = [
            ('news', 'stock'),
            ('stock', 'news'),
            ('reddit', 'stock'),
            ('stock', 'reddit')
        ]
        
        for x, y in data_pairs:
            data = pd.concat([df[x], df[y]], axis=1)
            key = f'{x}_to_{y}'
            
            # Perform Granger causality test
            granger_test = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            
            # Extract p-values for each lag (using F-test p-values)
            results[key] = {
                lag: round(test[0]['ssr_ftest'][1], 4)
                for lag, test in granger_test.items()
            }
            
        return results
    except Exception as e:
        print("Error performing Granger causality test:")
        traceback.print_exc()
        raise

def evaluate_direction_prediction(y_true, y_pred):
    """Evaluate direction prediction performance"""
    try:
        # Convert to direction (1 for up, 0 for down)
        y_true_dir = (np.diff(y_true) > 0).astype(int)
        y_pred_dir = (np.diff(y_pred) > 0).astype(int)
        
        # Calculate baseline accuracy (majority class)
        baseline_acc = max(
            np.mean(y_true_dir),  # all predicted as up
            1 - np.mean(y_true_dir)  # all predicted as down
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_dir, y_pred_dir)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_dir, y_pred_dir)
        
        # Check if we have both classes in predictions
        unique_classes = np.unique(np.concatenate([y_true_dir, y_pred_dir]))
        if len(unique_classes) == 1:
            # Only one class present, set f1 score to 0
            f1 = 0.0
            class_report = {
                'Down': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
                'Up': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}
            }
        else:
            # Both classes present, calculate f1 score
            f1 = f1_score(y_true_dir, y_pred_dir)
            class_report = classification_report(
                y_true_dir, 
                y_pred_dir,
                target_names=['Down', 'Up'],
                output_dict=True,
                zero_division=0
            )
        
        return {
            'accuracy': accuracy,
            'baseline_accuracy': baseline_acc,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'up_down_ratio': np.mean(y_true_dir)
        }
    except Exception as e:
        print("Error evaluating direction prediction:")
        traceback.print_exc()
        return {
            'accuracy': 0.0,
            'baseline_accuracy': 0.0,
            'f1_score': 0.0,
            'confusion_matrix': np.zeros((2, 2)),
            'classification_report': {
                'Down': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
                'Up': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}
            },
            'up_down_ratio': 0.0
        }

def build_prediction_model(X, y):
    """Enhanced prediction model evaluation with balanced validation criteria"""
    try:
        # Robust scaling using median and IQR
        def robust_scale(data):
            median = np.median(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            if iqr == 0:
                iqr = 1.0  # Prevent division by zero
            return (data - median) / iqr
        
        X_scaled = robust_scale(X)
        y_scaled = robust_scale(y)
        
        model = LinearRegression()
        tscv = TimeSeriesSplit(n_splits=5, test_size=30)
        
        scores = {
            'r2': [],
            'mse': [],
            'direction_metrics': []
        }
        
        valid_folds = 0
        total_folds = 0
        
        for train_idx, test_idx in tscv.split(X_scaled):
            total_folds += 1
            if len(train_idx) < 60:  # Ensure minimum training size
                continue
                
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
            
            # Reshape features
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
            
            try:
                # Fit model and make predictions
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate regression metrics
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                # More lenient validation conditions
                if r2 <= 1.0 and not np.isinf(mse):  # Allow negative R² but still check for infinity
                    scores['r2'].append(r2)
                    scores['mse'].append(mse)
                    
                    # Calculate direction prediction metrics
                    direction_metrics = evaluate_direction_prediction(y_test, y_pred)
                    scores['direction_metrics'].append(direction_metrics)
                    valid_folds += 1
                    
            except Exception as e:
                print(f"Warning: Error in fold {total_folds}: {str(e)}")
                continue
        
        # Require at least 2 valid folds or 40% of total folds
        min_required_folds = min(2, max(1, int(total_folds * 0.4)))
        
        if valid_folds >= min_required_folds:
            avg_metrics = {
                'r2': np.mean(scores['r2']) if scores['r2'] else 0.0,
                'mse': np.mean(scores['mse']) if scores['mse'] else float('inf'),
                'direction_metrics': {
                    'accuracy': np.mean([m['accuracy'] for m in scores['direction_metrics']]),
                    'baseline_accuracy': np.mean([m['baseline_accuracy'] for m in scores['direction_metrics']]),
                    'f1_score': np.mean([m['f1_score'] for m in scores['direction_metrics']]),
                    'up_down_ratio': np.mean([m['up_down_ratio'] for m in scores['direction_metrics']])
                },
                'coefficient': float(model.coef_[0])
            }
        else:
            print(f"Warning: Only {valid_folds}/{total_folds} valid folds")
            avg_metrics = {
                'r2': 0.0,
                'mse': float('inf'),
                'direction_metrics': {
                    'accuracy': 0.0,
                    'baseline_accuracy': 0.0,
                    'f1_score': 0.0,
                    'up_down_ratio': 0.0
                },
                'coefficient': 0.0
            }
        
        return avg_metrics
        
    except Exception as e:
        print("Error in build_prediction_model:")
        traceback.print_exc()
        return {
            'r2': 0.0,
            'mse': float('inf'),
            'direction_metrics': {
                'accuracy': 0.0,
                'baseline_accuracy': 0.0,
                'f1_score': 0.0,
                'up_down_ratio': 0.0
            },
            'coefficient': 0.0
        }

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
    """Plot prediction results comparison with updated metrics"""
    try:
        # Create figure with subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = [
            ('r2', 'R² Score'),
            ('direction_metrics.accuracy', 'Direction Accuracy'),
            ('direction_metrics.f1_score', 'F1 Score'),
            ('mse', 'Mean Squared Error')
        ]
        
        for (metric, title), ax in zip(metrics, axes.flat):
            # Extract metric data
            data = {
                'News': [get_nested_value(results['news'], metric) for results in prediction_results.values()],
                'Reddit': [get_nested_value(results['reddit'], metric) for results in prediction_results.values()]
            }
            df = pd.DataFrame(data, index=prediction_results.keys())
            
            # Create bar plot
            df.plot(kind='bar', ax=ax)
            ax.set_title(title)
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'prediction_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved prediction comparison plot to {output_path}")
        
    except Exception as e:
        print("Error plotting prediction comparison:")
        traceback.print_exc()

def get_nested_value(d, key_path):
    """Helper function to get nested dictionary values"""
    keys = key_path.split('.')
    value = d
    for key in keys:
        value = value[key]
    return value

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
    granger_results = {}  # New: store Granger causality results
    
    # Process each stock
    for stock in stocks:
        print(f"\nAnalyzing {stock.upper()}")
        try:
            # Load data
            file_path = f'dataset/20250315/merged/{stock}_merged_data.csv'
            df = load_data(file_path, stock)
            
            # Generate visualizations
            print(f"Generating visualizations for {stock}")
            plot_sentiment_and_stock_trends(df, stock, output_dir)
            plot_correlation_scatter_matrix(df, stock, output_dir)
            plot_rolling_correlations(df, stock, output_dir)
            
            # Calculate correlations
            print(f"Calculating correlations for {stock}")
            correlations[stock] = calculate_correlations(df)
            lagged_correlations[stock] = calculate_lagged_correlations(df)
            
            # New: Perform Granger causality tests
            print(f"Performing Granger causality tests for {stock}")
            granger_results[stock] = perform_granger_causality(df)
            
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
    
    # Generate summary visualizations
    print("\nGenerating summary visualizations...")
    plot_correlation_heatmap(correlations, output_dir)
    plot_lagged_correlations(lagged_correlations, output_dir)
    plot_prediction_comparison(prediction_results, output_dir)
    
    # Generate enhanced summary report
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
                
                for source in ['news', 'reddit']:
                    results = prediction_results[stock][source]
                    f.write(f'\n##### {source.capitalize()} Sentiment Model\n')
                    f.write(f'- R² Score: {results["r2"]:.4f}\n')
                    f.write(f'- MSE: {results["mse"]:.4f}\n')
                    f.write(f'- Direction Prediction:\n')
                    f.write(f'  - Accuracy: {results["direction_metrics"]["accuracy"]:.4f}\n')
                    f.write(f'  - Baseline Accuracy: {results["direction_metrics"]["baseline_accuracy"]:.4f}\n')
                    f.write(f'  - F1 Score: {results["direction_metrics"]["f1_score"]:.4f}\n')
                    f.write(f'  - Up/Down Ratio: {results["direction_metrics"]["up_down_ratio"]:.4f}\n')
                    f.write(f'- Coefficient: {results["coefficient"]:.4f}\n')
        
        # Add Granger causality results
        f.write('\n## Granger Causality Analysis\n')
        for stock in stocks:
            if stock in granger_results:
                f.write(f'\n### {stock.upper()}\n')
                
                # News sentiment to stock price
                f.write('\n#### News Sentiment → Stock Price\n')
                for lag, p_value in granger_results[stock]['news_to_stock'].items():
                    significance = ''
                    if p_value < 0.001:
                        significance = '***'
                    elif p_value < 0.01:
                        significance = '**'
                    elif p_value < 0.05:
                        significance = '*'
                    f.write(f'- Lag {lag}: p = {p_value:.4f} {significance}\n')
                
                # Reddit sentiment to stock price
                f.write('\n#### Reddit Sentiment → Stock Price\n')
                for lag, p_value in granger_results[stock]['reddit_to_stock'].items():
                    significance = ''
                    if p_value < 0.001:
                        significance = '***'
                    elif p_value < 0.01:
                        significance = '**'
                    elif p_value < 0.05:
                        significance = '*'
                    f.write(f'- Lag {lag}: p = {p_value:.4f} {significance}\n')
        
        # Add detailed direction prediction evaluation
        f.write('\n## Direction Prediction Analysis\n')
        for stock in stocks:
            if stock in prediction_results:
                f.write(f'\n### {stock.upper()}\n')
                
                for source in ['news', 'reddit']:
                    metrics = prediction_results[stock][source]['direction_metrics']
                    f.write(f'\n#### {source.capitalize()} Model\n')
                    f.write(f'- Accuracy: {metrics["accuracy"]:.4f}\n')
                    f.write(f'- Baseline Accuracy: {metrics["baseline_accuracy"]:.4f}\n')
                    f.write(f'- F1 Score: {metrics["f1_score"]:.4f}\n')
                    f.write(f'- Up/Down Ratio: {metrics["up_down_ratio"]:.4f}\n')
        
        # Add significance note
        f.write('\n### Significance Levels\n')
        f.write('- *** p < 0.001\n')
        f.write('- ** p < 0.01\n')
        f.write('- * p < 0.05\n')
    
    print(f"Analysis completed. Results saved to {output_dir}/")

if __name__ == "__main__":
    main() 