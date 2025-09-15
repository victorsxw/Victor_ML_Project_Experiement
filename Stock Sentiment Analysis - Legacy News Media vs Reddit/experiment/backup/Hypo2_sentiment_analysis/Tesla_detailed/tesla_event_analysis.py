import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import os

# Set style for all plots
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def calculate_ema(series, span=1):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

class TeslaEventAnalyzer:
    """Tesla Event Period Stock Price and Sentiment Analyzer"""
    
    def __init__(self):
        self.data = None
        self.events = {
            'earnings': {
                'Q1': '2024-04-23',  # Estimated date
                'Q2': '2024-07-24',
                'Q3': '2024-10-23',
                'Q4': '2025-01-22'
            },
            'election': '2024-11-05'  # 2024 US Presidential Election
        }
        self.window_size = 7  # Days before and after event
        self.output_dir = 'Hypo2_sentiment_analysis/Tesla_detailed'
        
    def load_data(self, file_path):
        """Load and validate Tesla stock and sentiment data"""
        print(f"Loading data from {file_path}")
        try:
            df = pd.read_pickle(file_path)
            print("Data loaded successfully")
            print("DataFrame shape:", df.shape)
            print("First few rows:\n", df.head())
            
            # Validate required columns
            required_columns = ['news', 'reddit', 'stock']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Handle missing values
            missing_count = df.isnull().sum()
            if missing_count.sum() > 0:
                print("Missing values detected:")
                for col in df.columns:
                    if missing_count[col] > 0:
                        print(f"  {col}: {missing_count[col]} missing values")
                
                # For small number of missing values, use forward fill
                if missing_count.max() < len(df) * 0.1:  # If missing values < 10%
                    print("Filling missing values using forward fill method")
                    df = df.fillna(method='ffill').fillna(method='bfill')
                else:
                    print("Too many missing values, dropping rows with any missing values")
                    df = df.dropna()
                
                print(f"After handling missing values, shape: {df.shape}")
            
            self.data = df
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
            
    def get_event_window(self, event_date, window_size=None):
        """Get data for event window period"""
        if window_size is None:
            window_size = self.window_size
            
        event_date = pd.to_datetime(event_date)
        start_date = event_date - timedelta(days=window_size)
        end_date = event_date + timedelta(days=window_size)
        
        window_data = self.data[start_date:end_date].copy()
        window_data['days_from_event'] = (window_data.index - event_date).days
        
        return window_data
        
    def calculate_rolling_correlation(self, data, window=3):
        """Calculate rolling correlation"""
        news_corr = data['news'].rolling(window).corr(data['stock'])
        reddit_corr = data['reddit'].rolling(window).corr(data['stock'])
        
        return pd.DataFrame({
            'news_correlation': news_corr,
            'reddit_correlation': reddit_corr
        }, index=data.index)
        
    def calculate_prediction_metrics(self, y_true, y_pred):
        """Calculate prediction accuracy metrics"""
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) < 2:
            return {
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1_score': np.nan
            }
        
        accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        # Calculate precision and recall for positive predictions
        true_pos = np.sum((y_true > 0) & (y_pred > 0))
        false_pos = np.sum((y_true <= 0) & (y_pred > 0))
        false_neg = np.sum((y_true > 0) & (y_pred <= 0))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    def analyze_event(self, event_date, event_name):
        """Analyze sentiment-stock relationship during event period"""
        # Get event window data
        window_data = self.get_event_window(event_date)
        if len(window_data) == 0:
            print(f"Warning: No data for {event_name} event window")
            return None
            
        # Check data points
        if len(window_data) < 5:
            print(f"Warning: Insufficient data points for {event_name} event window")
            return None
            
        # Calculate rolling correlation
        correlations = self.calculate_rolling_correlation(window_data)
        
        # Calculate Granger causality test
        granger_results = {
            'news': {},
            'reddit': {}
        }
        
        max_lag = min(2, len(window_data) - 3)
        
        for lag in range(1, max_lag + 1):
            try:
                # News -> Stock
                granger_news = grangercausalitytests(
                    window_data[['stock', 'news']].dropna(),
                    maxlag=lag,
                    verbose=False
                )
                granger_results['news'][lag] = granger_news[lag][0]['ssr_chi2test'][1]
                
                # Reddit -> Stock
                granger_reddit = grangercausalitytests(
                    window_data[['stock', 'reddit']].dropna(),
                    maxlag=lag,
                    verbose=False
                )
                granger_results['reddit'][lag] = granger_reddit[lag][0]['ssr_chi2test'][1]
            except Exception as e:
                print(f"Error in Granger causality test at lag {lag}: {str(e)}")
                granger_results['news'][lag] = np.nan
                granger_results['reddit'][lag] = np.nan
        
        # Calculate prediction metrics
        prediction_metrics = {
            'news': self.calculate_prediction_metrics(
                window_data['stock'].shift(-1),
                window_data['news']
            ),
            'reddit': self.calculate_prediction_metrics(
                window_data['stock'].shift(-1),
                window_data['reddit']
            )
        }
        
        # Generate event report
        self.generate_event_report(
            window_data,
            correlations,
            granger_results,
            prediction_metrics,
            event_name
        )
        
        # Generate visualization
        self.plot_event_analysis(window_data, correlations, event_name)
        
        return {
            'window_data': window_data,
            'correlations': correlations,
            'granger_results': granger_results,
            'prediction_metrics': prediction_metrics
        }
        
    def generate_event_report(self, data, correlations, granger_results, prediction_metrics, event_name):
        """Generate event analysis report"""
        report_path = os.path.join(self.output_dir, f'{event_name}_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Tesla {event_name} Event Period Analysis Report\n\n")
            
            # Basic statistics
            f.write("## 1. Basic Statistics\n\n")
            f.write("### 1.1 Data Overview\n")
            f.write(f"- Analysis Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n")
            f.write(f"- Sample Size: {len(data)} trading days\n\n")
            
            # Correlation analysis
            f.write("### 1.2 Correlation Analysis\n")
            news_corr = data['news'].corr(data['stock'])
            reddit_corr = data['reddit'].corr(data['stock'])
            f.write(f"- News-Stock Correlation: {news_corr:.4f}\n")
            f.write(f"- Reddit-Stock Correlation: {reddit_corr:.4f}\n\n")
            
            # Granger causality test results
            f.write("## 2. Granger Causality Test Results\n\n")
            f.write("### 2.1 News -> Stock\n")
            for lag, p_value in granger_results['news'].items():
                f.write(f"- Lag {lag} days: p-value = {p_value:.4f}\n")
            f.write("\n### 2.2 Reddit -> Stock\n")
            for lag, p_value in granger_results['reddit'].items():
                f.write(f"- Lag {lag} days: p-value = {p_value:.4f}\n")
            f.write("\n")
            
            # Prediction metrics
            f.write("## 3. Prediction Performance\n\n")
            f.write("### 3.1 News Sentiment Metrics\n")
            for metric, value in prediction_metrics['news'].items():
                f.write(f"- {metric}: {value:.4f}\n")
            f.write("\n### 3.2 Reddit Sentiment Metrics\n")
            for metric, value in prediction_metrics['reddit'].items():
                f.write(f"- {metric}: {value:.4f}\n")
            f.write("\n")
            
            # Conclusions
            f.write("## 4. Conclusions\n\n")
            if abs(news_corr) > abs(reddit_corr):
                stronger = "News sentiment"
                corr_diff = abs(news_corr) - abs(reddit_corr)
            else:
                stronger = "Reddit sentiment"
                corr_diff = abs(reddit_corr) - abs(news_corr)
            
            f.write(f"During the {event_name} event period:\n\n")
            f.write(f"1. {stronger} shows stronger predictive power, with a correlation advantage of {corr_diff:.4f}\n")
            
            # Granger causality summary
            significant_news = sum(1 for p in granger_results['news'].values() if p < 0.05)
            significant_reddit = sum(1 for p in granger_results['reddit'].values() if p < 0.05)
            f.write(f"2. News sentiment shows significance in {significant_news} lag periods, ")
            f.write(f"while Reddit sentiment shows significance in {significant_reddit} lag periods\n")
            
            # Prediction accuracy summary
            if prediction_metrics['news']['accuracy'] > prediction_metrics['reddit']['accuracy']:
                better_predictor = "News sentiment"
                acc_diff = prediction_metrics['news']['accuracy'] - prediction_metrics['reddit']['accuracy']
            else:
                better_predictor = "Reddit sentiment"
                acc_diff = prediction_metrics['reddit']['accuracy'] - prediction_metrics['news']['accuracy']
            
            f.write(f"3. {better_predictor} has higher prediction accuracy, with an advantage of {acc_diff:.4f}\n")
            
        print(f"Report saved to: {report_path}")
        
    def plot_event_analysis(self, data, correlations, event_name):
        """Generate event analysis visualization"""
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # First subplot: sentiment and stock trends
        ax1 = plt.subplot(211)
        
        # Calculate EMA
        news_ema = calculate_ema(data['news'])
        reddit_ema = calculate_ema(data['reddit'])
        stock_ema = calculate_ema(data['stock'])
        
        # Plot sentiment trends
        ax1.plot(data.index, news_ema, label='News Sentiment (EMA)', color='blue', linewidth=2)
        ax1.plot(data.index, reddit_ema, label='Reddit Sentiment (EMA)', color='orange', linewidth=2)
        ax1.plot(data.index, stock_ema, label='Stock Price (EMA)', color='green', linestyle='--', linewidth=2)
        ax1.axvline(x=data[data['days_from_event'] == 0].index[0], color='red', linestyle='--', alpha=0.5)
        ax1.set_title(f'Tesla {event_name} Event Period Analysis')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Standardized Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Second subplot: rolling correlation
        ax2 = plt.subplot(212)
        ax2.plot(correlations.index, correlations['news_correlation'], label='News-Stock Correlation', color='blue')
        ax2.plot(correlations.index, correlations['reddit_correlation'], label='Reddit-Stock Correlation', color='orange')
        ax2.axvline(x=data[data['days_from_event'] == 0].index[0], color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Rolling Correlation')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{event_name}_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_all_events(self):
        """Analyze all special periods"""
        all_results = {}
        
        # Analyze earnings events
        for quarter, date in self.events['earnings'].items():
            print(f"\nAnalyzing {quarter} earnings event...")
            results = self.analyze_event(date, f'earnings_{quarter}')
            if results:
                all_results[f'earnings_{quarter}'] = results
                
        # Analyze election event
        print("\nAnalyzing presidential election event...")
        results = self.analyze_event(self.events['election'], 'election')
        if results:
            all_results['election'] = results
            
        # Generate summary report
        self.generate_summary_report(all_results)
        
    def generate_summary_report(self, all_results):
        """Generate comprehensive analysis report"""
        report_path = os.path.join(self.output_dir, 'tesla_events_summary_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Tesla Special Events Analysis Summary Report\n\n")
            
            # Overall statistics
            f.write("## 1. Overall Statistics\n\n")
            f.write(f"- Total Events Analyzed: {len(all_results)}\n")
            f.write("- Event Types: Earnings Reports and Presidential Election\n\n")
            
            # Event comparison
            f.write("## 2. Sentiment Predictive Power Comparison\n\n")
            f.write("### 2.1 Prediction Accuracy Comparison\n\n")
            f.write("| Event | News Accuracy | Reddit Accuracy | Stronger Predictor |\n")
            f.write("|-------|---------------|-----------------|-------------------|\n")
            
            for event, results in all_results.items():
                news_acc = results['prediction_metrics']['news']['accuracy']
                reddit_acc = results['prediction_metrics']['reddit']['accuracy']
                stronger = "News" if news_acc > reddit_acc else "Reddit"
                f.write(f"| {event} | {news_acc:.4f} | {reddit_acc:.4f} | {stronger} |\n")
            
            # Granger causality comparison
            f.write("\n### 2.2 Granger Causality Test Significance\n\n")
            f.write("| Event | News Significant Lags | Reddit Significant Lags |\n")
            f.write("|-------|---------------------|----------------------|\n")
            
            for event, results in all_results.items():
                news_sig = sum(1 for p in results['granger_results']['news'].values() if p < 0.05)
                reddit_sig = sum(1 for p in results['granger_results']['reddit'].values() if p < 0.05)
                f.write(f"| {event} | {news_sig} | {reddit_sig} |\n")
            
            # Overall conclusions
            f.write("\n## 3. Overall Conclusions\n\n")
            
            # Calculate overall performance
            total_news_better = sum(1 for r in all_results.values() 
                                  if r['prediction_metrics']['news']['accuracy'] > 
                                  r['prediction_metrics']['reddit']['accuracy'])
            total_reddit_better = len(all_results) - total_news_better
            
            f.write("1. Across all analyzed events:\n")
            f.write(f"   - News sentiment performed better in {total_news_better} events\n")
            f.write(f"   - Reddit sentiment performed better in {total_reddit_better} events\n\n")
            
            # Event characteristics
            f.write("2. Event Period Characteristics:\n")
            f.write("   - Earnings Periods: [Findings to be filled based on results]\n")
            f.write("   - Election Period: [Findings to be filled based on results]\n\n")
            
            # Investment implications
            f.write("## 4. Investment Implications\n\n")
            f.write("Based on the analysis results, we recommend:\n\n")
            f.write("1. [Recommendations to be filled based on results]\n")
            f.write("2. [Recommendations to be filled based on results]\n")
            f.write("3. [Recommendations to be filled based on results]\n")
            
        print(f"Summary report saved to: {report_path}")

def main():
    """Main function"""
    # Create analyzer instance
    analyzer = TeslaEventAnalyzer()
    
    # Load data
    data_path = 'dataset/6datasets-2024-2025/tsla_compare.pkl'
    if not analyzer.load_data(data_path):
        print("Data loading failed, program exits")
        return
        
    # Analyze all special periods
    analyzer.analyze_all_events()
    
if __name__ == "__main__":
    main() 