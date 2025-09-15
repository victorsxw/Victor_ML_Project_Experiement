class SentimentAnalyzer:
    """Class for analyzing sentiment predictive power on stock prices"""
    
    def __init__(self, data_dir='../dataset', output_dir='Hypo2_sentiment_analysis'):
        """Initialize the analyzer with data and output directories"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.setup_directories()
        self.valid_results = {} 