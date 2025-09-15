# Stock Sentiment Analysis: Legacy News Media vs. Reddit

## Abstract

This comprehensive research project investigates the relationship between sentiment extracted from legacy news media and Reddit post titles and their impact on stock price movements for high-volume stocks. Using advanced large language models (LLMs) for sentiment classification, we constructed sentiment indices and analyzed their correlation with stock prices through multiple statistical approaches including Cross-Correlation, Lead-Lag Correlation, and Granger Causality Tests.

## Research Questions

### Primary Research Questions
* How do sentiment trends differ between legacy news media and Reddit across different stocks?
* How do these sentiment trends influence stock price movements and what is their predictive power?
* Do different types of sentiment (data-driven, fiscal-related, opinion-based) have different effects on stock prices?

### Hypotheses
* **H0**: Traditional news media and Reddit sentiment trends have no significant correlation
* **H1**: Traditional news media and Reddit sentiment trends show significant correlation
* **H0**: Sentiment from legacy news media and Reddit does not significantly predict stock price movements
* **H1**: Reddit sentiment has stronger correlation with short-term stock price movements than legacy news media

## Methodology

### 1. Data Collection
* **News Media**: Collected 28,947 news titles from March 2024 to April 2025 from financial websites including `nasdaq.com` and `benzinga.com`
* **Reddit**: Scraped 37,808 post titles from popular stock subreddits via the Reddit API
* **Stock Data**: Collected daily closing prices for six major stocks (AAPL, META, NVDA, PLTR, TSLA, and SPY) from the Yahoo! Finance API

### 2. Sentiment Analysis
* **Model**: Used Deepseek model (`deepseek-r1-distill-qwen-7b`) as a classifier
* **Classification**: Categorized sentiment as positive (+1), negative (-1), or neutral (0)
* **Index Construction**: Calculated daily sentiment scores by averaging all titles per day
* **Smoothing**: Applied moving averages to create smoothed sentiment indices

### 3. Statistical Analysis
* **Cross-Correlation Analysis**: Measured linear relationships between sentiment indices and stock prices
* **Lead-Lag Correlation**: Analyzed correlations at different time lags (0-5 days) to assess predictive power
* **Granger Causality Tests**: Evaluated whether past sentiment values significantly improve stock price predictions
* **Machine Learning Models**: Implemented predictive models to assess direction prediction accuracy

## Key Findings

### Sentiment Correlation Results
* **News vs. Reddit Sentiment**: Significant correlations found for five stocks (AAPL, NVDA, PLTR, TSLA, SPY)
* **Strongest Correlations**: NVDA (0.6718 news, 0.2076 Reddit) and TSLA (0.8429 news, 0.7850 Reddit)
* **Negative Correlations**: META showed negative correlations (-0.1904 news, -0.1713 Reddit)

### Predictive Power Analysis
* **News Sentiment**: Demonstrated significant predictive ability for AAPL, NVDA, PLTR, TSLA, and SPY
* **Reddit Sentiment**: Showed limited predictive power, with some significance for AAPL, TSLA, and NVDA
* **Granger Causality**: News sentiment showed stronger causal relationships with stock prices across multiple lags

### Direction Prediction Performance
* **Best Performers**: 
  - TSLA Reddit model: 60.69% accuracy
  - NVDA Reddit model: 62.07% accuracy
* **News vs. Reddit**: Mixed results with news sentiment performing better for some stocks (AAPL, TSLA) and Reddit for others (NVDA)

### Model Performance
* **Trading Strategy**: Backtested strategy achieved 1.123% return during study period
* **Combined Models**: Models incorporating both sentiment sources generally outperformed single-source models

## Technical Implementation

### Project Structure
```
experiment/
├── dataset/                    # Processed data files
├── Hypo2_analysis/            # Hypothesis 2 analysis results
├── meta_analysis_results/     # Meta-analysis findings
├── sentiment_analysis/        # Core sentiment analysis scripts
└── backup/                    # Historical analysis versions
```

### Key Scripts
* `meta_analysis.py`: Main meta-analysis implementation
* `time_series_analysis_new.py`: Time series analysis and visualization
* `predictive_power_analysis.py`: Predictive modeling and evaluation
* `hypothesis2_analysis.py`: Detailed hypothesis testing

### Data Processing
* Sentiment classification using Deepseek LLM
* Time series alignment and preprocessing
* Statistical significance testing
* Visualization generation for trend analysis

## Results Summary

### Statistical Significance
* **Correlation Coefficients**: Range from -0.2566 to 0.8429
* **P-values**: Most correlations significant at p < 0.05 level
* **Granger Causality**: News sentiment shows stronger causal relationships (p < 0.001 for multiple stocks)

### Investment Implications
1. **Source Selection**: Different stocks show varying sensitivity to news vs. Reddit sentiment
2. **Combined Approach**: Using both sentiment sources provides more comprehensive market view
3. **Timing Considerations**: News sentiment shows delayed effects, while Reddit provides immediate reactions
4. **Stock-Specific Patterns**: Each stock exhibits unique sentiment-price relationships

## Future Research Directions

1. **Expanded Dataset**: Include more stocks across different sectors and market caps
2. **Extended Timeframe**: Analyze longer periods to capture different market cycles
3. **Advanced Sentiment Models**: Explore more sophisticated sentiment extraction methods
4. **Market Condition Analysis**: Study sentiment effectiveness during bull/bear markets
5. **Real-time Trading**: Develop and backtest real-time sentiment-based trading strategies
6. **Cross-platform Analysis**: Include additional social media platforms (Twitter, LinkedIn)

---
