# Tesla earnings_Q4 Event Period Analysis Report

## 1. Basic Statistics

### 1.1 Data Overview
- Analysis Period: 2025-01-15 to 2025-01-29
- Sample Size: 10 trading days

### 1.2 Correlation Analysis
- News-Stock Correlation: 0.8731
- Reddit-Stock Correlation: 0.7077

## 2. Granger Causality Test Results

### 2.1 News -> Stock
- Lag 1 days: p-value = 0.5387
- Lag 2 days: p-value = 0.0990

### 2.2 Reddit -> Stock
- Lag 1 days: p-value = 0.2971
- Lag 2 days: p-value = 0.5911

## 3. Prediction Performance

### 3.1 News Sentiment Metrics
- accuracy: 0.8889
- precision: 1.0000
- recall: 0.8889
- f1_score: 0.9412

### 3.2 Reddit Sentiment Metrics
- accuracy: 1.0000
- precision: 1.0000
- recall: 1.0000
- f1_score: 1.0000

## 4. Conclusions

During the earnings_Q4 event period:

1. News sentiment shows stronger predictive power, with a correlation advantage of 0.1654
2. News sentiment shows significance in 0 lag periods, while Reddit sentiment shows significance in 0 lag periods
3. Reddit sentiment has higher prediction accuracy, with an advantage of 0.1111
