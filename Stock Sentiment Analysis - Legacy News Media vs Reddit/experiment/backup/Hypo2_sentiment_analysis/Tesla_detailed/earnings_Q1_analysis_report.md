# Tesla earnings_Q1 Event Period Analysis Report

## 1. Basic Statistics

### 1.1 Data Overview
- Analysis Period: 2024-04-16 to 2024-04-30
- Sample Size: 11 trading days

### 1.2 Correlation Analysis
- News-Stock Correlation: 0.7848
- Reddit-Stock Correlation: 0.4197

## 2. Granger Causality Test Results

### 2.1 News -> Stock
- Lag 1 days: p-value = 0.3456
- Lag 2 days: p-value = 0.5382

### 2.2 Reddit -> Stock
- Lag 1 days: p-value = 0.3759
- Lag 2 days: p-value = 0.0339

## 3. Prediction Performance

### 3.1 News Sentiment Metrics
- accuracy: 0.5000
- precision: 0.0000
- recall: 0.0000
- f1_score: 0.0000

### 3.2 Reddit Sentiment Metrics
- accuracy: 0.5000
- precision: 0.0000
- recall: 0.0000
- f1_score: 0.0000

## 4. Conclusions

During the earnings_Q1 event period:

1. News sentiment shows stronger predictive power, with a correlation advantage of 0.3651
2. News sentiment shows significance in 0 lag periods, while Reddit sentiment shows significance in 1 lag periods
3. Reddit sentiment has higher prediction accuracy, with an advantage of 0.0000
