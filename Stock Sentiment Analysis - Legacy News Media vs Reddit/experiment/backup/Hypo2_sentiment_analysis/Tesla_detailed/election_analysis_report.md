# Tesla election Event Period Analysis Report

## 1. Basic Statistics

### 1.1 Data Overview
- Analysis Period: 2024-10-29 to 2024-11-12
- Sample Size: 11 trading days

### 1.2 Correlation Analysis
- News-Stock Correlation: 0.7262
- Reddit-Stock Correlation: 0.8214

## 2. Granger Causality Test Results

### 2.1 News -> Stock
- Lag 1 days: p-value = 0.3988
- Lag 2 days: p-value = 0.6876

### 2.2 Reddit -> Stock
- Lag 1 days: p-value = 0.6663
- Lag 2 days: p-value = 0.9779

## 3. Prediction Performance

### 3.1 News Sentiment Metrics
- accuracy: 1.0000
- precision: 1.0000
- recall: 1.0000
- f1_score: 1.0000

### 3.2 Reddit Sentiment Metrics
- accuracy: 1.0000
- precision: 1.0000
- recall: 1.0000
- f1_score: 1.0000

## 4. Conclusions

During the election event period:

1. Reddit sentiment shows stronger predictive power, with a correlation advantage of 0.0952
2. News sentiment shows significance in 0 lag periods, while Reddit sentiment shows significance in 0 lag periods
3. Reddit sentiment has higher prediction accuracy, with an advantage of 0.0000
