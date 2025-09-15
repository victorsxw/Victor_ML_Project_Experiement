# Tesla earnings_Q3 Event Period Analysis Report

## 1. Basic Statistics

### 1.1 Data Overview
- Analysis Period: 2024-10-16 to 2024-10-30
- Sample Size: 11 trading days

### 1.2 Correlation Analysis
- News-Stock Correlation: 0.3802
- Reddit-Stock Correlation: 0.9653

## 2. Granger Causality Test Results

### 2.1 News -> Stock
- Lag 1 days: p-value = 0.0480
- Lag 2 days: p-value = 0.4684

### 2.2 Reddit -> Stock
- Lag 1 days: p-value = 0.1587
- Lag 2 days: p-value = 0.0009

## 3. Prediction Performance

### 3.1 News Sentiment Metrics
- accuracy: 0.7000
- precision: 0.7000
- recall: 1.0000
- f1_score: 0.8235

### 3.2 Reddit Sentiment Metrics
- accuracy: 0.7000
- precision: 0.7000
- recall: 1.0000
- f1_score: 0.8235

## 4. Conclusions

During the earnings_Q3 event period:

1. Reddit sentiment shows stronger predictive power, with a correlation advantage of 0.5851
2. News sentiment shows significance in 1 lag periods, while Reddit sentiment shows significance in 1 lag periods
3. Reddit sentiment has higher prediction accuracy, with an advantage of 0.0000
