# Hypothesis 2: Predictive Power Analysis Report

## Research Question
Does sentiment from different sources (news media vs Reddit) have different predictive power for stock price movements?

## Hypotheses
- H0: Sentiment from legacy news media and Reddit does not significantly predict stock price movements.
- H1: Reddit sentiment has a stronger correlation with short-term stock price movements than legacy news media.

## Results Summary

### Correlation Analysis

#### AAPL
- News Correlation: 0.7963 (p=0.0000)
- Reddit Correlation: 0.3708 (p=0.0000)

#### META
- News Correlation: -0.1904 (p=0.0020)
- Reddit Correlation: -0.1713 (p=0.0056)

#### TSLA
- News Correlation: 0.8429 (p=0.0000)
- Reddit Correlation: 0.7850 (p=0.0000)

#### NVDA
- News Correlation: 0.6718 (p=0.0000)
- Reddit Correlation: 0.2076 (p=0.0008)

#### PLTR
- News Correlation: -0.2566 (p=0.0000)
- Reddit Correlation: 0.0174 (p=0.7804)

#### SPY
- News Correlation: 0.0378 (p=0.5440)
- Reddit Correlation: 0.2151 (p=0.0005)

### Prediction Results

#### AAPL

##### News Sentiment Model
- R² Score: -3.5105
- MSE: 0.3895
- Direction Prediction:
  - Accuracy: 0.5379
  - Baseline Accuracy: 0.5931
  - F1 Score: 0.5186
  - Up/Down Ratio: 0.5034
- Coefficient: 0.9107

##### Reddit Sentiment Model
- R² Score: -4.2796
- MSE: 0.6992
- Direction Prediction:
  - Accuracy: 0.4621
  - Baseline Accuracy: 0.5931
  - F1 Score: 0.4256
  - Up/Down Ratio: 0.5034
- Coefficient: 0.4454

#### META

##### News Sentiment Model
- R² Score: -7.5762
- MSE: 1.6755
- Direction Prediction:
  - Accuracy: 0.5172
  - Baseline Accuracy: 0.5931
  - F1 Score: 0.4414
  - Up/Down Ratio: 0.4552
- Coefficient: -0.3993

##### Reddit Sentiment Model
- R² Score: -1.2610
- MSE: 0.6807
- Direction Prediction:
  - Accuracy: 0.4759
  - Baseline Accuracy: 0.5931
  - F1 Score: 0.3882
  - Up/Down Ratio: 0.4552
- Coefficient: -0.1205

#### TSLA

##### News Sentiment Model
- R² Score: -0.9375
- MSE: 0.2445
- Direction Prediction:
  - Accuracy: 0.5724
  - Baseline Accuracy: 0.6207
  - F1 Score: 0.4596
  - Up/Down Ratio: 0.4069
- Coefficient: 0.9586

##### Reddit Sentiment Model
- R² Score: -1.7863
- MSE: 0.3453
- Direction Prediction:
  - Accuracy: 0.6069
  - Baseline Accuracy: 0.6207
  - F1 Score: 0.4942
  - Up/Down Ratio: 0.4069
- Coefficient: 0.9707

#### NVDA

##### News Sentiment Model
- R² Score: -3.0393
- MSE: 0.5423
- Direction Prediction:
  - Accuracy: 0.4414
  - Baseline Accuracy: 0.5310
  - F1 Score: 0.4455
  - Up/Down Ratio: 0.4828
- Coefficient: 0.7849

##### Reddit Sentiment Model
- R² Score: -6.7550
- MSE: 1.0655
- Direction Prediction:
  - Accuracy: 0.6207
  - Baseline Accuracy: 0.5310
  - F1 Score: 0.5848
  - Up/Down Ratio: 0.4828
- Coefficient: 0.0793

#### PLTR

##### News Sentiment Model
- R² Score: -17.3708
- MSE: 1.0336
- Direction Prediction:
  - Accuracy: 0.4828
  - Baseline Accuracy: 0.5586
  - F1 Score: 0.5231
  - Up/Down Ratio: 0.4552
- Coefficient: -0.1305

##### Reddit Sentiment Model
- R² Score: -17.4867
- MSE: 1.1594
- Direction Prediction:
  - Accuracy: 0.4966
  - Baseline Accuracy: 0.5586
  - F1 Score: 0.3680
  - Up/Down Ratio: 0.4552
- Coefficient: 0.1498

#### SPY

##### News Sentiment Model
- R² Score: -1.5796
- MSE: 1.0230
- Direction Prediction:
  - Accuracy: 0.4276
  - Baseline Accuracy: 0.6069
  - F1 Score: 0.4483
  - Up/Down Ratio: 0.4759
- Coefficient: -0.0547

##### Reddit Sentiment Model
- R² Score: -1.4568
- MSE: 0.9775
- Direction Prediction:
  - Accuracy: 0.4552
  - Baseline Accuracy: 0.6069
  - F1 Score: 0.4497
  - Up/Down Ratio: 0.4759
- Coefficient: 0.0362

## Granger Causality Analysis

### AAPL

#### News Sentiment → Stock Price
- Lag 1: p = 0.0000 ***
- Lag 2: p = 0.0000 ***
- Lag 3: p = 0.0000 ***
- Lag 4: p = 0.0000 ***
- Lag 5: p = 0.0000 ***

#### Reddit Sentiment → Stock Price
- Lag 1: p = 0.0032 **
- Lag 2: p = 0.0162 *
- Lag 3: p = 0.0344 *
- Lag 4: p = 0.0776 
- Lag 5: p = 0.1661 

### META

#### News Sentiment → Stock Price
- Lag 1: p = 0.1430 
- Lag 2: p = 0.0416 *
- Lag 3: p = 0.0183 *
- Lag 4: p = 0.0263 *
- Lag 5: p = 0.0181 *

#### Reddit Sentiment → Stock Price
- Lag 1: p = 0.2218 
- Lag 2: p = 0.4002 
- Lag 3: p = 0.6279 
- Lag 4: p = 0.5087 
- Lag 5: p = 0.5817 

### TSLA

#### News Sentiment → Stock Price
- Lag 1: p = 0.0000 ***
- Lag 2: p = 0.0000 ***
- Lag 3: p = 0.0000 ***
- Lag 4: p = 0.0000 ***
- Lag 5: p = 0.0000 ***

#### Reddit Sentiment → Stock Price
- Lag 1: p = 0.0017 **
- Lag 2: p = 0.0019 **
- Lag 3: p = 0.0023 **
- Lag 4: p = 0.0056 **
- Lag 5: p = 0.0101 *

### NVDA

#### News Sentiment → Stock Price
- Lag 1: p = 0.0000 ***
- Lag 2: p = 0.0000 ***
- Lag 3: p = 0.0000 ***
- Lag 4: p = 0.0000 ***
- Lag 5: p = 0.0000 ***

#### Reddit Sentiment → Stock Price
- Lag 1: p = 0.0017 **
- Lag 2: p = 0.0019 **
- Lag 3: p = 0.0008 ***
- Lag 4: p = 0.0022 **
- Lag 5: p = 0.0058 **

### PLTR

#### News Sentiment → Stock Price
- Lag 1: p = 0.7254 
- Lag 2: p = 0.0187 *
- Lag 3: p = 0.0003 ***
- Lag 4: p = 0.0010 **
- Lag 5: p = 0.0026 **

#### Reddit Sentiment → Stock Price
- Lag 1: p = 0.8911 
- Lag 2: p = 0.1359 
- Lag 3: p = 0.1447 
- Lag 4: p = 0.2357 
- Lag 5: p = 0.2337 

### SPY

#### News Sentiment → Stock Price
- Lag 1: p = 0.0001 ***
- Lag 2: p = 0.0000 ***
- Lag 3: p = 0.0000 ***
- Lag 4: p = 0.0000 ***
- Lag 5: p = 0.0000 ***

#### Reddit Sentiment → Stock Price
- Lag 1: p = 0.0410 *
- Lag 2: p = 0.1463 
- Lag 3: p = 0.2381 
- Lag 4: p = 0.3900 
- Lag 5: p = 0.5223 

## Direction Prediction Analysis

### AAPL

#### News Model
- Accuracy: 0.5379
- Baseline Accuracy: 0.5931
- F1 Score: 0.5186
- Up/Down Ratio: 0.5034

#### Reddit Model
- Accuracy: 0.4621
- Baseline Accuracy: 0.5931
- F1 Score: 0.4256
- Up/Down Ratio: 0.5034

### META

#### News Model
- Accuracy: 0.5172
- Baseline Accuracy: 0.5931
- F1 Score: 0.4414
- Up/Down Ratio: 0.4552

#### Reddit Model
- Accuracy: 0.4759
- Baseline Accuracy: 0.5931
- F1 Score: 0.3882
- Up/Down Ratio: 0.4552

### TSLA

#### News Model
- Accuracy: 0.5724
- Baseline Accuracy: 0.6207
- F1 Score: 0.4596
- Up/Down Ratio: 0.4069

#### Reddit Model
- Accuracy: 0.6069
- Baseline Accuracy: 0.6207
- F1 Score: 0.4942
- Up/Down Ratio: 0.4069

### NVDA

#### News Model
- Accuracy: 0.4414
- Baseline Accuracy: 0.5310
- F1 Score: 0.4455
- Up/Down Ratio: 0.4828

#### Reddit Model
- Accuracy: 0.6207
- Baseline Accuracy: 0.5310
- F1 Score: 0.5848
- Up/Down Ratio: 0.4828

### PLTR

#### News Model
- Accuracy: 0.4828
- Baseline Accuracy: 0.5586
- F1 Score: 0.5231
- Up/Down Ratio: 0.4552

#### Reddit Model
- Accuracy: 0.4966
- Baseline Accuracy: 0.5586
- F1 Score: 0.3680
- Up/Down Ratio: 0.4552

### SPY

#### News Model
- Accuracy: 0.4276
- Baseline Accuracy: 0.6069
- F1 Score: 0.4483
- Up/Down Ratio: 0.4759

#### Reddit Model
- Accuracy: 0.4552
- Baseline Accuracy: 0.6069
- F1 Score: 0.4497
- Up/Down Ratio: 0.4759

### Significance Levels
- *** p < 0.001
- ** p < 0.01
- * p < 0.05
