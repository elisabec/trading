# Trading & Quantitative Research

A collection of quantitative research projects spanning predictive modelling, systematic signal development, and deep learning on alternative data.

---

## Projects

### 1. [Hedge Fund Stop Predictability](hedge-fund-stop-predictability/)
Forecasting hedge fund reporting stops and performance using XGBoost on 5,592 funds and 300k+ monthly observations. Published in the *Journal of Forecasting* (Wiley, 2020).

**87% accuracy** on reporting stop prediction · **75%** on absolute performance · **74%** on relative performance

`XGBoost` · `TimeSeriesSplit CV` · `scikit-learn` · `pandas`

---

### 2. [Quant Research — FX Signal Pipeline](quant-research/)
Systematic intraday mean-reversion signal using DCC-GARCH dynamic correlations. Includes a reusable backtesting engine, GARCH volatility modelling, and markout analysis.

**Sharpe 3.37** · In-sample backtest with grid search across signal types and parameters

`GARCH` · `DCC` · `PCA` · `arch` · `NumPy` · `pandas`

---

### 3. [Twitter Engagement Prediction](twitter-engagement-prediction/)
Predicting tweet engagement using 6 deep learning architectures — from bag-of-words baselines to BERT embeddings with metadata features. Final project for MIT Sloan's Hands-on Deep Learning (15.S04).

**64.57% accuracy** with BERT + metadata concatenated model

`TensorFlow/Keras` · `BERT` · `GloVe` · `Twitter API` · `scikit-learn`