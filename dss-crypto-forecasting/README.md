# Cryptocurrency Forecasting Decision Support System

An exploratory BNB/USD price-prediction project combining hourly market-data preprocessing, technical indicators, six forecasting models and an interactive Streamlit comparison dashboard.

## Workflow and source

- `2.py`: resample minute-level Binance candles to hourly observations and create calendar/cyclical features, RSI, MACD, moving averages, volatility, lagged prices and other indicators.
- `cat.py`, `gru.py`, `lstm.py`, `linear_reg.py`, `rf.py`, `xgbhoost.py`: train CatBoost, GRU, LSTM, Linear Regression, Random Forest and XGBoost against a shifted next-hour Close target.
- `train_lgbm.py`: an additional LightGBM experiment, separate from the six-model dashboard.
- `idss3.py`: dashboard with model comparison, RMSE/MAE/R2, predicted-versus-actual plots and feature importance where supported.
- `results/`: full historical prediction tables for six models and metrics recalculated from those tables.

## Setup

Create a separate Python environment and install `pip install -r requirements.txt`. Dependencies are inferred from imports, not a verified lockfile. Use TensorFlow-compatible Python and dependency versions for the recurrent models.

Run commands from this project directory:

1. Obtain an authorised `BNBUSD_1m_Binance.csv` input separately. Check its schema against `2.py`; it needs timestamp, OHLCV and trade-volume fields. Raw exchange datasets are excluded.
2. Run `python 2.py` to generate `engineered_output.csv`. Verify chronological ordering and the resulting feature columns.
3. Train the six dashboard models: `python cat.py`, `python gru.py`, `python lstm.py`, `python linear_reg.py`, `python rf.py` and `python xgbhoost.py`.
4. Confirm the corresponding model/scaler files exist in the current directory, then run `streamlit run idss3.py`.
5. Optionally run `python train_lgbm.py` for the separate LightGBM experiment.

The dashboard loads all six model/scaler pairs at startup; generated model files must exist before launch. It writes prediction tables to the working directory. Existing trained weights are excluded; regenerate them locally.

## Saved results and evaluation context

The six included prediction files contain approximately 13,000 rows each. `results/historical_metrics.json` recalculates RMSE, MAE and R2 from the full saved predictions. This verifies arithmetic on saved outputs; it does not establish how the saved models were trained.

The original training scripts use chronological train/test splits, but most fit their feature scaler on the entire dataset before splitting. Train-only scaling is required for a clean evaluation. The tabular and recurrent model outputs have different row counts; align timestamps and shifted target horizons before making direct model rankings. Verify the dashboard's sequence construction and time labels as part of reproduction.

The original dashboard's generic “Research Gap Closed” text is preserved as part of the source, but should not be treated as a demonstrated research contribution or comparative finding. No profitability or independently validated forecasting claim is made.

## Archive scope

Curated from `D:/DSS`, retaining original script names and code. Duplicate copies in `DSS1`, earlier dashboard variants, placeholder comparison scripts, raw market datasets, environment folders and trained weights are omitted. The original local project was not modified.

Python syntax, credential patterns and saved prediction numeric values were checked before publication. Full model training and the live dashboard were not run during this upload.
