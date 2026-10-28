# Energy Price Forecasting and Concept Drift Exploration

An electricity-price modelling project for the ALB0331 point of connection, combining data preprocessing, exploratory analysis, outlier detection, SARIMA residual analysis, ADWIN drift monitoring and LightGBM regression.

## Included work

- `energy_price_forecasting.ipynb`: source from the final local notebook, with saved outputs and execution counts cleared for a compact, readable archive. Code cells are preserved.
- `preprocessing.py`: original preprocessing implementation, including calendar/cyclical features and ALB0331 filtering.
- `results/predicted_vs_actual_historical.csv`: previously saved model predictions and actual values.
- `results/historical_results_summary.json`: extracted historical metric lines and artifact counts, with evaluation context.
- `requirements.txt`: starting dependency list derived from imports; not a tested lockfile.

## Run locally

1. Create a separate Python environment and install `pip install -r requirements.txt`.
2. Obtain the authorised electricity-price CSV inputs separately. Raw input data and trained model files are not included.
3. Configure `folder_path` in `preprocessing.py` (originally `D:/energy_price_predn/energy_data1`), then run `python preprocessing.py`.
4. Confirm `transformed_energy_data_ALB0331.csv` was generated. The notebook expects it at `/content/transformed_energy_data_ALB0331.csv`; update those paths for your environment or use the corresponding Colab location.
5. Launch `jupyter lab` and review/run notebook cells in order. The notebook retains an inline dependency-installation cell and a later consolidated modelling cell that repeats earlier modelling work.

## Method and evaluation context

The original notebook is preserved as an exploratory project. Its historical scores are not valid evidence of future forecasting accuracy:

- SARIMA is fitted before the split, and `SARIMA_Residuals` contains the observed target price minus its fitted value. Supplying that feature to LightGBM leaks target information into evaluation.
- The train/test split is random, not chronological. This does not establish performance on future periods.
- The ADWIN retraining branch reuses the same original training data; it does not demonstrate adaptation using newly observed data.
- Outlier-filtered and full-test comparisons evaluate different populations.
- A quantity labelled `Accuracy` in the original notebook is `1 - MAPE`, not classification accuracy. It is not promoted as an accuracy claim here.
- The preprocessing lag-like fields and predicted-versus-actual plot alignment should be reviewed before reuse.

A future forecasting benchmark would require target-independent features available at prediction time, correctly constructed past-only lags, chronological/walk-forward evaluation and a separately evaluated drift adaptation policy. These revisions have not been implemented in this archive.

## Provenance and validation

Curated from `D:/energy_price_predn/Final_Project`. Original local files were not modified. Notebook structure, Python syntax (excluding the notebook shell-install line), credential-pattern checks and saved-prediction numeric parsing were checked. The notebook was not retrained or executed end to end during publication.
