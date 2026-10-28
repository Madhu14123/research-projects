import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv('engineered_output.csv')

# Shift Close to get next hour's price
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Drop unnecessary columns
X = df.drop(columns=['Close', 'Target', 'Open time'])
y = df['Target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80/20) without shuffling for time series
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("XGBoost Model:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save model and scaler
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'xgb_scaler.pkl')
