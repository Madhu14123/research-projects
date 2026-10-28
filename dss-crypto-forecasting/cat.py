import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import joblib

# ======================
# Load and Prepare Data
# ======================
df = pd.read_csv('engineered_output.csv')

# Create Target: 'Close' price shifted 1 step ahead
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Drop unused columns and define features (X) and target (y)
drop_cols = ['Close', 'Target', 'Open time']
X = df.drop(columns=drop_cols)
y = df['Target']

# ======================
# Scale the features
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# Train-Test Split (Time Series – No Shuffling)
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# ======================
# Initialize and Train CatBoost Regressor
# ======================
cat_model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=False,
    random_state=42
)
cat_model.fit(X_train, y_train)

# ======================
# Predict and Evaluate
# ======================
y_pred = cat_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("CatBoost Regressor Performance:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# ======================
# Save Model and Scaler
# ======================
cat_model.save_model('catboost_model.cbm')
joblib.dump(scaler, 'catboost_scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')  # Save column order
