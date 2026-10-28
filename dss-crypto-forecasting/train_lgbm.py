import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('engineered_output.csv')  # update with your csv file path

# Convert 'Open time' to datetime
df['Open time'] = pd.to_datetime(df['Open time'])

# Sort by datetime just in case
df = df.sort_values('Open time')

# Create target: Close price 1 hour ahead
df['target'] = df['Close'].shift(-1)
df = df.dropna(subset=['target'])

# Use all features except 'Open time' and 'target'
excluded_cols = ['Open time', 'target']
feature_cols = [col for col in df.columns if col not in excluded_cols]

X_clean = df[feature_cols]
y_clean = df['target']

# Train-test split without shuffling (time-series)
X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=0.2, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

lgb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = lgb_model.predict(X_val_scaled)

# Metrics
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# Save model and scaler
joblib.dump(lgb_model, 'lgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler saved successfully.")

# Feature importance plot
importances = lgb_model.feature_importances_
feature_names = X_clean.columns

plt.figure(figsize=(10,6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.show()
