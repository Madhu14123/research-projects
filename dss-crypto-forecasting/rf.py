import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
df = pd.read_csv('engineered_output.csv')

# Shift Close to get the target (1 hour ahead)
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Features and target
X = df.drop(columns=['Close', 'Target', 'Open time'])
y = df['Target']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time-based train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Random Forest Regressor:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save model and scaler
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'rf_scaler.pkl')
