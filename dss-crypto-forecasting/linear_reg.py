import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ======================
# Load and Prepare Data
# ======================
df = pd.read_csv('engineered_output.csv')

# Create the target: Close price 1 hour ahead
df['Target'] = df['Close'].shift(-1)

# Drop rows with missing target
df.dropna(inplace=True)

# Drop columns not used as features
drop_cols = ['Open time', 'Close', 'Target']
X = df.drop(columns=drop_cols)
y = df['Target']

# Save feature column names for consistency
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'lr_feature_columns.pkl')

# ======================
# Scale the Features
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'lr_scaler.pkl')

# ======================
# Train-Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False  # keep time order
)

# ======================
# Train the Model
# ======================
lr = LinearRegression()
lr.fit(X_train, y_train)

# ======================
# Predict and Evaluate
# ======================
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# ======================
# Save Model and Report
# ======================
joblib.dump(lr, 'price_prediction_model.pkl')

print("1-Hour Ahead Price Prediction using Linear Regression:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("Model, scaler, and feature list saved successfully.")
