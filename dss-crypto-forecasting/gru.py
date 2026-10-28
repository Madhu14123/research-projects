import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import joblib

# ======================
# Load and Prepare Data
# ======================
df = pd.read_csv('engineered_output.csv')

# Sort by time to maintain time series order
df = df.sort_values('Open time')

# Create the target column: Close price 1 hour ahead
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Define feature columns: drop 'Open time', 'Close', and 'Target'
drop_cols = ['Open time', 'Close', 'Target']
X = df.drop(columns=drop_cols)
y = df['Target'].values

# Save the feature column names
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'gru_feature_columns.pkl')

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'gru_scaler.pkl')

# ======================
# Create sequences for GRU
# ======================
def create_sequences(features: np.ndarray, target: np.ndarray, time_steps: int = 10):
    X_seq, y_seq = [], []
    for i in range(len(features) - time_steps):
        X_seq.append(features[i:i + time_steps])
        y_seq.append(target[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

TIME_STEPS = 10
X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)

# ======================
# Train-Test Split (No Shuffle)
# ======================
split_index = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# ======================
# Define and Train GRU Model
# ======================
model = Sequential([
    GRU(64, input_shape=(TIME_STEPS, X.shape[1])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=2
)

# ======================
# Predict and Evaluate
# ======================
y_pred = model.predict(X_test).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("GRU Model Evaluation:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")

# ======================
# Save the Model
# ======================
model.save('gru_price_prediction_model.h5')
print("GRU model and scaler saved successfully.")
