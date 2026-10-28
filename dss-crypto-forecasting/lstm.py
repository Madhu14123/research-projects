import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Load and prepare data
df = pd.read_csv('engineered_output.csv')
df = df.sort_values('Open time')
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = df.drop(columns=['Open time', 'Close', 'Target'])
y = df['Target'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for LSTM
def create_sequences(features: np.ndarray, target: np.ndarray, time_steps: int = 10):
    X_seq, y_seq = [], []
    for i in range(len(features) - time_steps):
        X_seq.append(features[i:i + time_steps])
        y_seq.append(target[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

TIME_STEPS = 10
X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)

# Train-test split (80/20)
split_index = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(TIME_STEPS, X.shape[1]), return_sequences=False),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=2)

# Predict
y_pred = model.predict(X_test).flatten()

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("LSTM Model:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save model and scaler
model.save('lstm_model.h5')
joblib.dump(scaler, 'lstm_scaler.pkl')
