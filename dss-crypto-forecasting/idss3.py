import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from catboost import CatBoostRegressor

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Load dataset
df = pd.read_csv('engineered_output.csv')
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = df.drop(columns=['Open time', 'Close', 'Target'])
y = df['Target'].values
feature_cols = X.columns.to_list()

# Load models
catboost_model = CatBoostRegressor()
catboost_model.load_model('catboost_model.cbm')

models = {
    'CatBoost': catboost_model,
    'GRU': load_model('gru_price_prediction_model.h5'),
    'LSTM': load_model('lstm_model.h5'),
    'LinearRegression': joblib.load('price_prediction_model.pkl'),
    'RandomForest': joblib.load('random_forest_model.pkl'),
    'XGBoost': joblib.load('xgb_model.pkl'),
}

# Load scalers
scalers = {
    'CatBoost': joblib.load('catboost_scaler.pkl'),
    'GRU': joblib.load('gru_scaler.pkl'),
    'LSTM': joblib.load('lstm_scaler.pkl'),
    'LinearRegression': joblib.load('lr_scaler.pkl'),
    'RandomForest': joblib.load('rf_scaler.pkl'),
    'XGBoost': joblib.load('xgb_scaler.pkl'),
}

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "flatten"):
        y_pred = y_pred.flatten()
    elif isinstance(y_pred, (list, tuple)):
        y_pred = np.array(y_pred).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, mae, r2

# Insight explanations per model
def model_insights(model_name):
    insights = {
        'CatBoost': (
            "CatBoost is a powerful gradient boosting decision tree algorithm "
            "that handles categorical variables natively and reduces overfitting "
            "through ordered boosting. Its strong feature interaction modeling "
            "makes it highly effective for tabular data with complex patterns. "
            "In this dataset, it balances bias and variance well, but deep "
            "sequential patterns might be better captured by RNN models."
        ),
        'GRU': (
            "GRU (Gated Recurrent Unit) is a recurrent neural network architecture "
            "designed for sequential data. It efficiently captures temporal dependencies "
            "and is less complex than LSTM, reducing risk of overfitting. The GRU's "
            "ability to model time-series dynamics explains its strong performance, "
            "especially in forecasting stock or crypto prices where past patterns matter."
        ),
        'LSTM': (
            "LSTM (Long Short-Term Memory) networks are specifically designed to handle "
            "long-term dependencies in sequence data, addressing vanishing gradient issues "
            "common in vanilla RNNs. While effective in capturing complex temporal relationships, "
            "LSTMs are prone to overfitting if not properly regularized and may require "
            "more data or tuning than GRU."
        ),
        'LinearRegression': (
            "Linear Regression assumes a linear relationship between features and target, "
            "which limits its ability to capture non-linearities present in financial data. "
            "Its simplicity and interpretability are advantages, but it underperforms deep "
            "or ensemble models that model complex interactions."
        ),
        'RandomForest': (
            "Random Forest builds an ensemble of decision trees using bagging, which reduces "
            "variance and overfitting. It models non-linear relationships and feature interactions "
            "well but may not fully capture sequential dependencies present in time series."
        ),
        'XGBoost': (
            "XGBoost is a gradient boosting algorithm optimized for speed and performance. "
            "It effectively handles missing data and complex feature interactions. Despite its "
            "power, it lacks explicit sequence modeling capabilities, which can limit performance "
            "on temporal data compared to RNNs."
        )
    }
    return insights.get(model_name, "No detailed insights available for this model.")

def feature_importance_available(model_name):
    return model_name in ['CatBoost', 'RandomForest', 'XGBoost']

def get_feature_importance(model_name, model):
    if model_name == 'CatBoost':
        return model.get_feature_importance()
    elif model_name == 'RandomForest':
        return model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    elif model_name == 'XGBoost':
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            booster = model.get_booster()
            fmap = booster.get_score(importance_type='weight')
            return np.array([fmap.get(f, 0.) for f in feature_cols])
        except Exception:
            return None
    else:
        return None

st.set_page_config(layout='wide')
st.title("Intelligent Decision Support System: Crypto Price Forecasting")

results = []

# Evaluate each model
for name in models:
    model = models[name]
    scaler = scalers.get(name)
    if scaler is None:
        st.warning(f"No scaler found for model {name}. Skipping...")
        continue

    if name in ['GRU', 'LSTM']:
        X_scaled = scaler.transform(X)
        TIME_STEPS = 10
        X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)
        test_size = int(len(X_seq) * 0.2)
        X_test = X_seq[-test_size:]
        y_test = y_seq[-test_size:]
        date_index = df['Open time'].iloc[-(test_size):].values
    else:
        X_scaled = scaler.transform(X)
        test_size = int(len(X) * 0.2)
        X_test = X_scaled[-test_size:]
        y_test = y[-test_size:]
        date_index = df['Open time'].iloc[-test_size:].values

    y_pred, rmse, mae, r2 = evaluate_model(name, model, X_test, y_test)

    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2,
        'y_pred': y_pred,
        'y_test': y_test,
        'date_index': date_index
    })

    pd.DataFrame({
        'Date': date_index,
        'Actual': y_test,
        'Predicted': y_pred
    }).to_csv(f'predicted_vs_actual_{name}.csv', index=False)

# Summary table
st.subheader("Model Performance Summary")
df_results = pd.DataFrame(results)
st.dataframe(df_results.drop(columns=['y_pred', 'y_test', 'date_index']).set_index('Model'))

# Select model to analyze
model_to_view = st.selectbox("Select model for detailed analysis:", df_results['Model'].tolist())

selected_result = next((r for r in results if r['Model'] == model_to_view), None)
if selected_result:
    # Plot actual vs predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_result['date_index'], y=selected_result['y_test'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=selected_result['date_index'], y=selected_result['y_pred'], mode='lines', name='Predicted'))
    fig.update_layout(title=f'Actual vs Predicted for {model_to_view}', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.markdown(f"### Evaluation Metrics for {model_to_view}")
    st.markdown(f"- **RMSE:** {selected_result['RMSE']:.4f}")
    st.markdown(f"- **MAE:** {selected_result['MAE']:.4f}")
    st.markdown(f"- **R2 Score:** {selected_result['R2 Score']:.4f}")

    # Performance comparison statements
    st.markdown(f"### Performance Comparison for {model_to_view}")
    selected_rmse = selected_result['RMSE']
    for res in results:
        if res['Model'] == model_to_view:
            continue
        diff_percent = (res['RMSE'] - selected_rmse) / selected_rmse * 100
        if diff_percent > 0:
            st.markdown(f"✅ {model_to_view} outperforms {res['Model']} by {diff_percent:.2f}% lower RMSE.")
        else:
            st.markdown(f"⚠️ {model_to_view} performs worse than {res['Model']} by {abs(diff_percent):.2f}% higher RMSE.")

    # Technical insights
    st.markdown("### Technical Insights & Research Gap Analysis")
    insight_text = model_insights(model_to_view)
    st.write(insight_text)

    # Highlight research gap closed by this model's approach
    gap_text = {
        'CatBoost': (
            "Research Gap Closed: This project demonstrates that CatBoost's "
            "native handling of categorical features and ordered boosting "
            "effectively reduces overfitting in volatile crypto data, which "
            "is often overlooked in financial time series forecasting research."
        ),
        'GRU': (
            "Research Gap Closed: GRU models efficiently capture temporal "
            "dependencies with fewer parameters than LSTM, reducing overfitting "
            "on relatively small datasets—a critical insight for crypto price prediction."
        ),
        'LSTM': (
            "Research Gap Closed: While LSTM excels at modeling long-term dependencies, "
            "this project highlights the trade-off between model complexity and overfitting "
            "in crypto price forecasting, emphasizing the need for regularization."
        ),
        'LinearRegression': (
            "Research Gap Closed: Linear models, despite simplicity, serve as important "
            "baselines and highlight non-linear complexities in crypto price data."
        ),
        'RandomForest': (
            "Research Gap Closed: This analysis underscores Random Forest's utility in "
            "capturing non-linear feature interactions but its limitations with sequential data."
        ),
        'XGBoost': (
            "Research Gap Closed: This project benchmarks XGBoost's strength in handling "
            "heterogeneous tabular features but demonstrates its relative weakness in "
            "capturing temporal dynamics compared to RNNs."
        ),
    }
    st.markdown(gap_text.get(model_to_view, "No specific research gap statement available."))

    # Feature Importance plot
    if feature_importance_available(model_to_view):
        st.markdown("### Feature Importance")
        importance = get_feature_importance(model_to_view, models[model_to_view])
        if importance is not None:
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            fig_bar = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h'
            ))
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), title=f'Feature Importance for {model_to_view}')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")
    else:
        st.info(f"Feature importance is not applicable for {model_to_view} (likely a black-box model).")
