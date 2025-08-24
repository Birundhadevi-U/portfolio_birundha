# stock_predictor_app.py

import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set page
st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Prediction App")

# Function to load stock data
def load_data(ticker, period='5y'):
    data = yf.download(ticker, period=period)
    return data[['Close']]

# Preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler

# Create dataset for LSTM
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# User input
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, TSLA)", "AAPL")

if st.button("Predict"):
    with st.spinner("Training model and predicting..."):
        df = load_data(ticker)
        st.subheader("📉 Historical Closing Prices")
        st.line_chart(df)

        scaled_data, scaler = preprocess_data(df)
        X, y = create_dataset(scaled_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=5, batch_size=64, verbose=0)

        x_input = scaled_data[-60:].reshape(1, -1)
        x_input = x_input.reshape((1, 60, 1))

        predictions = []
        for _ in range(30):
            pred = model.predict(x_input, verbose=0)[0][0]
            predictions.append(pred)
            x_input = np.append(x_input[:, 1:, :], [[[pred]]], axis=1)

        forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
        forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicted Close'])

        st.subheader("🔮 Next 30 Days Price Prediction")
        st.line_chart(forecast_df)
        st.dataframe(forecast_df.round(2))
        st.success("Prediction complete ✅")
