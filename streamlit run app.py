# Stock Price Prediction App using Streamlit and LSTM

import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load Data
# -----------------------------
def load_data(ticker, period='5y'):
    df = yf.download(ticker, period=period)
    return df[['Close']]

# -----------------------------
# Step 2: Preprocess Data
# -----------------------------
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# -----------------------------
# Step 3: Create Dataset
# -----------------------------
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# -----------------------------
# Step 4: Build Model
# -----------------------------
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# -----------------------------
# Step 5: Streamlit UI
# -----------------------------
st.title("📈 Stock Price Prediction App")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", value='AAPL')

if st.button("Predict"):
    # Load data
    df = load_data(ticker)
    st.write("### Historical Closing Price")
    st.line_chart(df)

    # Preprocess
    scaled_data, scaler = preprocess_data(df)
    X, y = create_dataset(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train model
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=64, verbose=0)

    # Predict last 60 days
    last_60 = scaled_data[-60:].reshape(1, 60, 1)
    predictions = []
    for _ in range(30):
        next_pred = model.predict(last_60)[0][0]
        predictions.append(next_pred)
        last_60 = np.append(last_60[:,1:,:], [[[next_pred]]], axis=1)

    # Rescale back
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

    # Plot
    st.write("### Next 30 Days Prediction")
    fig, ax = plt.subplots()
    ax.plot(predicted_prices, label='Predicted')
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.success("Prediction Complete ✅")
