import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model dan scaler
model = joblib.load('model_xgb.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Dashboard Prediksi Curah Hujan")

# Input user
suhu = st.slider("Suhu (°C)", 20.0, 40.0, 30.0)
kelembapan = st.slider("Kelembapan (%)", 10.0, 100.0, 70.0)
bulan = st.selectbox("Bulan", list(range(1, 13)))

# Prediksi
if st.button("Prediksi"):
    data = np.array([[suhu, kelembapan, bulan]])
    data_scaled = scaler.transform(data)
    prediksi = model.predict(data_scaled)
    st.success(f"Prediksi Curah Hujan: {prediksi[0]:.2f} mm")
