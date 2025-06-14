import streamlit as st
import pandas as pd
import joblib

# Load model dan preprocessing
model = joblib.load("models/model_obesitas.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("Prediksi Tingkat Obesitas Berdasarkan Gaya Hidup")

# Form input pengguna
def user_input():
    data = {
        "Age": st.slider("Usia", 10, 100, 25),
        "Height": st.slider("Tinggi Badan (meter)", 1.0, 2.5, 1.7),
        "Weight": st.slider("Berat Badan (kg)", 30, 200, 70),
        "FCVC": st.slider("Frekuensi makan sayur (1-3)", 1, 3, 2),
        "NCP": st.slider("Jumlah makan besar/hari", 1, 4, 3),
        "CH2O": st.slider("Konsumsi air (1-3)", 1, 3, 2),
        "FAF": st.slider("Aktivitas fisik", 0, 3, 1),
        "TUE": st.slider("Durasi pakai teknologi", 0, 2, 1),
        "Gender": st.selectbox("Jenis Kelamin", ["Male", "Female"]),
        "FAVC": st.selectbox("Konsumsi makanan tinggi kalori?", ["yes", "no"]),
        "SCC": st.selectbox("Pantau kalori harian?", ["yes", "no"]),
        "SMOKE": st.selectbox("Merokok?", ["yes", "no"]),
        "family_history_with_overweight": st.selectbox("Riwayat keluarga obesitas?", ["yes", "no"]),
        "CALC": st.selectbox("Konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"]),
        "CAEC": st.selectbox("Ngemil antar waktu makan?", ["no", "Sometimes", "Frequently", "Always"]),
        "MTRANS": st.selectbox("Transportasi biasa digunakan?", ["Public_Transportation", "Walking", "Automobile", "Bike", "Motorbike"])
    }
    return pd.DataFrame([data])

input_df = user_input()

# Encoding kategorikal
for col in label_encoders:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Scaling
input_scaled = scaler.transform(input_df)

# Prediksi
pred = model.predict(input_scaled)
label = target_encoder.inverse_transform(pred)

st.subheader("Hasil Prediksi:")
st.success(label[0])
