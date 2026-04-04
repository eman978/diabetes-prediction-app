import streamlit as st
import pickle
import pandas as pd
import numpy as np


# 📥 Load Model & Scaler
@st.cache_resource
def load_models():
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('diabetes_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


model, scaler = load_models()

# 🎨 UI Setup
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Patient ke details enter karein taaki model predict kar sake ke diabetes hai ya nahi.")

# 📝 Input Fields
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, step=1, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01, value=0.5)
    age = st.number_input("Age", min_value=10, max_value=100, step=1, value=30)

# 🚀 Prediction Button
if st.button("🔍 Predict Diabetes"):
    try:
        # DataFrame banana zaroori hai taaki columns ka order training jaisa hi rahe
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })

        # Scale & Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # Result Display
        st.markdown("---")
        if prediction == 1:
            st.error("🔴 **Result: Diabetes Detected**\nPatient ko diabetes hone ka risk hai. Doctor se consult karein.")
        else:
            st.success("✅ **Result: No Diabetes**\nPatient mein diabetes ke koi clear signs nahi mile.")

    except Exception as e:
        st.error(f"⚠️ Koi error aaya: {e}")

# ℹ️ Footer
st.markdown("---")
st.caption("⚡ Model: Random Forest | 📊 Accuracy: ~80.5% | 🔒 Data locally process hota hai.")