import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("scaler (5).pkl") 
model = joblib.load("random_forest_model (5).pkl") 

st.title("Stress Level Prediction App")
st.markdown("Fill in your details to estimate your stress level (1 to 10).")

# UI Inputs
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Male", "Female","Others"])
    age = st.number_input("Age", min_value=1, max_value=120, value=28)
    occupation = st.selectbox("Occupation", options=[
        "Sales Representative", "Software Engineer", "Teacher", "Nurse","Salesperson","Doctor","Accountant","Engineer","Lawyer","Scientist"
])
    sleep_duration = st.number_input("Sleep Duration (hrs)", min_value=0.0, max_value=24.0, value=6.0)
    quality_of_sleep = st.slider("Quality of Sleep (1-10)", min_value=1, max_value=10, value=6)
    physical_activity = st.slider("Physical Activity Level", min_value=0, max_value=100, value=30)

with col2:
    bmi_category = st.selectbox("BMI Category", options=["Normal Weight", "Overweight", "Obese"])
    heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=80)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=3000)
    bp_1 = st.number_input("Blood Pressure (Systolic)", min_value=50, max_value=200, value=120)
    bp_2 = st.number_input("Blood Pressure (Diastolic)", min_value=30, max_value=130, value=80)
    sleep_disorder = st.selectbox("Sleep Disorder", options=["None", "Insomnia", "Sleep Apnea"])

# Manual encoding
gender_map = {"Male": 0, "Female": 1,"Others": 2}
gender_encoded = gender_map[gender]
bmi_map = {"Normal Weight": 0, "Overweight": 1, "Obese": 2}
occupation_map = {
    "Sales Representative": 0,
    "Software Engineer": 1,
    "Teacher": 2,
    "Nurse": 3,
    "Salesperson": 4,
    "Doctor" : 5,
    "Accountant": 6,
    "Engineer": 7,
    "Lawyer": 8,
    "Scientist": 9
}
sleep_disorder_map = {
    "None": 0,
    "Insomnia": 1,
    "Sleep Apnea": 2
}

# Final input array
input_data = np.array([[
    gender_encoded,
    age,
    occupation_map[occupation],
    sleep_duration,
    quality_of_sleep,
    physical_activity,
    bmi_map[bmi_category],
    heart_rate,
    daily_steps,
    bp_1,
    bp_2,
    sleep_disorder_map[sleep_disorder]
]])

# Prediction
if st.button("Predict Stress Level"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        stress = round(prediction[0], 2)
        st.success(f"Predicted Stress Level: {stress} / 10")
    except Exception as e:
        st.error(f"Prediction error: {e}")