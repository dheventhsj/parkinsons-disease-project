import streamlit as st
import pandas as pd
import pickle

MODEL_FILE = 'best_parkinsons_model.pkl'

with open(MODEL_FILE, 'rb') as f:
    saved_data = pickle.load(f)

scaler = saved_data['scaler']
model = saved_data['model']
features = saved_data['features']
targets = saved_data['targets']

st.set_page_config(page_title="Parkinson's UPDRS Prediction", layout="centered")

st.title("Parkinson's Disease UPDRS Prediction System")
st.write("Machine Learning-based Telemonitoring Application (Reduced Feature Version)")
st.markdown("---")

st.subheader("Enter Patient Voice Feature Values")

input_data = {}

# Group 1: Patient Information
with st.expander("Patient Information", expanded=True):
    if 'age' in features:
        input_data['age'] = st.number_input("Age", min_value=0, max_value=120, value=60, step=1)
    if 'sex' in features:
        sex_value = st.selectbox("Sex", ("Male", "Female"))
        input_data['sex'] = 1 if sex_value == "Male" else 0
    if 'test_time' in features:
        input_data['test_time'] = st.number_input("Test Time (days since first visit)", value=0.0, step=0.1)

# Group 2: Jitter Features
with st.expander("Jitter Features"):
    for col in features:
        if "Jitter" in col:
            if col not in input_data:
                input_data[col] = st.number_input(col, value=0.0, step=0.00001, format="%.5f")

# Group 3: Shimmer Features
with st.expander("Shimmer Features"):
    for col in features:
        if "Shimmer" in col:
            if col not in input_data:
                input_data[col] = st.number_input(col, value=0.0, step=0.00001, format="%.5f")

# Group 4: Noise and Nonlinear Features
with st.expander("Noise and Nonlinear Features"):
    for col in features:
        if col not in input_data:
            input_data[col] = st.number_input(col, value=0.0, step=0.00001, format="%.5f")

st.markdown("---")


def severity_category(score: float) -> str:
    if score < 10:
        return "Mild"
    elif score < 30:
        return "Moderate"
    else:
        return "Severe"


if st.button("Predict UPDRS Scores"):
    X_df = pd.DataFrame([input_data])[features]
    X_scaled = scaler.transform(X_df)
    prediction = model.predict(X_scaled)[0]

    motor_updrs = float(prediction[0])
    total_updrs = float(prediction[1])

    motor_severity = severity_category(motor_updrs)
    total_severity = severity_category(total_updrs)

    st.subheader("Predicted Parkinson’s UPDRS Scores (Reduced Features)")
    st.write(f"Motor UPDRS: {motor_updrs:.2f}  ({motor_severity} severity)")
    st.write(f"Total UPDRS: {total_updrs:.2f}  ({total_severity} severity)")

    st.markdown("---")
    st.info(
        "Note: This system estimates disease severity for patients who already have Parkinson’s Disease. "
        "It does not perform initial diagnosis but assists in remote monitoring."
    )