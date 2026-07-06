import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load artifacts
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    with open("models/metadata.json", "r") as f:
        metadata = json.load(f)
    return model, preprocessor, feature_cols, metadata

model, preprocessor, feature_cols, metadata = load_artifacts()

st.set_page_config(page_title="Insurance Prediction", layout="wide")

menu = ["Home", "About Dataset", "About Model", "Prediction", "Developer"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.title("Medical Insurance Cost Prediction")
    st.subheader("Summer School 2026")
    st.markdown(f"**Author:** Tanmay Gupta")
    st.markdown("--- ")
    st.write(f"**Problem Type:** {metadata['Problem Type']}")
    st.write(f"**Best Model:** {metadata['Best Model']}")

elif choice == "About Dataset":
    st.title("About Dataset")
    st.write(f"**Dataset Name:** {metadata['Dataset Name']}")
    st.write(f"**Target Column:** {metadata['Target Column']}")
    st.write("**Number of Features:** 6")

elif choice == "About Model":
    st.title("About Model")
    st.write(f"**Best Model:** {metadata['Best Model']}")
    st.write(f"**MAE:** {metadata['MAE']:.2f}")
    st.write(f"**RMSE:** {metadata['RMSE']:.2f}")
    st.write(f"**R² Score:** {metadata['R2 Score']:.4f}")

elif choice == "Prediction":
    st.title("Predict Insurance Cost")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
        with col2:
            sex = st.selectbox("Sex", ["male", "female"])
            smoker = st.selectbox("Smoker", ["yes", "no"])
            region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
        
        submit = st.form_submit_button("Predict")
        
    if submit:
        try:
            with st.spinner("Calculating..."):
                input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                                       columns=["age", "sex", "bmi", "children", "smoker", "region"])
                prediction = model.predict(input_df)[0]
                st.success(f"### Predicted Medical Insurance Cost: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif choice == "Developer":
    st.title("Developer Information")
    st.write("**Developer:** Tanmay Gupta")
    st.write("**Programme:** Summer School 2026")
