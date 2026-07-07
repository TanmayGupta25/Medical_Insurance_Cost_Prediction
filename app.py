import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    layout="wide"
)

# ---------- Load Artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")

    try:
        preprocessor = joblib.load("models/preprocessor.pkl")
    except Exception:
        preprocessor = None

    try:
        feature_cols = joblib.load("models/feature_columns.pkl")
    except Exception:
        feature_cols = None

    with open("models/metadata.json", "r") as f:
        metadata = json.load(f)

    return model, preprocessor, feature_cols, metadata


try:
    model, preprocessor, feature_cols, metadata = load_artifacts()
except Exception as e:
    st.error("Error loading project artifacts.")
    st.exception(e)
    st.stop()

# ---------- Sidebar ----------
menu = [
    "Home",
    "About Dataset",
    "About Model",
    "Prediction",
    "Developer"
]

choice = st.sidebar.selectbox("Navigation", menu)

# ---------- Home ----------
if choice == "Home":

    st.title("Medical Insurance Cost Prediction")
    st.subheader("Summer School 2026")

    st.markdown("### Developer")
    st.write("Tanmay Gupta")

    st.markdown("---")

    st.write("**Problem Type:**", metadata.get("Problem Type", "Regression"))
    st.write("**Best Model:**", metadata.get("Best Model", "N/A"))

# ---------- Dataset ----------
elif choice == "About Dataset":

    st.title("About Dataset")

    st.write("**Dataset Name:**", metadata.get("Dataset Name", "Insurance Dataset"))
    st.write("**Target Column:**", metadata.get("Target Column", "charges"))
    st.write("**Number of Features:**", 6)

# ---------- Model ----------
elif choice == "About Model":

    st.title("About Model")

    st.write("**Best Model:**", metadata.get("Best Model", "N/A"))

    if "MAE" in metadata:
        st.write(f"**MAE:** {metadata['MAE']:.2f}")

    if "RMSE" in metadata:
        st.write(f"**RMSE:** {metadata['RMSE']:.2f}")

    if "R2 Score" in metadata:
        st.write(f"**R² Score:** {metadata['R2 Score']:.4f}")

# ---------- Prediction ----------
elif choice == "Prediction":

    st.title("Medical Insurance Cost Prediction")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=30
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=25.0
            )

            children = st.number_input(
                "Children",
                min_value=0,
                max_value=10,
                value=0
            )

        with col2:

            sex = st.selectbox(
                "Sex",
                ["male", "female"]
            )

            smoker = st.selectbox(
                "Smoker",
                ["yes", "no"]
            )

            region = st.selectbox(
                "Region",
                [
                    "southwest",
                    "southeast",
                    "northwest",
                    "northeast"
                ]
            )

        submitted = st.form_submit_button("Predict")

    if submitted:

        input_df = pd.DataFrame(
            [[
                age,
                sex,
                bmi,
                children,
                smoker,
                region
            ]],
            columns=[
                "age",
                "sex",
                "bmi",
                "children",
                "smoker",
                "region"
            ]
        )

        try:

            with st.spinner("Predicting..."):

                prediction = model.predict(input_df)[0]

            st.success(
                f"Predicted Medical Insurance Cost: ${prediction:,.2f}"
            )

        except Exception as e:

            st.error("Prediction failed.")
            st.exception(e)

# ---------- Developer ----------
elif choice == "Developer":

    st.title("Developer")

    st.write("**Developer:** Tanmay Gupta")
    st.write("**Programme:** Summer School 2026")
