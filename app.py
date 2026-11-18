
import streamlit as st
import pandas as pd
import joblib

FEATURES = ["pclass", "sex", "age", "fare"]

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

st.title("Titanic Survival Prediction")
st.write("An educational machine learning model trained on the Titanic dataset.")

st.subheader("Passenger Details")

pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.radio("Sex", ["Female", "Male"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare Paid", 0, 500, 30)

if st.button("Predict Survival"):
    try:
        model = load_model()

        sex_value = 1 if sex == "Female" else 0

        passenger = pd.DataFrame(
            [[pclass, sex_value, age, fare]],
            columns=FEATURES
        )

        prediction = model.predict(passenger)[0]
        probability = model.predict_proba(passenger)[0][1]

        st.metric(
            "Estimated Survival Probability",
            f"{probability:.1%}"
        )

        if prediction == 1:
            st.success("Prediction: Survived")
        else:
            st.info("Prediction: Did not survive")

    except FileNotFoundError:
        st.error("Model file is missing. Please upload model.joblib.")

st.caption(
    "Educational project using historical Titanic data. "
    "This prediction is not advice and does not represent real-world certainty."
)
