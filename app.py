import streamlit as st
import pickle
import numpy as np


# ============================================
# Load Trained Model
# ============================================
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)


# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Wale's Loan Approval Predictor",
    page_icon="💵",
    layout="centered"
)


st.title("💵 Wale's Loan Approval Predictor")
st.write(
    "Fill in the applicant's details below to predict whether the loan "
    "application is likely to be approved."
)


st.markdown("---")


# ============================================
# Applicant Information
# ============================================


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


married = st.selectbox(
    "Marital Status",
    ["Yes", "No"]
)


dependents = st.selectbox(
    "Number of Dependents",
    ["0", "1", "2", "3+"]
)


education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)


self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)


credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0],
    help="1 = Good Credit History, 0 = Bad Credit History"
)


property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)


st.markdown("### Financial Information")


applicant_income = st.number_input(
    "Applicant Income",
    min_value=1.0,
    value=5000.0
)


coapplicant_income = st.number_input(
    "Co-applicant Income",
    min_value=0.0,
    value=0.0
)


loan_amount = st.number_input(
    "Loan Amount",
    min_value=1.0,
    value=120.0
)


loan_term = st.number_input(
    "Loan Amount Term (Months)",
    min_value=1.0,
    value=360.0
)


# ============================================
# Encoding Dictionaries
# ============================================


gender_map = {
    "Female": 0,
    "Male": 1
}


married_map = {
    "No": 0,
    "Yes": 1
}


education_map = {
    "Graduate": 0,
    "Not Graduate": 1
}


self_employed_map = {
    "No": 0,
    "Yes": 1
}


dependents_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}


property_area_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}


# ============================================
# Prediction
# ============================================


if st.button("Predict Loan Status", use_container_width=True):


    # Derived Features
    total_income = applicant_income + coapplicant_income


    # IMPORTANT:
    # Change np.log to np.log1p if your notebook used np.log1p()
    applicant_income_log = np.log(applicant_income)
    loan_amount_log = np.log(loan_amount)
    loan_amount_term_log = np.log(loan_term)
    total_income_log = np.log(total_income)


    input_data = np.array([[
        gender_map[gender],
        married_map[married],
        dependents_map[dependents],
        education_map[education],
        self_employed_map[self_employed],
        credit_history,
        property_area_map[property_area],
        applicant_income_log,
        loan_amount_log,
        loan_amount_term_log,
        total_income_log
    ]])


    prediction = model.predict(input_data)[0]


    try:
        probability = model.predict_proba(input_data)[0]
        confidence = max(probability) * 100
    except Exception:
        confidence = None


    st.markdown("---")


    if prediction == 1:
        st.success("✅ Congratulations! The loan is likely to be Approved.")
        st.balloons()
    else:
        st.error("❌ LOL!!! Come back Later")


    # if confidence is not None:
    #     st.info(f"Prediction Confidence: **{confidence:.2f}%**")


# ============================================
# Footer
# ============================================


st.markdown("---")
st.success("Developed by Wale Adedeji")
