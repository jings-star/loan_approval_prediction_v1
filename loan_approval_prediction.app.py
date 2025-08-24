import streamlit as st
import pandas as pd
import joblib

# Load trained RandomForest model
model = joblib.load("loan_approval_rfmodel.joblib")

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("📊 Loan Approval Prediction Prototype")
st.write("Fill in the applicant details to predict loan approval.")

# --- Applicant Input Form ---
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=120, value=30, step=1)
        person_income = st.number_input("Annual Income ($)", min_value=1000.00, max_value=500000.00, 
                                        value=50000.00, step=1000.00, format="%.2f")
        person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5, step=1)
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10, step=1)

        person_education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        person_gender = st.selectbox("Gender", ["male", "female"])
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    with col2:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500.00, max_value=100000.00, 
                                    value=10000.00, step=500.00, format="%.2f")
        loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=40.0, 
                                        value=12.5, step=0.1, format="%.2f")
        loan_percent_income = st.number_input("Loan Percent Income (loan_amnt/person_income)", 
                                              min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.2f")
        loan_intent = st.selectbox("Loan Intent", 
                                   ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        previous_loan_defaults_on_file = st.selectbox("Previous Loan Default on File", ["no", "yes"])

    # Submit button
    submitted = st.form_submit_button("🔮 Predict Loan Approval")

# --- Collect applicant data ---
if submitted:
    applicant = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "person_home_ownership": person_home_ownership,
        "person_education": person_education,
        "person_gender": person_gender,
        "loan_intent": loan_intent,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
    }

    input_df = pd.DataFrame([applicant])

    # Ensure column order matches training model
    if hasattr(model, "feature_names_in_"):
        input_df = input_df.reindex(columns=model.feature_names_in_)

    # --- Prediction ---
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Loan Approved! (Approval Probability: {probability:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Approval Probability: {probability:.2f})")

    with st.expander("📑 See Applicant Data"):
        st.write(input_df)
