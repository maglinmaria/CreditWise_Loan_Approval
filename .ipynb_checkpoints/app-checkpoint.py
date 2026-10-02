import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD YOUR TRAINED FILES
# ============================================================

model = joblib.load("creditwise_model.pkl")
scaler = joblib.load("creditwise_scaler.pkl")
ohe = joblib.load("creditwise_ohe.pkl")
education_encoder = joblib.load("education_encoder.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CreditWise",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    font-size: 48px;
    font-weight: 800;
}

.subtitle {
    font-size: 20px;
    color: #aaaaaa;
}

.section {
    font-size: 28px;
    font-weight: 700;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">💳 CreditWise</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the applicant's information to estimate "
    "the likelihood of loan approval."
)


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.markdown(
    '<div class="section">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

with col2:

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

with col3:

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )


col1, col2 = st.columns(2)

with col1:

    education = st.selectbox(
        "Education Level",
        [
            "Graduate",
            "Not Graduate"
        ]
    )

with col2:

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married",
            "Single"
        ]
    )


# ============================================================
# EMPLOYMENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section">💼 Employment Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    employment_status = st.selectbox(
        "Employment Status",
        [
            "Contract",
            "Salaried",
            "Self-employed",
            "Unemployed"
        ]
    )

with col2:

    employer_category = st.selectbox(
        "Employer Category",
        [
            "Business",
            "Government",
            "MNC",
            "Private",
            "Unemployed"
        ]
    )


# ============================================================
# LOAN INFORMATION
# ============================================================

st.markdown(
    '<div class="section">🏦 Loan Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Business",
            "Car",
            "Education",
            "Home",
            "Personal"
        ]
    )

with col2:

    property_area = st.selectbox(
        "Property Area",
        [
            "Rural",
            "Semiurban",
            "Urban"
        ]
    )

with col3:

    loan_term = st.number_input(
        "Loan Term",
        min_value=1,
        max_value=480,
        value=48,
        step=1
    )


col1, col2 = st.columns(2)

with col1:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )

with col2:

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

st.markdown(
    '<div class="section">💰 Financial Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=10548.0,
        step=500.0
    )

with col2:

    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

with col3:

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=9940.0,
        step=500.0
    )


col1, col2, col3 = st.columns(3)

with col1:

    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=24802.0,
        step=500.0
    )

with col2:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=678.0,
        step=1.0
    )

with col3:

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.34,
        step=0.01
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Loan Approval",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # 1. EDUCATION ENCODING
        # ----------------------------------------------------

        # Your training used LabelEncoder.
        # LabelEncoder alphabetically maps:
        # Graduate -> 0
        # Not Graduate -> 1

        if education == "Graduate":
            education_encoded = 0
        else:
            education_encoded = 1


        # ----------------------------------------------------
        # 2. FEATURE ENGINEERING
        # ----------------------------------------------------

        dti_ratio_sq = dti_ratio ** 2

        credit_score_sq = credit_score ** 2


        # ----------------------------------------------------
        # 3. CREATE CATEGORICAL DATAFRAME
        # ----------------------------------------------------

        categorical_data = pd.DataFrame({

            "Employment_Status": [
                employment_status
            ],

            "Marital_Status": [
                marital_status
            ],

            "Loan_Purpose": [
                loan_purpose
            ],

            "Property_Area": [
                property_area
            ],

            "Gender": [
                gender
            ],

            "Employer_Category": [
                employer_category
            ]
        })


        # ----------------------------------------------------
        # 4. APPLY YOUR SAVED ONE-HOT ENCODER
        # ----------------------------------------------------

        encoded_data = ohe.transform(
            categorical_data
        )


        encoded_columns = ohe.get_feature_names_out()

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoded_columns
        )


        # ----------------------------------------------------
        # 5. CREATE NUMERICAL DATA
        # ----------------------------------------------------

        numerical_data = pd.DataFrame({

            "Applicant_Income": [
                applicant_income
            ],

            "Coapplicant_Income": [
                coapplicant_income
            ],

            "Age": [
                age
            ],

            "Dependents": [
                dependents
            ],

            "Existing_Loans": [
                existing_loans
            ],

            "Savings": [
                savings
            ],

            "Collateral_Value": [
                collateral_value
            ],

            "Loan_Amount": [
                loan_amount
            ],

            "Loan_Term": [
                loan_term
            ],

            "Education_Level": [
                education_encoded
            ],

            "DTI_Ratio_sq": [
                dti_ratio_sq
            ],

            "Credit_Score_sq": [
                credit_score_sq
            ]
        })


        # ----------------------------------------------------
        # 6. COMBINE NUMERICAL + ENCODED DATA
        # ----------------------------------------------------

        final_data = pd.concat(
            [
                numerical_data,
                encoded_df
            ],
            axis=1
        )


        # ----------------------------------------------------
        # 7. FORCE EXACT TRAINING COLUMN ORDER
        # ----------------------------------------------------

        expected_columns = list(
            scaler.feature_names_in_
        )

        final_data = final_data.reindex(
            columns=expected_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # 8. SCALE USING YOUR TRAINED SCALER
        # ----------------------------------------------------

        scaled_data = scaler.transform(
            final_data
        )


        # ----------------------------------------------------
        # 9. PREDICT USING YOUR TRAINED MODEL
        # ----------------------------------------------------

        prediction = model.predict(
            scaled_data
        )[0]


        probability = model.predict_proba(
            scaled_data
        )[0][1]


        # ----------------------------------------------------
        # 10. DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.header("📊 Prediction Result")


        if prediction == 1:

            st.success(
                "✅ LOAN LIKELY TO BE APPROVED"
            )

        else:

            st.error(
                "❌ LOAN LIKELY TO BE REJECTED"
            )


        st.metric(
            "Approval Probability",
            f"{probability * 100:.2f}%"
        )


        st.progress(
            float(probability)
        )


        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        with st.expander("📋 View Application Summary"):

            st.write(
                f"**Age:** {age}"
            )

            st.write(
                f"**Education:** {education}"
            )

            st.write(
                f"**Employment:** {employment_status}"
            )

            st.write(
                f"**Loan Amount:** ₹{loan_amount:,.2f}"
            )

            st.write(
                f"**Credit Score:** {credit_score}"
            )

            st.write(
                f"**DTI Ratio:** {dti_ratio:.2f}"
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)