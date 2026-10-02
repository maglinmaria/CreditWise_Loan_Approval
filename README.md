# 💳 CreditWise

CreditWise is an AI-powered loan approval prediction application built using Machine Learning and Streamlit.

## Features

- Loan approval prediction
- Logistic Regression model
- Numerical feature preprocessing
- Categorical feature encoding
- Missing-value handling
- Feature engineering
- Approval probability
- Interactive Streamlit interface

## Machine Learning

The application uses Logistic Regression for binary loan approval classification.

### Feature Engineering

The model uses:

- DTI Ratio squared
- Credit Score squared

### Preprocessing

Numerical features are:

- Missing-value imputed
- Standardized using StandardScaler

Categorical features are:

- Missing-value imputed
- One-hot encoded

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt