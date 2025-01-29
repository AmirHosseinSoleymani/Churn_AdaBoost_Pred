import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained churn prediction model
model = joblib.load("churn_adaboost.joblib")


# Define a function to preprocess user input
def preprocess_input(data):
    """Convert user inputs into a format compatible with the trained model."""
    
    # Convert binary categorical values to numerical (0 and 1)
    data["gender"] = 1 if data["gender"] == "Male" else 0
    data["SeniorCitizen"] = 1 if data["SeniorCitizen"] == "Yes" else 0
    data["Partner"] = 1 if data["Partner"] == "Yes" else 0
    data["Dependents"] = 1 if data["Dependents"] == "Yes" else 0
    data["PhoneService"] = 1 if data["PhoneService"] == "Yes" else 0
    data["MultipleLines"] = 1 if data["MultipleLines"] == "Yes" else 0
    data["OnlineSecurity"] = 1 if data["OnlineSecurity"] == "Yes" else 0
    data["OnlineBackup"] = 1 if data["OnlineBackup"] == "Yes" else 0
    data["DeviceProtection"] = 1 if data["DeviceProtection"] == "Yes" else 0
    data["TechSupport"] = 1 if data["TechSupport"] == "Yes" else 0
    data["StreamingTV"] = 1 if data["StreamingTV"] == "Yes" else 0
    data["StreamingMovies"] = 1 if data["StreamingMovies"] == "Yes" else 0
    data["PaperlessBilling"] = 1 if data["PaperlessBilling"] == "Yes" else 0

    # Ensure numerical fields are correctly formatted
    data["tenure"] = int(data["tenure"])
    data["MonthlyCharges"] = float(data["MonthlyCharges"])
    data["TotalCharges"] = float(data["TotalCharges"])

    # Define all possible categorical features as one-hot encoded
    categorical_features = {
        "Contract_DSL": 0,
        "Contract_Fiber optic": 0,
        "Contract_No": 0,
        "PaymentMethod_Month-to-month": 0,
        "PaymentMethod_One year": 0,
        "PaymentMethod_Two year": 0,
        "InternetService_Bank transfer (automatic)": 0,
        "InternetService_Credit card (automatic)": 0,
        "InternetService_Electronic check": 0,
        "InternetService_Mailed check": 0,
    }

    # Assign correct value based on user selection
    categorical_features[f"Contract_{data['Contract']}"] = 1
    categorical_features[f"PaymentMethod_{data['PaymentMethod']}"] = 1
    categorical_features[f"InternetService_{data['InternetService']}"] = 1

    # Remove original categorical fields
    del data["Contract"], data["PaymentMethod"], data["InternetService"]

    # Merge numerical and categorical data
    data.update(categorical_features)

    # Ensure the order of columns matches model input
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
        'MonthlyCharges', 'TotalCharges', 'Contract_DSL', 'Contract_Fiber optic',
        'Contract_No', 'PaymentMethod_Month-to-month', 'PaymentMethod_One year',
        'PaymentMethod_Two year', 'InternetService_Bank transfer (automatic)',
        'InternetService_Credit card (automatic)', 'InternetService_Electronic check',
        'InternetService_Mailed check'
    ]

    # Convert to DataFrame with correct order
    df = pd.DataFrame([data], columns=feature_order)

    return df


# Streamlit App
def main():
    st.title("Customer Churn Prediction App")
    st.subheader("Predict if a customer is likely to churn based on their details.")

    # Input fields for user data
    gender = st.radio("Gender", ["Male", "Female"])
    senior_citizen = st.radio("Senior Citizen", ["Yes", "No"])
    partner = st.radio("Partner", ["Yes", "No"])
    dependents = st.radio("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=100, step=1)
    phone_service = st.radio("Phone Service", ["Yes", "No"])
    multiple_lines = st.radio("Multiple Lines", ["Yes", "No"])
    online_security = st.radio("Online Security", ["Yes", "No"])
    online_backup = st.radio("Online Backup", ["Yes", "No"])
    device_protection = st.radio("Device Protection", ["Yes", "No"])
    tech_support = st.radio("Tech Support", ["Yes", "No"])
    streaming_tv = st.radio("Streaming TV", ["Yes", "No"])
    streaming_movies = st.radio("Streaming Movies", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=0.1)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=0.1)

    # Predict button
    if st.button("Predict Churn"):
        # Create a dictionary of user inputs
        user_input = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "InternetService": internet_service,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        # Preprocess user input
        processed_data = preprocess_input(user_input)

        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        # Display result
        if prediction == 1:
            st.error(f"Churn is YES")
            st.error(f"The customer is likely to churn. (Confidence: {probability:.2%})")
        else:
            st.success(f"Churn is NO")
            st.success(f"The customer is not likely to churn. (Confidence: {(1 - probability):.2%})")

if __name__ == "__main__":
    main()
