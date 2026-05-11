import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -----------------------
# Load dataset
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Loan_Prediction.csv")

df = pd.read_csv(csv_path)

# Drop Loan_ID
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# -----------------------
# Fill missing values
# -----------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# -----------------------
# Encode categorical
# -----------------------
encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Remove any infinity values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# -----------------------
# Features / Target
# -----------------------
X = df.drop("Credit_History", axis=1)
y = df["Credit_History"]

# -----------------------
# Train model
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# -----------------------
# Streamlit UI
# -----------------------
st.title("🏦 Loan Credit Prediction")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

ApplicantIncome = st.number_input("Applicant Income", 0, 100000, 5000)
CoapplicantIncome = st.number_input("Coapplicant Income", 0, 100000, 2000)
LoanAmount = st.number_input("Loan Amount", 0, 1000, 150)
Loan_Amount_Term = st.number_input("Loan Amount Term", 0, 500, 360)

Property_Area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# -----------------------
# Input dataframe
# -----------------------
input_data = pd.DataFrame({
    "Gender": [Gender],
    "Married": [Married],
    "Dependents": [Dependents],
    "Education": [Education],
    "Self_Employed": [Self_Employed],
    "ApplicantIncome": [ApplicantIncome],
    "CoapplicantIncome": [CoapplicantIncome],
    "LoanAmount": [LoanAmount],
    "Loan_Amount_Term": [Loan_Amount_Term],
    "Property_Area": [Property_Area]
})

# Encode input
for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

input_data = input_data.reindex(columns=X.columns, fill_value=0)

# -----------------------
# Predict
# -----------------------
if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Good Credit History")
    else:
        st.error("❌ Bad Credit History")
