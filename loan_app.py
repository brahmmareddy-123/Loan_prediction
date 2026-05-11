import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Loan_Prediction.csv")

df = pd.read_csv(csv_path)
# Drop Loan_ID
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical columns
encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Features and target
X = df.drop("Credit_History", axis=1)
y = df["Credit_History"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏦 Credit History Prediction App")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", value=5000)
CoapplicantIncome = st.number_input("Coapplicant Income", value=2000)
LoanAmount = st.number_input("Loan Amount", value=150)
Loan_Amount_Term = st.number_input("Loan Amount Term", value=360)
Property_Area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# Input dataframe
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

# Encode
for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

# Match training columns exactly
input_data = input_data[X.columns]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Good Credit History")
    else:
        st.error("❌ Bad Credit History")
