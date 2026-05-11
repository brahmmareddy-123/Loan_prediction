import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Load Dataset
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Loan_Prediction.csv")

df = pd.read_csv(csv_path)

# Drop Loan_ID if exists
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# -------------------------------
# Handle Missing Values
# -------------------------------
for col in df.columns:
    if df[col].dtype == "object" or str(df[col].dtype) == "string":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# -------------------------------
# Encode Categorical Columns
# -------------------------------
encoders = {}

for col in df.columns:
    if df[col].dtype == "object" or str(df[col].dtype) == "string":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# -------------------------------
# Features & Target
# -------------------------------
X = df.drop("Credit_History", axis=1)
y = df["Credit_History"]

# -------------------------------
# Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Loan Prediction App")
st.title("🏦 Loan Credit History Prediction")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

ApplicantIncome = st.number_input(
    "Applicant Income", min_value=0, value=5000
)

CoapplicantIncome = st.number_input(
    "Coapplicant Income", min_value=0, value=2000
)

LoanAmount = st.number_input(
    "Loan Amount", min_value=0, value=150
)

Loan_Amount_Term = st.number_input(
    "Loan Amount Term", min_value=0, value=360
)

Property_Area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# -------------------------------
# Create Input DataFrame
# -------------------------------
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

# Match training columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Good Credit History")
    else:
        st.error("❌ Bad Credit History")
