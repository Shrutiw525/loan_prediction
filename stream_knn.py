import streamlit as st
import pandas as pd

# =========================
# Load dataset
# =========================
user = pd.read_csv("task1_dataset.csv")

# Drop unnecessary column
user.drop("date", axis=1, inplace=True)

st.write("Original Dataset:")
st.write(user.head())

# =========================
# Preprocessing (TRAINING DATA)
# =========================

# Fill missing values
numerical_cols = ["income","loan_amount","credit_score","annual_spend","num_transactions"]

for col in numerical_cols:
    user[col].fillna(user[col].median(), inplace=True)

# Outlier handling
def handle_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)

for col in numerical_cols:
    handle_outliers(user, col)

# =========================
# Encoding
# =========================
from sklearn.preprocessing import OneHotEncoder

categorical_cols = ["city", "employment_type", "loan_type"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(user[categorical_cols])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(categorical_cols)
)

user = pd.concat([user.drop(categorical_cols, axis=1), encoded_df], axis=1)

# =========================
# Scaling
# =========================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
user[numerical_cols] = scaler.fit_transform(user[numerical_cols])

st.write("Preprocessed Dataset:")
st.write(user.head())

# =========================
# Train Model
# =========================
x = user.drop("target", axis=1)
y = user["target"]

from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

# =========================
# USER INPUT
# =========================
st.header("Enter User Details")

age = st.number_input("age", min_value=18, max_value=100, value=30)
income = st.number_input("income", min_value=0, value=50000)
loan_amount = st.number_input("loan_amount", min_value=0, value=10000)
credit_score = st.number_input("credit_score", min_value=300, max_value=850, value=700)
annual_spend = st.number_input("annual_spend", min_value=0, value=20000)
num_transactions = st.number_input("num_transactions", min_value=0, value=50)

# IMPORTANT: use original dataset for dropdown options
city = st.selectbox("city", options=pd.read_csv("task1_dataset.csv")["city"].unique())
employment_type = st.selectbox("employment_type", options=pd.read_csv("task1_dataset.csv")["employment_type"].unique())
loan_type = st.selectbox("loan_type", options=pd.read_csv("task1_dataset.csv")["loan_type"].unique())

# =========================
# Convert input to DataFrame
# =========================
df = pd.DataFrame({
    "age": [age],
    "income": [income],
    "loan_amount": [loan_amount],
    "credit_score": [credit_score],
    "annual_spend": [annual_spend],
    "num_transactions": [num_transactions],
    "city": [city],
    "employment_type": [employment_type],
    "loan_type": [loan_type]
})

# =========================
# Apply SAME preprocessing to input
# =========================

# Fill missing
for col in numerical_cols:
    df[col].fillna(user[col].median(), inplace=True)

# Outlier handling
for col in numerical_cols:
    handle_outliers(df, col)

# Encoding (use SAME encoder)
encoded_input = encoder.transform(df[categorical_cols])

encoded_input_df = pd.DataFrame(
    encoded_input,
    columns=encoder.get_feature_names_out(categorical_cols)
)

df = pd.concat([df.drop(categorical_cols, axis=1), encoded_input_df], axis=1)

# Scaling (use SAME scaler)
df[numerical_cols] = scaler.transform(df[numerical_cols])

# Align columns with training data
df = df.reindex(columns=x_train.columns, fill_value=0)

# =========================
# Prediction
# =========================
if st.button("Predict"):
    prediction = lr.predict(df)
    st.success(f"Prediction: {prediction[0]:.2f}")