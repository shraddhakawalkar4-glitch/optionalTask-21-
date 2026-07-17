# Question 01
# Streamlit is used to build interactive web apps directly from Python scripts
import streamlit as st       

# Pandas is used for data manipulation and analysis, especially with tabular datasets
import pandas as pd

# Joblib is used to save and load machine learning models efficiently
import joblib
from sklearn.preprocessing import StandardScaler

# Q2. (Loading Model and Preprocessing Objects)
model = joblib.load('./insurence_model1.pkl')          
scaler = joblib.load('./scaler1.pkl')          
encoded_columns = joblib.load('./columns1.pkl')

# Q3. (Page Configuration)
st.set_page_config(
    page_title = "Medical Insurance Cost Predictor",
    layout = "centered", 
)


# Q4. (Title and Description)
st.title("Medical Insurance Cost Predictor")
st.write("Enter patient details below to predict medical insurance cost.")

# Q5. (Numerical Input Fields)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

annual_income_usd = st.number_input(
    "Annual Income (USD)",
    min_value=0,
    max_value=1000000,
    value=50000,
    step=1000
)

exercise_level = st.number_input(
    "Exercise Level (hours per week)",
    min_value=0,
    max_value=40,
    value=3
)

chronic_diseases = st.number_input(
    "Number of Chronic Diseases",
    min_value=0,
    max_value=10,
    value=0
)

doctor_visits_per_year = st.number_input(
    "Doctor Visits per Year",
    min_value=0,
    max_value=50,
    value=2
)

hospitalizations_last_year = st.number_input(
    "Hospitalizations Last Year",
    min_value=0,
    max_value=10,
    value=0
)

alcohol_consumption_per_week = st.number_input(
    "Alcohol Consumption (drinks per week)",
    min_value=0,
    max_value=50,
    value=0
)
insurance_plan=st.number_input(
    "Insurance plan",
    min_value=0,
    max_value=50,
    value=0
)

# Q6. (Categorical Input using Dropdowns)
gender = st.selectbox(
    "gender",
    ["male", "female"]
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Q7. (Predict Button)
if st.button("Predict Insurance Cost"):

    # Q8. (Creating Input DataFrame & Encoding)
   # Create input dataframe
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "children": [children],
        "annual_income_usd": [annual_income_usd],
        "exercise_level": [exercise_level],
        "chronic_diseases": [chronic_diseases],
        "doctor_visits_per_year": [doctor_visits_per_year],
        "hospitalizations_last_year": [hospitalizations_last_year],
        "alcohol_consumption_per_week": [alcohol_consumption_per_week],
        "insurance_plan": [insurance_plan],
        "smoker": [smoker],
        "region": [region]
    })

# One-hot encoding
    encoded_input_df = pd.get_dummies(input_df)

    # Match training columns
    encoded_input_df = encoded_input_df.reindex(columns=encoded_columns, fill_value=0)

    # Predict
    prediction = model.predict(encoded_input_df)

    st.success(f"Predicted Insurance Cost: ₹{prediction[0]:,.2f}")

