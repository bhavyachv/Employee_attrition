import streamlit as st
import pandas as pd
import joblib
model=joblib.load('attrition_model.pkl')

st.title("Employee Attrition Predictor")

#Input
age = st.number_input("Age", 18, 65, 30)
years = st.number_input("Years at Company", 0, 50, 5)
income = st.number_input("Monthly Income", 1000, 20000, 5000)
overtime = st.selectbox("Overtime?", ["No", "Yes"])

if st.button("Predict"):
    ot_val=1 if overtime== "Yes" else 0
    input_data=pd.DataFrame([[age,years,income,ot_val]],
                            columns=['Age','YearsAtCompany','MonthlyIncome','OverTime'])
    prediction=model.predict(input_data)[0]

if prediction==1:
    st.error("High Rish Of Leaving")
else:
    st.success("Low Risk Of Leaving")