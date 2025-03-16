import streamlit as st
import pandas as pd
import joblib


st.title('Car insurance prediction')

# Read dataset to fill values in dropdown list
df = pd.read_csv('train.csv')

# Create the input file       
id = st.selectbox("id", pd.unique(df['id']))
Gender = st.selectbox("Gender", pd.unique(df['Gender']))
Age = st.selectbox("Age", pd.unique(df['Age']))
Driving_License = st.selectbox("Driving_License", pd.unique(df['Driving_License']))
Region_Code = st.selectbox("Region_Code", pd.unique(df['Region_Code']))
Previously_Insured = st.selectbox("Previously_Insured", pd.unique(df['Previously_Insured']))
Vehicle_Damage = st.selectbox("Vehicle_Damage", pd.unique(df['Vehicle_Damage']))
Annual_Premium = st.selectbox("Annual_Premium", pd.unique(df['Annual_Premium']))
Policy_Sales_Channel = st.selectbox("Policy_Sales_Channel", pd.unique(df['Policy_Sales_Channel']))
Vintage = st.selectbox("Vintage", pd.unique(df['Vintage']))
Vehicle_Age = st.selectbox("Vehicle_Age", pd.unique(df['Vehicle_Age']))

# Convert the input values into a dictionary
input = {
    "id": id,
    "Gender": Gender,
    "Age": Age,
    "Driving_License": Driving_License,
    "Region_Code": Region_Code,
    "Previously_Insured": Previously_Insured,
    "Vehicle_Damage": Vehicle_Damage,
    "Annual_Premium": Annual_Premium,
    "Policy_Sales_Channel": Policy_Sales_Channel,
    "Vintage": Vintage,
    "Vehicle_Age": Vehicle_Age
}


# Click for prediction (Prediction button)


if st.button("Predict"):
    model = joblib.load("cv_random_forest_model.pkl")
    # Input to the fields
    X_input = pd.DataFrame(input, index=[0])
    # Modle prediction
    prediction = model.predict(X_input)
    # Display the prediction
    st.write(prediction)

# streamlit run webapp.py

