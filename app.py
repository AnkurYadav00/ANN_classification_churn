import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
import pickle

## load the model
model = tf.keras.models.load_model('model.h5')

## load encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.title("Customer Churn Prediction")

# User Input
CreditScore = st.number_input('Credit Score')
Geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender', label_encoder_gender.classes_)
Age = st.slider('Age', 18, 90)
Tenure = st.slider('Tenure', 0, 10)
Balance = st.number_input('Balance')
NumOfProducts = st.slider('Number Of Products', 1, 4)
HasCrCard = st.selectbox('Has Credit Card', [0, 1])
IsActiveMember = st.selectbox('Is Active Member', [0, 1])
EstimatedSalary = st.number_input('Estimated Salary')

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

## One Hot Encoding Geography
geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Combining Encoded Geography column to input dataframe
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scale the data
input_data_scaled = scaler.transform(input_data)

## Prediction Churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('The Customer is likely to churn.')
else:
    st.write('The Customer is not likely to churn.')
