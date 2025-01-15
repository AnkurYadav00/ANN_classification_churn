import streamlit as st
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pandas as pd 
import pickle

## initialize model
model = tf.keras.models.load_model('regression_model.h5')

## Load Encoder and Scalers
with open('label_encoder_gender0.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo0.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler0.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Input data 

Geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])

input_data = pd.DataFrame({
    'CreditScore':[st.number_input('Credit Score', min_value=0)],
    'Gender':[st.selectbox('Gender', label_encoder_gender.classes_)], 
    'Age':[st.slider('Age', 18, 90)], 
    'Tenure': [st.slider('Tenure', 0, 10)], 
    'Balance': [st.number_input('Balance', min_value=0)], 
    'NumOfProducts': [st.slider('Number of Products',1,4)],
    'HasCrCard' : [st.selectbox('Has Credit Card', [0, 1])], 
    'IsActiveMember': [st.selectbox('Is Active Member', [0, 1])], 
    'Exited': [st.selectbox('Exited', [0, 1])], 
 })

 ## Encode Gender and Geography
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

encoded_geo = onehot_encoder_geo.transform([[Geography]]).toarray()
encoded_geo_df = pd.DataFrame(encoded_geo, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_df = pd.concat([pd.DataFrame(input_data).reset_index(drop=True), encoded_geo_df], axis=1)

prediction = model.predict(input_df)
prediction_prob = prediction[0][0]
print(prediction_prob)

st.write(f'Estimated Salary {prediction_prob}')