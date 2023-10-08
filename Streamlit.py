#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
import joblib

# Load the trained model from a file
loaded_model = joblib.load(r"C:\Users\PRIYANKA\OneDrive\Desktop\Hack\prediction_ML-Copy1.py")  # Replace 'trained_model.pkl' with your model file path

# Create a Streamlit app
st.title("Disease Predictor")

# Create an input form to collect symptoms
st.sidebar.header("Enter Symptoms")

# Define the list of symptoms
symptoms_list = [
    'itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 'shivering', 'chills',
    'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting', 'vomiting',
    'burning micturition', 'spotting urination', 'fatigue', 'weight gain', 'anxiety', 'cold hands and feets',
    'mood swings', 'weight loss', 'restlessness', 'lethargy', 'patches in throat', 'irregular sugar level',
    'cough', 'high fever', 'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
    'headache', 'yellowish skin', 'dark urine', 'nausea', 'loss of appetite', 'pain behind the eyes',
    'back pain', 'constipation', 'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine', 'yellowing of eyes',
    'acute liver failure', 'fluid overload', 'swelling of stomach', 'swelled lymph nodes', 'malaise',
    'blurred and distorted vision', 'phlegm', 'throat irritation', 'redness of eyes', 'sinus pressure',
    'runny nose', 'congestion', 'chest pain', 'weakness in limbs', 'fast heart rate',
    'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus', 'neck pain',
    'dizziness', 'cramps', 'bruising', 'obesity', 'swollen legs', 'swollen blood vessels', 'puffy face and eyes',
    'enlarged thyroid', 'brittle nails', 'swollen extremities', 'excessive hunger', 'extra marital contacts',
    'drying and tingling lips', 'slurred speech', 'knee pain', 'hip joint pain', 'muscle weakness', 'stiff neck',
    'swelling joints', 'movement stiffness', 'spinning movements', 'loss of balance', 'unsteadiness',
    'weakness of one body side', 'loss of smell', 'bladder discomfort', 'foul smell of urine',
    'continuous feel of urine', 'passage of gases', 'internal itching', 'toxic look (typhos)', 'depression',
    'irritability', 'muscle pain', 'altered sensorium', 'red spots over body', 'belly pain',
    'abnormal menstruation', 'dischromic patches', 'watering from eyes', 'increased appetite', 'polyuria',
    'family history', 'mucoid sputum', 'rusty sputum', 'lack of concentration', 'visual disturbances',
    'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding',
    'distention of abdomen', 'history of alcohol consumption', 'blood in sputum', 'prominent veins on calf',
    'palpitations', 'painful walking', 'pus-filled pimples', 'blackheads', 'scurring', 'skin peeling',
    'silver-like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around the nose',
    'yellow crust ooze', 'prognosis'
]

# Create dropdowns for symptom selection
symptom1 = st.sidebar.selectbox("Symptom 1", symptoms_list)
symptom2 = st.sidebar.selectbox("Symptom 2", symptoms_list)
# Add more symptoms as needed

# Create a button to trigger predictions
if st.sidebar.button("Predict"):
    # Perform data preprocessing based on selected symptoms
    input_data = [symptom1, symptom2]  # Add more symptoms here
    # Perform data preprocessing and invoke the model
    # Replace this with your preprocessing code and model prediction code
    predicted_disease = loaded_model.predict([input_data])[0]

    # Display the results
    st.header("Prediction Results")
    st.write("Predicted Disease:", predicted_disease)
    # Add code to display disease description and recommendations here


# In[ ]:




