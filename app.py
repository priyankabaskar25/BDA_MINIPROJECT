import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('Health_Disease_RandomForestModel.joblib')

# List of symptoms that the model expects (ensure it matches the model's training data)
symptoms_list = [
    "Hypertension", "Diabetes", "Migraine", "Acne", "Cough", "Fever", "Nausea", 
    "Vomiting", "Fatigue", "Dizziness", "Weight loss", "Abdominal pain", "Back pain", 
    "Chest pain", "Joint pain", "Shortness of breath", "Loss of appetite"
]

# Streamlit app title
st.title("Health Disease Prediction")

# Instruction
st.write("Select symptoms to predict the possible disease.")

# Create a multiselect box for symptoms
selected_symptoms = st.multiselect("Symptoms", symptoms_list)

# Prediction button
if st.button("Predict"):
    if selected_symptoms:
        # Convert selected symptoms to a feature vector
        features = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
        prediction = model.predict([features])
        
        # Display the prediction result
        st.write(f"Predicted Disease: **{prediction[0]}**")
    else:
        st.write("Please select at least one symptom to make a prediction.")
