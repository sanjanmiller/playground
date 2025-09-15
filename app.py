import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
model = pickle.load(open(r'C:\Users\HP\VS CODE\MACHINE LEARNING\CLASSIFIERS\classifier_models.pkl', 'rb'))

logit_model = model['Logit']
knn_model = model['KNN']
# Create a function to make predictions
def predict(model, input_data):
    # Standardizing input data using the same scaler
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    prediction = model.predict(input_data_scaled)
    return prediction

# Frontend UI (Streamlit app layout)
st.title("Model Prediction App")

st.write("""
    This app allows you to use Logistic Regression or KNN model to make predictions based on your input data.
    Select a model, enter the features, and get the predicted outcome.
""")
# User selects a model (Logistic Regression or KNN)
model_choice = st.selectbox("Select Model", ["Logistic Regression", "KNN"])

# User inputs the features
feature1 = st.number_input("Enter Feature 1 (X1)", value=0.0)
feature2 = st.number_input("Enter Feature 2 (X2)", value=0.0)

# User submits the input
submit_button = st.button("Make Prediction")

if submit_button:
    input_data = np.array([[feature1, feature2]])  # Get the features as a 2D array
    
    # Make prediction based on selected model
    if model_choice == "Logistic Regression":
        prediction = predict(logit_model, input_data)
        st.write(f"The Logistic Regression model predicts: {prediction[0]}")
    else:
        prediction = predict(knn_model, input_data)
        st.write(f"The KNN model predicts: {prediction[0]}")

    # Display the prediction result
    st.write(f"Prediction result: {'Class 1' if prediction[0] == 1 else 'Class 0'}")
