
import streamlit as st
import joblib

# Load the saved model and supporting files
model = joblib.load("iris_model.pkl")
flower_names = joblib.load("flower_names.pkl")

# Set up the website
st.title("My First Machine Learning App")
st.write("Enter flower measurements to predict the flower type.")

# Create input boxes
sepal_length = st.number_input(
    "Sepal length (cm)", min_value=0.0, value=5.1
)

sepal_width = st.number_input(
    "Sepal width (cm)", min_value=0.0, value=3.5
)

petal_length = st.number_input(
    "Petal length (cm)", min_value=0.0, value=1.4
)

petal_width = st.number_input(
    "Petal width (cm)", min_value=0.0, value=0.2
)

# Predict when the button is clicked
if st.button("Predict Flower"):
    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted flower: {flower_names[prediction]}"
    )
