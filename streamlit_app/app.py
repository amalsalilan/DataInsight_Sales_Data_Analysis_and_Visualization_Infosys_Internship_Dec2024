import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and encoder
import os

# Define the correct file paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "random_forest_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "saved", "encoder.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_encoder():
    return joblib.load(ENCODER_PATH)

best_rf = load_model()
encoder, feature_names = load_encoder()


# Define categorical options for user selection
order_priority_options = ['Critical', 'High', 'Medium', 'Low']
ship_mode_options = ['Same Day', 'First Class', 'Regular Air', 'Delivery Truck', 'Express Air']
region_options = ['West', 'East', 'South', 'Central']

# Streamlit UI
st.title("Shipping Delay Prediction")
st.write("Predict whether an order's shipping delay will exceed the expected time.")

# User Input Fields
order_priority = st.selectbox("Order Priority", order_priority_options)
ship_mode = st.selectbox("Ship Mode", ship_mode_options)
region = st.selectbox("Region", region_options)
shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, step=0.1)
order_quantity = st.number_input("Order Quantity", min_value=1, step=1)
discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, step=0.1)
profit = st.number_input("Profit ($)", step=0.1)


# Process user input
def preprocess_input():
    input_df = pd.DataFrame({
        'Order Priority': [order_priority],
        'Ship Mode': [ship_mode],
        'Region': [region],
        'Shipping Cost': [shipping_cost],
        'Order Quantity': [order_quantity],
        'Discount': [discount],
        'Profit': [profit]
    })

    # Encode categorical variables
    categorical_encoded = encoder.transform(input_df[['Order Priority', 'Ship Mode', 'Region']])
    categorical_df = pd.DataFrame(categorical_encoded, columns=feature_names)

    # Concatenate with numerical features
    numerical_df = input_df[['Shipping Cost', 'Order Quantity', 'Discount', 'Profit']].reset_index(drop=True)
    processed_input = pd.concat([categorical_df, numerical_df], axis=1)

    return processed_input


# Predict on button click
if st.button("Predict Shipping Delay"):
    processed_input = preprocess_input()
    prediction = best_rf.predict(processed_input)
    result = "Delayed" if prediction[0] == 1 else "On Time"

    st.write(f"### Prediction: {result}")




#     erro hande
# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
#
# # Load the trained model and encoder with error handling
# try:
#     best_rf = joblib.load("random_forest_model.pkl")
# except (FileNotFoundError, joblib.externals.loky.process_executor.TerminatedWorkerError) as e:
#     st.error("Error loading model: random_forest_model.pkl not found or corrupted.")
#     st.stop()
#
# try:
#     encoder, feature_names = joblib.load("encoder.pkl")
# except (FileNotFoundError, joblib.externals.loky.process_executor.TerminatedWorkerError) as e:
#     st.error("Error loading encoder: encoder.pkl not found or corrupted.")
#     st.stop()
#
# # Define categorical options for user selection
# order_priority_options = ['Critical', 'High', 'Medium', 'Low']
# ship_mode_options = ['Same Day', 'First Class', 'Regular Air', 'Delivery Truck', 'Express Air']
# region_options = ['West', 'East', 'South', 'Central']
#
# # Streamlit UI
# st.title("Shipping Delay Prediction")
# st.write("Predict whether an order's shipping delay will exceed the expected time.")
#
# # User Input Fields
# order_priority = st.selectbox("Order Priority", order_priority_options)
# ship_mode = st.selectbox("Ship Mode", ship_mode_options)
# region = st.selectbox("Region", region_options)
# shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, step=0.1)
# order_quantity = st.number_input("Order Quantity", min_value=1, step=1)
# discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, step=0.1)
# profit = st.number_input("Profit ($)", step=0.1)
#
#
# # Process user input
# def preprocess_input():
#     input_df = pd.DataFrame({
#         'Order Priority': [order_priority],
#         'Ship Mode': [ship_mode],
#         'Region': [region],
#         'Shipping Cost': [shipping_cost],
#         'Order Quantity': [order_quantity],
#         'Discount': [discount],
#         'Profit': [profit]
#     })
#
#     # Input validation
#     if any(input_df.isnull().sum() > 0):
#         st.error("Missing values detected. Please provide valid inputs.")
#         st.stop()
#
#     if shipping_cost < 0 or order_quantity <= 0 or discount < 0 or discount > 100 or profit < 0:
#         st.error("Invalid values detected. Please enter realistic values.")
#         st.stop()
#
#     # Encode categorical variables
#     try:
#         categorical_encoded = encoder.transform(input_df[['Order Priority', 'Ship Mode', 'Region']])
#         categorical_df = pd.DataFrame(categorical_encoded, columns=feature_names)
#     except Exception as e:
#         st.error(f"Error during encoding: {e}")
#         st.stop()
#
#     # Concatenate with numerical features
#     numerical_df = input_df[['Shipping Cost', 'Order Quantity', 'Discount', 'Profit']].reset_index(drop=True)
#     processed_input = pd.concat([categorical_df, numerical_df], axis=1)
#
#     return processed_input
#
#
# # Predict on button click
# if st.button("Predict Shipping Delay"):
#     try:
#         processed_input = preprocess_input()
#         prediction = best_rf.predict(processed_input)
#         result = "Delayed" if prediction[0] == 1 else "On Time"
#         st.write(f"### Prediction: {result}")
#     except Exception as e:
#         st.error(f"Error during prediction: {e}")
#
