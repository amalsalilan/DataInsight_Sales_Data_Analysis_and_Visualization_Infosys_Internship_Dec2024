import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Streamlit app title
st.title("Sales Forecasting with Prophet")

# Define the file path for the model
file_path = r"C:\Users\manu\OneDrive\Documents\spin\prophet_sales_predict.joblib"

# Load the model with caching
@st.cache_resource  # Cache the model to avoid reloading it multiple times
def load_model():
    return load(file_path)

model = load_model()
st.success("Model loaded successfully!")

# Get user input for the number of months to forecast
input_months = st.number_input("Enter the number of months you want to forecast:", min_value=1, max_value=60, step=1, value=12)

# Generate forecast when the user clicks the button
if st.button("Generate Forecast"):
    # Create a future dataframe for forecasting
    future = model.make_future_dataframe(periods=input_months, freq='M')

    # Generate forecasts
    forecast = model.predict(future)

    # Filter the forecasted data for the future period
    future_forecast = forecast[forecast['ds'] > future['ds'].max() - pd.Timedelta(days=30 * input_months)]

    # Display the forecasted data
    st.subheader("Forecasted Data:")
    st.write(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # Plot the forecast
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the forecasted line
    ax.plot(
        future_forecast['ds'],
        future_forecast['yhat'],
        color='blue',
        linewidth=2,
        label='Forecast'
    )

    # Highlight confidence intervals
    ax.fill_between(
        future_forecast['ds'],
        future_forecast['yhat_lower'],
        future_forecast['yhat_upper'],
        color='skyblue',
        alpha=0.3,
        label='Confidence Interval'
    )

    # Add labels, legend, and grid
    ax.set_title(f'Forecast for Next {input_months} Months', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Display the plot in Streamlit
    st.pyplot(fig)