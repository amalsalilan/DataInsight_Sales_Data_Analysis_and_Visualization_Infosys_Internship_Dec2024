import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from joblib import load

# Streamlit App Title
st.title("📈 Sales Forecasting with Prophet")

# Sidebar for User Inputs
st.sidebar.header("Forecast Configuration")
input_months = st.sidebar.number_input(
    "Months to forecast:",
    min_value=1,
    max_value=60,
    value=12,
    step=1,
    help="Select the number of months for the forecast."
)

# Load Prophet Model
st.sidebar.header("Model Configuration")
try:
    model = load("Prophet_model.joblib")  # Update the file path if necessary
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# Button to Trigger Forecasting
if st.sidebar.button("Generate Forecast"):
    st.header(f"Forecast for Next {input_months} Months")

    # Generate a future dataframe for the forecast
    future = model.make_future_dataframe(periods=input_months, freq='ME')

    # Generate predictions
    forecast = model.predict(future)

    # Filter forecast for future data only
    last_training_date = future['ds'].max() - pd.Timedelta(days=30 * input_months)
    future_forecast = forecast[forecast['ds'] > last_training_date]

    # Display Forecasted Data Table
    st.subheader("📋 Forecasted Data")
    st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2))

    # Create a Matplotlib Plot
    st.subheader("📊 Forecast Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        future_forecast['ds'],
        future_forecast['yhat'],
        color='blue',
        label='Forecast',
        linewidth=2
    )
    ax.fill_between(
        future_forecast['ds'],
        future_forecast['yhat_lower'],
        future_forecast['yhat_upper'],
        color='skyblue',
        alpha=0.3,
        label='Confidence Interval'
    )
    ax.set_title(f"Sales Forecast for Next {input_months} Months", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Sales", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Show the Plot in Streamlit
    st.pyplot(fig)

    # Download Forecasted Data
    st.subheader("📥 Download Forecasted Data")
    csv_data = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name=f"sales_forecast_{input_months}_months.csv",
        mime='text/csv'
    )
