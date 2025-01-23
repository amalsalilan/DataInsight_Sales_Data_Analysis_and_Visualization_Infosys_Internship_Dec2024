import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Walmart Retail Data - Sales Forecasting")

# File upload section
st.sidebar.title("Upload your Walmart dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Show data preview
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Data Preprocessing
    st.write("### Data Preprocessing")
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
    data.dropna(subset=['Order Date', 'Sales'], inplace=True)

    # Group data by Order Date and sum up sales
    sales_data = data.groupby('Order Date')['Sales'].sum().reset_index()

    # Plot historical sales
    st.write("### Historical Sales Data")
    plt.figure(figsize=(10, 6))
    plt.plot(sales_data['Order Date'], sales_data['Sales'], label='Sales')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales Over Time")
    plt.legend()
    st.pyplot(plt)

    # Forecasting section
    st.write("### Sales Forecasting")
    sales_data.set_index('Order Date', inplace=True)
    sales_data = sales_data.resample('M').sum()  # Resample monthly

    # Split data into training and testing
    train, test = train_test_split(sales_data, test_size=0.2, shuffle=False)

    # Train the forecasting model
    model = ExponentialSmoothing(train['Sales'], seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(len(test))

    # Display Forecast vs Actual
    st.write("#### Forecast vs Actual")
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train['Sales'], label='Training Data')
    plt.plot(test.index, test['Sales'], label='Actual Sales')
    plt.plot(test.index, forecast, label='Forecasted Sales')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    st.pyplot(plt)

    # Allow user to forecast future sales
    st.write("#### Predict Future Sales")
    periods = st.slider("Select number of months to forecast", 1, 12, 3)
    future_forecast = model.forecast(periods)
    st.write(f"Forecasted Sales for the next {periods} months:")
    st.write(future_forecast)

else:
    st.write("Please upload a dataset to get started.")
