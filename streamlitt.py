import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('sales_prediction_model.pkl')

# Load dataset for insights
file_path = r"C:\Users\kamal\Desktop\CODE\walmart_Retail_Dataset.xlsx"
df = pd.ExcelFile(file_path).parse('walmart Retail Data')

# Derive Order Month for insights
df['Order Month'] = pd.to_datetime(df['Order Date']).dt.to_period('M')

# Calculate shipping delay
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Shipping Delay'] = (df['Ship Date'] - df['Order Date']).dt.days

# App title
st.title("Walmart Sales Prediction App")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.selectbox("Select Option", ["Predict Sales", "Top-Selling Products", "Shipping Delay Analysis"])

if options == "Predict Sales":
    st.header("Predict Product Sales")
    product_name = st.text_input("Enter Product Name")

    if st.button("Predict"):
        # Filter product
        filtered_df = df[df['Product Name'] == product_name]

        if filtered_df.empty:
            st.error("Product not found in the dataset. Please enter a valid product name.")
        else:
            # Extract necessary features (ensure they match model training features)
            category_code = filtered_df['Product Category'].astype('category').cat.codes.values[0]
            avg_discount = filtered_df['Discount'].mean()
            avg_shipping_delay = filtered_df['Shipping Delay'].mean()
            avg_sales = filtered_df['Sales'].mean()

            # Ensure input matches the model's required feature count
            input_data = np.array([[category_code, avg_discount, avg_shipping_delay, avg_sales]])

            try:
                prediction = model.predict(input_data)
                st.success(f"Predicted Sales: ${prediction[0]:.2f}")
            except ValueError as e:
                st.error(f"Model input error: {e}")

            # Display shipping delay for the entered product
            st.write(f"Average Shipping Delay for {product_name}: {avg_shipping_delay:.2f} days")

elif options == "Top-Selling Products":
    st.header("Top-Selling Products")
    selected_month = st.selectbox("Select Month", df['Order Month'].unique())

    if selected_month:
        top_products = (
            df[df['Order Month'] == selected_month]
            .groupby('Product Name')['Sales']
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        if top_products.empty:
            st.warning("No sales data available for the selected month.")
        else:
            st.write("Top-Selling Products:", top_products)

elif options == "Shipping Delay Analysis":
    st.header("Shipping Delay Analysis")

    avg_delay = df['Shipping Delay'].mean()
    st.write(f"Average Shipping Delay: {avg_delay:.2f} days")

    st.subheader("Shipping Delay by Category")
    delay_by_category = df.groupby('Product Category')['Shipping Delay'].mean().sort_values()
    st.bar_chart(delay_by_category)
