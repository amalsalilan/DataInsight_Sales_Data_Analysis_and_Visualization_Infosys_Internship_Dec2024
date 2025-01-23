import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('sales_prediction_model.pkl')

# Load dataset for insights
file_path = r"C:\Users\kamal\Downloads\walmart Retail Dataset.xlsx"
df = pd.ExcelFile(file_path).parse('walmart Retail Data')

# Derive Order Month for insights
df['Order Month'] = pd.to_datetime(df['Order Date']).dt.to_period('M')

# App title
st.title("Walmart Sales Prediction App")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.selectbox("Select Option", ["Predict Sales", "Top-Selling Products", "Shipping Delay Analysis"])

if options == "Predict Sales":
    st.header("Predict Product Sales")
    product_name = st.text_input("Enter Product Name")
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100)
    product_margin = st.number_input("Product Base Margin", min_value=0.0, step=0.01)

    if st.button("Predict"):
        # Filter product
        filtered_df = df[df['Product Name'] == product_name]

        if filtered_df.empty:
            st.error("Product not found in the dataset. Please enter a valid product name.")
        else:
            # Convert to category codes
            category_code = filtered_df['Product Category'].astype('category').cat.codes.values[0]
            input_data = [[category_code, customer_age, product_margin, 0]]  # 0 as placeholder for sales
            prediction = model.predict(input_data)
            st.success(f"Predicted Sales: ${prediction[0]:.2f}")

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
    # Add shipping delay insights and logic
    st.write("Feature under development!")
