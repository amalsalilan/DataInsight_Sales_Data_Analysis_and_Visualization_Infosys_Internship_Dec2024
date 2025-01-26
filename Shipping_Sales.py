import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from joblib import load
from prophet import Prophet

# Streamlit App Title
st.title("📦 Shipping Delay Prediction & 📈 Sales Forecasting")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Shipping Delay Prediction", "Sales Forecasting"])

if page == "Shipping Delay Prediction":
    # Shipping Delay Prediction
    st.header("Shipping Delay Prediction")

    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_excel("walmart Retail Data.xlsx")

    def preprocess_data(df):
        df['Order Priority'] = df['Order Priority'].replace('Not Specified', np.nan)

        imputer = SimpleImputer(strategy='most_frequent')
        df['Order Priority'] = imputer.fit_transform(df[['Order Priority']]).ravel()
        # Feature Engineering: Calculate Shipping Delay (days)
        df['Shipping Delay'] = (df['Ship Date'] - df['Order Date']).dt.days

        # Define delay threshold based on Ship Mode
        delay_thresholds = {
            "Same Day": 0,
            "First Class": 1,
            "Regular Air": 3,
            "Delivery Truck": 5,
            "Express Air": 2
        }

        df['Expected Delay'] = df['Ship Mode'].map(delay_thresholds)
        df['Target_Classification'] = (df['Shipping Delay'] > df['Expected Delay']).astype(int)

        # Drop rows with missing target values
        df = df.dropna(subset=['Target_Classification'])

        return df

    def build_model(df):
        # Define features and target
        features = ['Order Priority', 'Ship Mode', 'Region', 'Shipping Cost', 'Order Quantity', 'Discount', 'Profit']
        X = df[features]
        y = df['Target_Classification']

        # Preprocessing Pipeline
        categorical_features = ['Order Priority', 'Ship Mode', 'Region']
        numerical_features = ['Shipping Cost', 'Order Quantity', 'Discount', 'Profit']

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numerical_features)
            ]
        )

        # Transform features
        X_transformed = preprocessor.fit_transform(X)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

        # Train Classification Model using XGBoost
        clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate Model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return clf, preprocessor, categorical_features, numerical_features, accuracy, report

    def predict_delay_status(clf, preprocessor, categorical_features, numerical_features, data):
        """
        Predict whether a shipment will be delayed or on-time.

        Args:
            clf: Trained model.
            preprocessor: Preprocessing pipeline.
            categorical_features (list): List of categorical feature names.
            numerical_features (list): List of numerical feature names.
            data (dict): Input data as a dictionary containing feature values.

        Returns:
            str: "Delayed" if the shipment is predicted to be delayed, otherwise "On-Time".
        """
        df_input = pd.DataFrame([data])

        # Preprocess input data
        X_encoded = preprocessor.transform(df_input)

        # Make prediction
        prediction = clf.predict(X_encoded)
        return "Delayed" if prediction[0] == 1 else "On-Time"

    df = load_data()
    df = preprocess_data(df)
    clf, preprocessor, categorical_features, numerical_features, accuracy, report = build_model(df)

    st.sidebar.header("Input Features")

    # User input
    input_data = {
        'Order Priority': st.sidebar.selectbox("Order Priority", df['Order Priority'].unique()),
        'Ship Mode': st.sidebar.selectbox("Ship Mode", df['Ship Mode'].unique()),
        'Region': st.sidebar.selectbox("Region", df['Region'].unique()),
        'Shipping Cost': st.sidebar.number_input("Shipping Cost", min_value=0.0, step=0.1),
        'Order Quantity': st.sidebar.number_input("Order Quantity", min_value=1, step=1),
        'Discount': st.sidebar.slider("Discount", min_value=0.0, max_value=1.0, step=0.01),
        'Profit': st.sidebar.number_input("Profit", min_value=-1000.0, step=1.0)
    }

    if st.sidebar.button("Predict Shipping Status"):
        prediction = predict_delay_status(clf, preprocessor, categorical_features, numerical_features, input_data)
        st.subheader("Prediction")
        st.write(f"Predicted Shipping Status: {prediction}")

elif page == "Sales Forecasting":
    # Sales Forecasting
    st.header("Sales Forecasting")

    st.sidebar.header("Forecast Configuration")
    input_months = st.sidebar.number_input(
        "Months to forecast:",
        min_value=1,
        max_value=60,
        value=12,
        step=1,
        help="Select the number of months for the forecast."
    )

    st.sidebar.header("Model Configuration")
    try:
        model = load("Prophet_model.joblib")  # Update the file path if necessary
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.stop()

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
