import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from joblib import load
from prophet import Prophet

def main():
    # Set up the page configuration for Streamlit
    st.set_page_config(
        page_title="🚀 Advanced Sales Forecasting",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Stylish Page Title
    st.markdown(
        "<h1 style='text-align: center; color:rgb(112, 255, 69);'>📈 Advanced Sales Forecasting Dashboard</h1>", 
        unsafe_allow_html=True
    )

    # Sidebar Configuration
    st.sidebar.title("Forecast Configuration")

    # Load the Prophet model
    with st.spinner("🔄 Loading model..."):
        model = load("C:\\Users\\DELL\\Desktop\\infosys\\prediction.joblib")

    st.sidebar.success("Model loaded successfully.")

    # Sidebar input for forecast horizon (months)
    input_months = st.sidebar.number_input(
        "📅 Forecast horizon (Monthly basis):",
        min_value=1,
        max_value=24,
        value=4,
        step=1
    )

    # Checkbox to optionally show entire forecast
    show_entire_forecast = st.sidebar.checkbox(
        "📊 Show entire forecast (historical + future)",
        value=False
    )

    # Dropdown (selectbox) to optionally display table data
    display_table_option = st.sidebar.selectbox(
        " Display Forecast Data Table?",
        options=["No", "Yes"],
        index=0
    )

    # Button to trigger the forecast
    if st.sidebar.button(" Generate Forecast"):
        # Create the future dataframe
        future = model.make_future_dataframe(
            periods=input_months,
            freq='M'
        )

        # Generate the forecast
        forecast = model.predict(future)

        # Decide how much of the forecast to display
        if show_entire_forecast:
            df_forecast = forecast.copy()
        else:
            df_forecast = forecast[forecast['ds'] > forecast['ds'].max() - pd.DateOffset(months=input_months)]

        # ----------------------------------------
        # Conditionally display the forecast table
        # ----------------------------------------
        if display_table_option == "Yes":
            st.subheader("📊 Forecasted Data")
            st.dataframe(
                df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].style.format({
                    'yhat': '{:.2f}',
                    'yhat_lower': '{:.2f}',
                    'yhat_upper': '{:.2f}'
                })
            )

        # ----------------------------------------
        # Create an interactive Plotly Figure
        # ----------------------------------------
        fig = go.Figure()

        # Add forecast line (yhat) in **cyan**
        fig.add_trace(go.Scatter(
            x=df_forecast['ds'],
            y=df_forecast['yhat'],
            mode='lines',
            name='📈 Forecast',
            line=dict(color='#00FFFF', width=3)  # **Cyan forecast line**
        ))

        # Add confidence interval fill in **shaded blue**
        fig.add_trace(go.Scatter(
            x=pd.concat([df_forecast['ds'], df_forecast['ds'][::-1]]),
            y=pd.concat([df_forecast['yhat_upper'], df_forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 191, 255, 0.2)',  # **Light blue fill**
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        # Update layout with **new template & dark theme**
        fig.update_layout(
            title_text="📊 Sales Forecast (Monthly Basis)",
            title_font=dict(size=24, color='#FFDD44'),  # **Gold Title**
            xaxis_title="📆 Date",
            yaxis_title="💰 Sales",
            legend=dict(
                bgcolor="rgba(0,0,0,0.2)",  # **Transparent black legend box**
                font=dict(color="white"),
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            template='plotly_dark'  # **Dark theme**
        )

        # Add interactive date selection
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="🔹 1m", step="month", stepmode="backward"),
                    dict(count=3, label="🔹 3m", step="month", stepmode="backward"),
                    dict(count=6, label="🔹 6m", step="month", stepmode="backward"),
                    dict(step="all", label="🔹 All")
                ])
            )
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(" Configure your forecast options in the sidebar and click 'Generate Forecast'.")

# Run the app
if __name__ == "__main__":
    main()
