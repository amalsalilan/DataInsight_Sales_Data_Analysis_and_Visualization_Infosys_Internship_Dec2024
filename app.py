import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from prophet import Prophet
import plotly.express as px  # Optional if you want to use Plotly



# -- Plotly Interactive Chart --
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
from prophet import Prophet

def main():
    # Set up the page configuration for Streamlit
    st.set_page_config(
        page_title="Advanced Sales Forecasting Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Page Title
    st.title("Advanced Sales Forecasting Dashboard")

    # Sidebar Configuration
    st.sidebar.title("Forecast Configuration")

    # Load the Prophet model
    with st.spinner("Loading model..."):
        # model = load("prophet_sales_model(1).joblib")
        model = load("prophet_sales_model (1).joblib")
    st.sidebar.success("Model loaded successfully.")

    # Sidebar input for the forecast horizon (weeks)
    input_weeks = st.sidebar.number_input(
        "Forecast horizon (in weeks):",
        min_value=1,
        max_value=104,
        value=4,
        step=1
    )

    # Checkbox to optionally show entire forecast (historical + future)
    show_entire_forecast = st.sidebar.checkbox(
        "Show entire forecast (historical + future)",
        value=False
    )

    # Dropdown (selectbox) to optionally display the table data
    display_table_option = st.sidebar.selectbox(
        "Display Forecast Data Table?",
        options=["No", "Yes"],
        index=0
    )

    # Button to trigger the forecast
    if st.sidebar.button("Generate Forecast"):
        # Create the future dataframe
        future = model.make_future_dataframe(
            periods=input_weeks,
            freq='W-MON'
        )

        # Generate the forecast
        forecast = model.predict(future)

        # Decide how much of the forecast to display
        if show_entire_forecast:
            df_forecast = forecast.copy()
        else:
            # Show only newly forecasted portion
            df_forecast = forecast[
                forecast['ds'] > future['ds'].max() - pd.Timedelta(weeks=input_weeks)
            ]

        # ----------------------------------------
        # Conditionally display the forecast table
        # ----------------------------------------
        if display_table_option == "Yes":
            st.subheader("Forecasted Data")
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

        # Add forecast line (yhat)
        fig.add_trace(go.Scatter(
            x=df_forecast['ds'],
            y=df_forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=2)
        ))

        # Add confidence interval fill
        fig.add_trace(go.Scatter(
            x=pd.concat([df_forecast['ds'], df_forecast['ds'][::-1]]),
            y=pd.concat([df_forecast['yhat_upper'], df_forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(135, 206, 250, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))

        # Update layout (titles, legend, template)
        fig.update_layout(
            title_text=f"Sales Forecast for Next {input_weeks} Week(s)",
            xaxis_title="Date",
            yaxis_title="Sales",
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            template='plotly_white'
        )

        # ----------------------------------------
        # Fix the range selector to use days instead of "week"
        # ----------------------------------------
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=28, label="4w", step="day", stepmode="backward"),   # 4 weeks = 28 days
                    dict(count=84, label="12w", step="day", stepmode="backward"),  # 12 weeks = 84 days
                    dict(count=168, label="24w", step="day", stepmode="backward"), # 24 weeks = 168 days
                    dict(step="all", label="All")
                ])
            )
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Configure your forecast options in the sidebar and click 'Generate Forecast'.")

# Run the app
if __name__ == "__main__":
    main()
