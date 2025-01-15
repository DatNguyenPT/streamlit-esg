import pandas as pd
import streamlit as st
from joblib import load
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# Load the saved model
model = load("best_esg_prediction_model.pkl")

# Function to create lag features
def create_lag_features(data, target_column, lag_days):
    """Create lag features for time series data."""
    df = data.copy()
    for lag in range(1, lag_days + 1):
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df

# Function to generate future dates
def generate_future_dates(last_date, num_days):
    """Generate future dates for prediction."""
    future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]
    return future_dates

# Function to forecast the next 3 days based on the model and average data
def forecast_next_days(model, avg_data, num_days=3):
    """Forecast the next specified number of days using the average data."""
    predictions = []
    current_input = avg_data.copy()

    for _ in range(num_days):
        prediction = model.predict(current_input.reshape(1, -1))[0]
        predictions.append(prediction)
        current_input = np.roll(current_input, -1)
        current_input[-1] = prediction

    return predictions


# Streamlit app setup
st.set_page_config(layout="wide")
st.title("ESG Score Prediction by ESGLOBAL - Global Hackathon 2025")

# Sidebar navigation
section = st.sidebar.radio(
    "Navigation",
    ("Average Values and Predictions", "Visualization", "Conclusions", "Optimization Tips")
)

# Shared variables
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load data
        input_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success("Data uploaded successfully.")

        # Required columns
        required_columns = [
            "Date", "Industry", "Carbon_Emissions", "Governance_Score",
            "Social_Score", "Environmental_Score", "ESG_Score"
        ]

        if all(col in input_data.columns for col in required_columns):
            input_data['Date'] = pd.to_datetime(input_data['Date'])
            input_data = input_data.sort_values('Date')
            lag_days = 30
            input_data = create_lag_features(input_data, "ESG_Score", lag_days).dropna()
            
            feature_columns = [col for col in input_data.columns if col not in ['Date', 'Industry', 'ESG_Score']]
            input_data_features = input_data[feature_columns]
            avg_values = input_data_features.mean()

            avg_data = avg_values.values
            future_predictions = forecast_next_days(model, avg_data, num_days=3)
            future_dates = generate_future_dates(input_data['Date'].iloc[-1], 3)

            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_ESG_Score': future_predictions
            })

            # Section-specific logic
            if section == "Average Values and Predictions":
                st.subheader("Average Values")
                st.write(avg_values)

                st.subheader("Predicted ESG Scores for Next 3 Days")
                st.write(prediction_df)

            elif section == "Visualization":
                st.subheader("Visualization")

                # Select date range
                start_date = st.date_input("Start Date", input_data['Date'].min())
                end_date = st.date_input("End Date", input_data['Date'].max())
                filtered_data = input_data[(input_data['Date'] >= pd.to_datetime(start_date)) & (input_data['Date'] <= pd.to_datetime(end_date))]
                
                historical_data = filtered_data[['Date', 'ESG_Score']]
                combined_data = pd.concat([
                    historical_data,
                    pd.DataFrame({'Date': future_dates, 'ESG_Score': future_predictions})
                ])

                plt.figure(figsize=(10, 6))
                sns.lineplot(data=combined_data, x='Date', y='ESG_Score', marker='o')
                plt.title("ESG Scores: Historical and Predicted")
                plt.xticks(rotation=45)
                st.pyplot()

            elif section == "Conclusions":
                st.subheader("Conclusions")
                st.write(f"The average values of the features are as follows:\n{avg_values}")
                st.write("The predicted ESG scores for the next 3 days are:")
                st.write(prediction_df)

            elif section == "Optimization Tips":
                st.subheader("Optimization Tips")
                st.write("""
                Based on the provided data, here are some recommendations:
                - **Environmental**: Reduce carbon emissions by adopting renewable energy sources.
                - **Social**: Invest in employee welfare programs and community outreach.
                - **Governance**: Ensure transparency in leadership and enhance shareholder rights.
                """)

        else:
            st.error(f"The file is missing required columns: {', '.join(required_columns)}.")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a file to proceed.")
