import pandas as pd
import streamlit as st
from joblib import load
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load the saved model
model = load("best_esg_prediction_model.pkl")

# Fake account credentials for login
fake_account = {"username": "team5", "password": "globalhackathon"}

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

# Function to forecast the next n days based on the model and average data
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
# Login/Signup functionality
def login_page():
    """Handle user login/signup."""
    st.subheader("Login / Sign Up")

    login_option = st.radio("Select Login or Sign Up", ("Login", "Sign Up"))

    if login_option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == fake_account["username"] and password == fake_account["password"]:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
            else:
                st.error("Invalid credentials. Please try again.")

    elif login_option == "Sign Up":
        username = st.text_input("Choose Username")
        password = st.text_input("Choose Password", type="password")

        if st.button("Sign Up"):
            # Here we just simulate account creation by printing the new account details (no real database)
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Account created successfully for {username}! You are now logged in.")

# Check if the user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login_page()  # Show login/signup page
else:
    # Once logged in, show the ESG score prediction page
    section = st.sidebar.radio(
        "Navigation",
        ("Average Values and Predictions", "Visualization", "Conclusions", "Optimization Tips")
    )

    # Shared variables
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            # Load and preprocess data
            input_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success("Data uploaded successfully.")
            input_data['Date'] = pd.to_datetime(input_data['Date'])
            input_data = input_data.sort_values('Date')

            # Required columns
            required_columns = ["Date", "Industry", "Carbon_Emissions", "Governance_Score", 
                                "Social_Score", "Environmental_Score", "ESG_Score"]

            if all(col in input_data.columns for col in required_columns):
                lag_days = 30
                input_data = create_lag_features(input_data, "ESG_Score", lag_days).dropna()
                feature_columns = [col for col in input_data.columns if col not in ['Date', 'Industry', 'ESG_Score']]
                avg_values = input_data[feature_columns].mean()
                avg_data = avg_values.values

                # Calculate predictions globally
                prediction_horizon = st.slider("Select prediction horizon (days)", 1, 30, 3)
                future_predictions = forecast_next_days(model, avg_data, num_days=prediction_horizon)
                future_dates = generate_future_dates(input_data['Date'].iloc[-1], prediction_horizon)

                prediction_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_ESG_Score': future_predictions
                })

                # Section-specific logic
                if section == "Average Values and Predictions":
                    st.subheader("Average Values")
                    st.write(avg_values)

                    st.subheader("Predicted ESG Scores for Next Days")
                    st.write(prediction_df)

                elif section == "Visualization":
                    st.subheader("Visualization")

                    # Select date range
                    start_date = st.date_input("Start Date", input_data['Date'].min())
                    end_date = st.date_input("End Date", input_data['Date'].max())
                    filtered_data = input_data[
                        (input_data['Date'] >= pd.to_datetime(start_date)) & 
                        (input_data['Date'] <= pd.to_datetime(end_date))
                    ]

                    # Select granularity for the data
                    granularity = st.radio(
                        "Select data granularity for visualization",
                        ("Daily", "Monthly", "Yearly")
                    )

                    # Ensure only numeric columns are aggregated
                    numeric_columns = filtered_data.select_dtypes(include=['number']).columns
                    filtered_data = filtered_data.set_index('Date')

                    if granularity == "Daily":
                        aggregated_data = filtered_data.reset_index()
                    elif granularity == "Monthly":
                        aggregated_data = filtered_data[numeric_columns].resample('M').mean().reset_index()
                    elif granularity == "Yearly":
                        aggregated_data = filtered_data[numeric_columns].resample('Y').mean().reset_index()

                    # Chart type selection
                    chart_type = st.radio(
                        "Select chart type",
                        ("Line Chart", "3D Scatter Plot")
                    )

                    if chart_type == "Line Chart":
                        historical_data = aggregated_data[['Date', 'ESG_Score']]
                        combined_data = pd.concat([
                            historical_data,
                            pd.DataFrame({'Date': future_dates, 'ESG_Score': future_predictions})
                        ])

                        plt.figure(figsize=(10, 6))
                        sns.lineplot(data=combined_data, x='Date', y='ESG_Score', marker='o')
                        plt.title(f"ESG Scores: Historical and Predicted ({granularity} Data)")
                        plt.xticks(rotation=45)
                        st.pyplot()

                    elif chart_type == "3D Scatter Plot":
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')

                        # Plot using three features
                        ax.scatter(
                            aggregated_data['Carbon_Emissions'],
                            aggregated_data['Governance_Score'],
                            aggregated_data['Social_Score'],
                            c=aggregated_data['Environmental_Score'],
                            cmap='viridis',
                            s=50
                        )

                        ax.set_xlabel("Carbon Emissions")
                        ax.set_ylabel("Governance Score")
                        ax.set_zlabel("Social Score")
                        ax.set_title(f"3D Scatter Plot of Features ({granularity} Data)")

                        st.pyplot(fig)


                elif section == "Conclusions":
                    st.subheader("Conclusions")
                    st.write(f"The average values of the features are as follows:\n{avg_values}")
                    st.write("The predicted ESG scores for the next days are:")
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
