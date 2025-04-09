import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

#Graph to predict last bit of 2024 revenue ---
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = (10, 6)
st.title("📊 SARIMAX Forecasting App for Facilities")

# take in the data
@st.cache_data
def load_data():
    df = pd.read_csv('merged_data_2021_2022_2023_2024.csv')
    df['BusinessDate'] = pd.to_datetime(df['BusinessDate'])
    df['FacilityID'] = df['FacilityID'].astype(str).str[:-2]
    df['Revenue'] = df['Revenue'].str.replace(',', '').astype(float)
    df['F&B Revenue'] = df['F&B Revenue'].str.replace(',', '').astype(float)
    df['TotalRevenue'] = df['TotalRevenue'].str.replace(',', '').astype(float)
    df.set_index('BusinessDate', inplace=True)
    return df

df = load_data()

# unique facilities to look through
unique_facilities = df['FacilityID'].unique()
selected_facility = st.selectbox("🏨 Select a Facility ID", unique_facilities)
df_facility = df[df['FacilityID'] == selected_facility].copy().sort_index()


if len(df_facility) < 30:
    st.warning("Not enough data for this facility to train SARIMAX model.")
else:
    #features
    df_facility_m = df_facility[['Sold', 'Occ', 'ADR', 'RevPAR', 'F&B Revenue']]
    y = df_facility['TotalRevenue']
    X = df_facility_m

    train_len = int(len(df_facility) * 0.8)
    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    
    model = SARIMAX(y_train, exog=X_train, order=(7, 1, 7))
    results = model.fit(disp=0)

    #Make predictions
    start = len(X_train)
    end = len(X_train) + len(X_test) - 1
    predictions = results.predict(start=start, end=end, exog=X_test)
    predictions.index = y_test.index
    st.subheader(f"SARIMAX Forecast for Facility {selected_facility}")
    fig, ax = plt.subplots()
    ax.plot(y_train, label='Train')
    ax.plot(y_test, label='Test')
    ax.plot(predictions, label='Predictions', linestyle='dashed')
    ax.legend()
    st.pyplot(fig)


    #---End of code to predict that end of 2024 revenue

    #Beginning of code that will forcast for 2025
    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('merged_data_2021_2022_2023_2024.csv')
df['BusinessDate'] = pd.to_datetime(df['BusinessDate'])
df['FacilityID'] = df['FacilityID'].astype(str).str[:-2]
df['Revenue'] = df['Revenue'].str.replace(',', '').astype(float)
df['F&B Revenue'] = df['F&B Revenue'].str.replace(',', '').astype(float)
df['TotalRevenue'] = df['TotalRevenue'].str.replace(',', '').astype(float)
df.set_index('BusinessDate', inplace=True)

# Get all unique Facility IDs from the dataset
unique_facilities = df["FacilityID"].unique()

# Streamlit UI elements
st.title('SARIMAX Forecasting for Facility Revenue')
facility_id = st.selectbox('Select a Facility', unique_facilities)

# Function to run SARIMAX model and forecast for selected FacilityID
def run_sarimax(facility_id):
    df_facility = df[df["FacilityID"] == facility_id]
    df_facility_m = df_facility[['Sold', 'Occ', 'ADR', 'RevPAR', 'F&B Revenue']]
    X = df_facility_m
    y = df_facility['TotalRevenue']

    # Train-test split (80% training data, 20% test data)
    train_len = int(len(df_facility) * 0.8)
    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    # Fit SARIMAX model on training data
    sarimax = SARIMAX(y_train, exog=X_train, order=(7, 1, 7))
    sarimax_results = sarimax.fit(disp=0)

    # Refit the model on the full data (for forecasting)
    sarimax_model = SARIMAX(y, exog=X, order=(7, 1, 7))
    sarimax_result = sarimax_model.fit(disp=0)

    # Forecast for the next 30 days
    forecast_days = 30
    future_exog = X.tail(forecast_days).copy().reset_index(drop=True)

    # Forecast the next 30 days
    sarimax_forecast = sarimax_result.forecast(steps=forecast_days, exog=future_exog)

    # Create a date range for the forecast
    forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    # Set the index of the forecasted values
    sarimax_forecast.index = forecast_index

    # Plot the actual data and forecast
    plt.figure(figsize=(14, 6))
    plt.plot(df_facility['TotalRevenue'], label='Actual Total Revenue')
    plt.plot(sarimax_forecast, label='Forecast (Next 30 Days)', linestyle='--', color='orange')
    plt.title(f'SARIMAX Forecast for Facility {facility_id}')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display the plot using Streamlit's built-in method

# Run SARIMAX and display the plot when a facility is selected
run_sarimax(facility_id)
