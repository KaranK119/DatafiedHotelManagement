import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings


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
