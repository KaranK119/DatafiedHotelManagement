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
st.title("📊Datafied Hotel Management")
#Graph to predict last bit of 2024 revenue ---

st.title("SARIMAX Predicting for Facility Revenue")
# take in the data
@st.cache_data
def load_data():
    

    #IF DATASET needs to be changed please do so here---->>>> and upload to github with specific file name.


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

    #Reimport in case
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

#prepare

df['BusinessDate'] = pd.to_datetime(df['BusinessDate'])
df['FacilityID'] = df['FacilityID'].astype(str).str[:-2]
df['Revenue'] = df['Revenue'].str.replace(',', '').astype(float)
df['F&B Revenue'] = df['F&B Revenue'].str.replace(',', '').astype(float)
df['TotalRevenue'] = df['TotalRevenue'].str.replace(',', '').astype(float)
df.set_index('BusinessDate', inplace=True)

#unique ids
unique_facilities = df["FacilityID"].unique()


st.title('SARIMAX Forecasting for Facility Revenue')
facility_id = st.selectbox('🏨  Select a Facility', unique_facilities)

#prediction
def run_sarimax(facility_id):
    df_facility = df[df["FacilityID"] == facility_id]
    df_facility_m = df_facility[['Sold', 'Occ', 'ADR', 'RevPAR', 'F&B Revenue']]
    X = df_facility_m
    y = df_facility['TotalRevenue']

   
    train_len = int(len(df_facility) * 0.8)
    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]
    sarimax = SARIMAX(y_train, exog=X_train, order=(7, 1, 7))
    sarimax_results = sarimax.fit(disp=0)
    sarimax_model = SARIMAX(y, exog=X, order=(7, 1, 7))
    sarimax_result = sarimax_model.fit(disp=0)
    # forcast for the next 30 days
    forecast_days = 30
    future_exog = X.tail(forecast_days).copy().reset_index(drop=True)
    sarimax_forecast = sarimax_result.forecast(steps=forecast_days, exog=future_exog)

    #finish
    forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    
    sarimax_forecast.index = forecast_index

   
    plt.figure(figsize=(14, 6))
    plt.plot(df_facility['TotalRevenue'], label='Actual Total Revenue')
    plt.plot(sarimax_forecast, label='Forecast (Next 30 Days)', linestyle='--', color='orange')
    plt.title(f'SARIMAX Forecast for Facility {facility_id}')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

run_sarimax(facility_id)


# End of the forcasting 30 days

#Seperate EDA viewpoints
#still have to push to github

#weekdays of all the facilities in given dataset.



st.title('EDA (Explatory Data Analysis) of Weekdays of all Facilities Together')

#libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#data prep

df['BusinessDate'] = pd.to_datetime(df['BusinessDate'])
df['Weekday'] = df['BusinessDate'].dt.day_name()
df['Occ'] = pd.to_numeric(df['Occ'], errors='coerce')
#setting the parameters
avg_occ_by_weekday = df.groupby('Weekday')['Occ'].mean().sort_values()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_occ_by_weekday = avg_occ_by_weekday.reindex(day_order)
#making the graph
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(avg_occ_by_weekday.index, avg_occ_by_weekday.values, color='skyblue')
ax.set_xlabel('Day of the Week', fontsize=14)
ax.set_ylabel('Average Occupancy Rate (%)', fontsize=14)
ax.set_title('Average Occupancy Rate by Day of the Week', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


#revpar grid

st.title('EDA (Explatory Data Analysis) of RevPAR of each Unique facility')
#libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
#data prep

df['BusinessDate'] = pd.to_datetime(df['BusinessDate'])
df['FacilityID'] = df['FacilityID'].astype(str).str[:-2]
df['RevPAR'] = pd.to_numeric(df['RevPAR'], errors='coerce')
unique_facilities = df['FacilityID'].unique()
#making grid and dashboard feel
n_facilities = len(unique_facilities)
n_cols = 2
n_rows = int(np.ceil(n_facilities / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
axs = axs.flatten()
#loop to get each distint graph
for i, facility in enumerate(unique_facilities):
    df_fac = df[df['FacilityID'] == facility].sort_values('BusinessDate')
    axs[i].plot(df_fac['BusinessDate'], df_fac['RevPAR'], color='teal', linewidth=1)
    axs[i].set_title(f'Facility {facility}')
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('RevPAR')
    axs[i].tick_params(axis='x', rotation=45)
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
st.pyplot(fig)


#--- End of EDA(s)

