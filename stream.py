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
st.title("SARIMAX Predicting for Facility Revenue")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv('merged_data_2021_2022_2023_2024.csv')
    
    # Normalize column names
    df.columns = df.columns.str.strip()
    
    df['BusinessDate'] = pd.to_datetime(df['BusinessDate'])
    df['FacilityID'] = df['FacilityID'].astype(str).str[:-2]
    df['Revenue'] = df['Revenue'].astype(str).str.replace(',', '', regex=False).astype(float)
    df['F&B Revenue'] = df['F&B Revenue'].astype(str).str.replace(',', '', regex=False).astype(float)
    df['TotalRevenue'] = df['TotalRevenue'].astype(str).str.replace(',', '', regex=False).astype(float)
    
    df.set_index('BusinessDate', inplace=True)
    return df

df = load_data()

# Facility selection
unique_facilities = df['FacilityID'].unique()
selected_facility = st.selectbox("🏨 Select a Facility ID", unique_facilities)
df_facility = df[df['FacilityID'] == selected_facility].copy().sort_index()

if len(df_facility) < 30:
    st.warning("Not enough data for this facility to train SARIMAX model.")
else:
    # Features and target
    df_facility_m = df_facility[['Sold', 'Occ', 'ADR', 'RevPAR', 'F&B Revenue']]
    y = df_facility['TotalRevenue']
    X = df_facility_m

    train_len = int(len(df_facility) * 0.8)
    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    # SARIMAX model
    model = SARIMAX(y_train, exog=X_train, order=(7, 1, 7))
    results = model.fit(disp=0)

    # Predictions
    start = len(X_train)
    end = len(X_train) + len(X_test) - 1
    predictions = results.predict(start=start, end=end, exog=X_test)
    predictions.index = y_test.index

    # Plot
    st.subheader(f"SARIMAX Forecast for Facility {selected_facility}")
    fig, ax = plt.subplots()
    ax.plot(y_train, label='Train')
    ax.plot(y_test, label='Test')
    ax.plot(predictions, label='Predictions', linestyle='dashed')
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
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
ff = pd.read_csv('merged_data_2021_2022_2023_2024.csv')
ff['BusinessDate'] = pd.to_datetime(ff['BusinessDate'])
ff['Weekday'] = ff['BusinessDate'].dt.day_name()
ff['Occ'] = pd.to_numeric(ff['Occ'], errors='coerce')
#setting the parameters
avg_occ_by_weekday = ff.groupby('Weekday')['Occ'].mean().sort_values()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_occ_by_weekday = avg_occ_by_weekday.reindex(day_order)
#making the graph p
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
gf = pd.read_csv('merged_data_2021_2022_2023_2024.csv')
gf['BusinessDate'] = pd.to_datetime(gf['BusinessDate'])
gf['FacilityID'] = gf['FacilityID'].astype(str).str[:-2]
gf['RevPAR'] = pd.to_numeric(gf['RevPAR'], errors='coerce')
unique_facilities = gf['FacilityID'].unique()
#making grid and dashboard feel
n_facilities = len(unique_facilities)
n_cols = 2
n_rows = int(np.ceil(n_facilities / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
axs = axs.flatten()
#loop to get each distint graph
for i, facility in enumerate(unique_facilities):
    gf_fac = gf[gf['FacilityID'] == facility].sort_values('BusinessDate')
    axs[i].plot(gf_fac['BusinessDate'], gf_fac['RevPAR'], color='teal', linewidth=1)
    axs[i].set_title(f'Facility {facility}')
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('RevPAR')
    axs[i].tick_params(axis='x', rotation=45)
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
st.pyplot(fig)


#--- End of EDA(s)

