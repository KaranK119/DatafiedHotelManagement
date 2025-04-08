# here I will clean up the merged file.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
plt.rcParams["figure.figsize"] = (10,6)
import datetime
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

df = pd.read_csv('merged_data_2021_2022_2023_2024.csv')
df['BusinessDate'] = pd.to_datetime(df['BusinessDate'])
df['FacilityID'] = df['FacilityID'].astype(str)
df['FacilityID'] = df['FacilityID'].str[:-2]
df['Revenue'] = df['Revenue'].str.replace(',', '').astype(float)
df['F&B Revenue'] = df['F&B Revenue'].str.replace(',', '').astype(float)
df['TotalRevenue'] = df['TotalRevenue'].str.replace(',', '').astype(float)
df.set_index('BusinessDate', inplace=True)
df_19713 = df[df['FacilityID']=='19713']
df_15671 = df[df['FacilityID']=='15671']
df_9550 = df[df['FacilityID']=='9550']
df_60421 = df[df['FacilityID']=='60421']
df_87722 = df[df['FacilityID']=='87722']



#Making it be dynamically useable for properties in file
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns

#getting all ids from the dataset
unique_facilities = df["FacilityID"].unique()

#drop down menue
facility_dropdown = widgets.Dropdown(
    options= unique_facilities ,
    description="🏨 Facility:",
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px', height='40px')
)


# Function to run SARIMAX model for selected FacilityID
def run_sarimax(facility_id):
    
    df_facility = df[df["FacilityID"] == facility_id].copy()
    df_facility = df_facility.sort_values("BusinessDate")
    df_facility_m = df_facility[['Sold', 'Occ', 'ADR', 'RevPAR', 'F&B Revenue']]
    X = df_facility_m
    y = df_facility['TotalRevenue']
    train_len = int(len(df_facility) * 0.8)
    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]
    sarimax = sm.tsa.statespace.SARIMAX(y_train, exog=X_train, order=(7, 1, 7))
    sarimax_results = sarimax.fit(disp=0)
    start = len(X_train)
    end = len(X_train) + len(X_test) - 1
    sarimax_pred = sarimax_results.predict(start, end, exog=X_test)
    sarimax_pred.index = X_test.index




    plt.figure(figsize=(12, 5))
    plt.plot(y_train, label='Train')
    plt.plot(y_test, label='Test')
    plt.plot(sarimax_pred, label='SARIMAX Predictions', linestyle="dashed")
    plt.title(f'SARIMAX Predictions for Facility {facility_id}')
    plt.xlabel('Date')
    plt.legend()
    plt.show()
    

#UI
interactive_plot = widgets.interactive(run_sarimax, facility_id=facility_dropdown)
display(interactive_plot)



