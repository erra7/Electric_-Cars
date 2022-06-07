import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.linear_model import LinearRegression

ecars = pd.read_csv('electriccars.csv')
ecars.drop(columns=['PriceinUK'] , axis=1, inplace=True)

st.title("Find out your affordable ElectricCar")

# data cleaning
ecars_cl = (ecars
    .assign(
        drive = lambda df_: df_['Drive'].str.lower(),
        acceleration_in_sec = lambda df_ : df_['Acceleration'].str.replace(' sec', '').astype('float32'), 
        top_speed_km_h = lambda df_ : df_['TopSpeed'].str.replace(' km/h','').astype('int32'), 
        range_km = lambda df_ : df_['Range'].str.replace(' km','').astype('int32'), 
        price = lambda df_ : df_['PriceinGermany'].str.replace('[ €]|[,]', '', regex=True).astype('float32')
        )
    .filter(['acceleration_in_sec','top_speed_km_h','range_km', 'price','drive'])       
    .dropna()
)

ecars_cl.sample(5)

# model training
pred_vars = ['acceleration_in_sec','range_km','top_speed_km_h','drive']
X = ecars_cl[pred_vars]
y = ecars_cl['price']

# pre process
X_processed = pd.get_dummies(X, drop_first=True)

# lm training
lm = LinearRegression()
lm.fit(X_processed, y)

with st.sidebar:

    # acceleration in seconds
    acc_sel = st.slider(
        'Select acceleration in seconds', 
        min(ecars_cl['acceleration_in_sec']),
        max(ecars_cl['acceleration_in_sec']),
        step = 0.5
    )
    
with st.sidebar:

    # acceleration in seconds
    range_sel = st.slider(
        'Select range in kilometers', 
        min(ecars_cl['range_km']),
        max(ecars_cl['range_km']),
        step = 1
    )
    
with st.sidebar:

    # acceleration in seconds
    speed_sel = st.slider(
        'Select speed in kilometers per hour', 
        min(ecars_cl['top_speed_km_h']),
        max(ecars_cl['top_speed_km_h']),
        step = 1
    )

    drive_sel = st.selectbox('Select drive', ecars_cl.drive.unique(),key = 'drive_slider')


# create df to run model
car_to_predict = {
    'acceleration_in_sec': acc_sel,
    'range_km': range_sel,
    'top_speed_km_h': speed_sel,
    'drive_front wheel drive':0, 
    'drive_all wheel drive':0,
    'drive_rear wheel drive':0
    }

car_to_predict['drive_' + drive_sel] = 1

car_to_predict = (
    pd.DataFrame(car_to_predict, index=[0])
    .filter(X_processed.columns.tolist())
    )

# run model
pred_price = lm.predict(car_to_predict)[0]


# print predicted price
st.write('Predicted price for your car ', str(round(pred_price) / 1_000), "€")

# suggest cars from the dataset
st.write('Cars you may be interested:')


sel_cars_id = ecars_cl.query(f"({acc_sel - 2} < acceleration_in_sec < {acc_sel + 2}) & ({range_sel - 100} < range_km < {range_sel + 100}) & ({speed_sel - 10} < top_speed_km_h < {speed_sel + 10}) & (drive == '{str(drive_sel)}')").index
st.dataframe(
    ecars.iloc[sel_cars_id,:]
)