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
pred_vars = ['acceleration_in_sec','range_km','drive']
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

    drive_sel = st.selectbox('Select drive', ecars_cl.drive.unique(),key = 'drive_slider')


# create df to run model
car_to_predict = {
    'acceleration_in_sec': acc_sel,
    'range_km': range_sel,
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


sel_cars_id = ecars_cl.query(f"({acc_sel - 2} < acceleration_in_sec < {acc_sel + 2}) & ({range_sel - 100} < range_km < {range_sel + 100}) & (drive == '{str(drive_sel)}')").index
st.dataframe(
    ecars.iloc[sel_cars_id,:]
)

# again import 

e_cars = pd.read_csv(r'Python/Jupyter/Final_Project_ElectricCars/electriccars.csv')
e_cars.drop(columns=['PriceinUK'] , axis=1, inplace=True)
electric_cars = e_cars.dropna()

# Displays the Price and it features if we give car name  as Input

st.write("""
### Are you Interested in Specific Car? 
""")

name = st.selectbox('Choose the name of the car which you like',(electric_cars['Name'].unique()))

def name_of_car(electric_cars: pd.DataFrame, Name: str):
    
    data=electric_cars.copy()
    
    return(data         
    .query('Name == @Name')  
    .filter(['Name', 'Subtitle','Acceleration', 'TopSpeed', 'Range','Efficiency', 'FastChargeSpeed', 'Drive',                   'NumberofSeats', 'PriceinGermany'])       
    )

output_1 = name_of_car(electric_cars, name)

st.text("""
 Car Features  
""")
    
st.dataframe(output_1)

# data cleaning 

# extract brand name from name
electric_cars['BrandName'] = electric_cars['Name'].apply(lambda x: x.split()[0])

# extract numeric data
def extract_num(x):
    """
    this function extracts the numeric data from the string 
    and converts the data type to float. 
    It uses a regex to extact intergers and floats.
    """
    return float(re.findall(r"[-+]?\d*\.?\d+|\d+", x)[0])

electric_cars['PriceinGermany'] = electric_cars['PriceinGermany']\
                                           .fillna('-1')\
                                           .apply(lambda x: re.sub(',', '', x))\
                                           .apply(extract_num)\
                                           .replace(-1, np.nan)

# Displays top 10 brands of a car

st.write("""
### Top 10 Brands
""")

def top_brands_10(electric_cars: pd.DataFrame):
    return(
       electric_cars
         .groupby(['BrandName'])['PriceinGermany']
         .sum()
         .sort_values(ascending=False)
         .head(10)
         .reset_index()
          )

top_brands = top_brands_10(electric_cars)

st.dataframe(top_brands)

#Displays top 10 Cars 

st.write("""
### Top 10 Cars
""")

def top_cars_10(electric_cars : pd.DataFrame) :
    return(
       electric_cars
        .groupby(['Name','Acceleration', 'TopSpeed', 'Range','Efficiency','FastChargeSpeed','Drive','NumberofSeats']) ['PriceinGermany']      
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
)

top_cars = top_cars_10(electric_cars)

st.dataframe(top_cars)
