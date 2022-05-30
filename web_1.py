import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import re


df = pd.read_csv('electriccars.csv')

st.title("Find out your affordable ElectricCar")


#Data Cleaning 

# extract numeric data
def extract_num(x):
    """
    this function extracts the numeric data from the string 
    and converts the data type to float. 
    It uses a regex to extact intergers and floats.
    """
    return float(re.findall(r"[-+]?\d*\.?\d+|\d+", x)[0])

#df['Acceleration'][0]

print(extract_num(df['Acceleration'][0]))
print(type(extract_num(df['Acceleration'][0])))

df['BatteryCapacity'] = df['Subtitle'].apply(extract_num)

# extract brand name from name
df['BrandName'] = df['Name'].apply(lambda x: x.split()[0])

# example
#df['Name'][0].split()

#df['Name'][0].split()[0]

for col_name in ['Acceleration', 'TopSpeed', 'Range', 'Efficiency']:
    df[col_name] = df[col_name].apply(extract_num)
    
df.drop(columns=['PriceinUK', 'Subtitle'] , axis=1, inplace=True)    

df['PriceinGermany'] = df['PriceinGermany']\
                                           .fillna('-1')\
                                           .apply(lambda x: re.sub(',', '', x))\
                                           .apply(extract_num)\
                                           .replace(-1, np.nan)

FastChargeSpeed=[]
for item in df['FastChargeSpeed']:
    FastChargeSpeed+=[int(item.replace(' km/h','').replace('-','0'))]
df['FastChargeSpeed']=FastChargeSpeed

df['FastChargeSpeed'] = df['FastChargeSpeed'].replace(0, np.nan)

df1 = df.dropna()

cars = df1.drop_duplicates()

cars.Name = cars.Name.str.strip()

cars_1 = cars.rename(columns = {"Acceleration": "acceleration_in_sec", 
                                "Range": "range_km", 
                                "TopSpeed": "top_speed_km_h",
                                "Efficiency": "efficiency_Wh_km",
                                "FastChargeSpeed": "fast_charge_speed_km_h",
                                "PriceinGermany": "price_in_euros",
                                "BatteryCapacity": "batter_capacity_kWh"
                               })

# Displays top 10 brands of a car

st.write("""
### Top 10 Brands
""")

def top_brands_10(cars_1: pd.DataFrame):
    return(
       cars_1
         .groupby(['BrandName'])['price_in_euros']
         .sum()
         .sort_values(ascending=False)
         .head(10)
         .reset_index()
          )

top_brands = top_brands_10(cars_1)

st.dataframe(top_brands)

#Displays top 10 Cars 

st.write("""
### Top 10 Cars
""")


def top_cars_10(cars_1 : pd.DataFrame) :
    return(
       cars_1
        .groupby(['Name', 'batter_capacity_kWh', 'top_speed_km_h', 'fast_charge_speed_km_h'])['price_in_euros']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
      )
top_brands = top_cars_10(cars_1)

st.dataframe(top_brands)

# Displays the Price and it features if we give car name  as Input

st.write("""
### Choose the Name of the car 
""")

name = st.selectbox(
    ' ',
     (cars_1['Name'].unique()))

def name_of_car(cars_1: pd.DataFrame, Name: str):
    
    data=cars_1.copy()
    
    return(data         
    .query('Name == @Name')  
    .filter(['price_in_euros','batter_capacity_kWh', 'acceleration_in_sec', 'TopSpeed', 'fast_charge_speed_km_h', 
             'range_km', 'efficiency_Wh_km', 'Drive', 'NumberofSeats'])        
    )

output_1 = name_of_car(cars_1, name)

st.write("""
### Car Features  
""")
    
st.dataframe(output_1)


# To Choose the Price range of a Cars 

st.write("""
### Choose the Price Range 
""")

numberFrom,numberTo = st.slider('Select the Price Range', value=[min(cars_1['price_in_euros']),max(cars_1['price_in_euros'])])
# numberFrom = range[0]
# numberTo = range[1]

def price_range(cars_1: pd.DataFrame, from_PriceinGermany: int, to_PriceinGermany: int):
    data=cars_1.copy()
    return(data    
    .query('price_in_euros >= @from_PriceinGermany & price_in_euros <= @to_PriceinGermany')
     .filter(['Name','batter_capacity_kWh', 'acceleration_in_sec','top_speed_km_h', 'fast_charge_speed_km_h', 
             'range_km', 'efficiency_Wh_km', 'Drive', 'NumberofSeats', 'price_in_euros']) 
          )

output_2 = price_range(cars_1, numberFrom, numberTo)
st.dataframe(output_2)

##To Choose the Range of a Car 

st.write("""
### Choose the Range of the Car
""")

numberFrom,numberTo = st.slider('Select the Range of the Car', value=[min(cars_1['range_km']),max(cars_1['range_km'])])

def range_of_car(cars_1: pd.DataFrame, from_range: int, to_range: int):
    
    data=cars_1.copy()
    
    return(data    
    .query('range_km >= @from_range & range_km <= @to_range')
    .filter(['Name', 'price_in_euros', 'batter_capacity_kWh', 'acceleration_in_sec','top_speed_km_h', 
             'fast_charge_speed_km_h', 'range_km', 'efficiency_Wh_km', 'Drive', 'NumberofSeats'])        
          )


output_4 = range_of_car(cars_1, 150,200)
st.dataframe(output_4)
