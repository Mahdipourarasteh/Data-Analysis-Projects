import pandas as pd
import matplotlib.pyplot as plt

uber_df= pd.read_csv("uber-raw-data-sep14.csv")
#print(uber_df.head(5))
#print(uber_df.tail())
#print(uber_df.shape)
#print(uber_df.info())

#Change the "Date/Time" column's data type from string to datetime
uber_df['Date/Time']= pd.to_datetime(uber_df['Date/Time'])

uber_df["Day"]= uber_df['Date/Time'].apply(lambda x: x.day)
uber_df["Hour"]= uber_df['Date/Time'].apply(lambda x: x.hour)
uber_df["Weekday"]= uber_df['Date/Time'].apply(lambda x: x.weekday())
#print(uber_df.head(5))

