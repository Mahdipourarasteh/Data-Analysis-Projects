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

#visualize the density of rides per Day of month
fig1,ax = plt.subplots(figsize= (12,6))
plt.hist(uber_df.Day, width= 0.6, bins= 30)
plt.title("Density of trips per Day", fontsize=16)
plt.xlabel("Day", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)
fig1.show()

#Visualize the Density of rides per Weekday
fig2,ax = plt.subplots(figsize = (12,6))
plt.hist(uber_df.Weekday, width= 0.6, range= (0, 6.5), bins=7, color= "green")
plt.title("Density of trips per Weekday", fontsize=16)
plt.xlabel("Weekday", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)
fig2.show()

#Visualize the Density of rides per hour
fig3,ax = plt.subplots(figsize = (12,6))
plt.hist(uber_df.Hour, width= 0.6, bins=24, color= "orange")
plt.title("Density of trips per Hour", fontsize=16)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)
fig3.show()

#Visualize the Density of rides per location
fig4,ax = plt.subplots(figsize = (12,6))
x= uber_df.Lon
y= uber_df.Lat
plt.scatter(x, y, color= "purple")
plt.title("Density of trips per Hour", fontsize=16)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)
fig4.show()

input()