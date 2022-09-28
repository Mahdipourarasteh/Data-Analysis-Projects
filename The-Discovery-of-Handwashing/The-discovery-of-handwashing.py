import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the yearly dataset
yearly_df= pd.read_csv("yearly_deaths_by_clinic.csv")
# print(yearly_df)
# print(yearly_df.shape)
# print(yearly_df.info())

#print(yearly_df.groupby("clinic")["deaths"].sum())

#To make the analysis easier, we can calculate the proportion of deaths.
yearly_df["proportion of deaths"]= yearly_df["deaths"]/ yearly_df["births"]
#print(yearly_df)

#Separate the dataset into 2 datasets, one for each clinic
clinic1= yearly_df[yearly_df["clinic"] == "clinic 1"]
clinic2= yearly_df[yearly_df["clinic"] == "clinic 2"]
#print(clinic1)
#print(clinic2)

#Visualize the Number of deaths every year in clinic 1
fig1,ax = plt.subplots(figsize= (10,4))
plt.bar(clinic1.year, clinic1.deaths, width= 0.6, color= "red")
plt.title("Clinic 1: Number of Deaths per Year", fontsize= 16)
plt.xlabel("year", fontsize= 14)
plt.ylabel("Number of Deaths", fontsize= 14)

#Visualize the Number of deaths every year in clinic 2
fig2,ax = plt.subplots(figsize= (10,4))
plt.bar(clinic2.year, clinic2.deaths, width= 0.6, color= "green")
plt.title("Clinic 2: Number of Deaths per Year", fontsize= 16)
plt.xlabel("year", fontsize= 14)
plt.ylabel("Number of Deaths", fontsize= 14)


ax= clinic1.plot(x="year", y="proportion of deaths", label="Clinic 1", color="red")
clinic2.plot(x="year", y="proportion of deaths", label="Clinic 2", ax=ax, ylabel="Proportion of Deaths", color="green")

plt.show()
input()