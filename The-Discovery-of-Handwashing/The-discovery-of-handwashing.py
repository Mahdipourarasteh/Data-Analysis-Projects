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


#######################################################################################################################

# Read the monthly dataset
monthly_df= pd.read_csv("monthly_deaths.csv")
# print(monthly_df.head(5))
# print(monthly_df.shape)
# print(monthly_df.info())

#Calculate the proportion of deaths per month
monthly_df["proportion of deaths"]= monthly_df["deaths"]/ monthly_df["births"]
#print(monthly_df.head(5))

#Change the data type of "date" column from string to datatime
#print(monthly_df.dtypes)
monthly_df['date']= pd.to_datetime(monthly_df['date'])

# Label the date at which handwashing started to "start_handwashing"
start_handwashing= pd.to_datetime('1847-06-01')

# Split monthly into before and after handwashing_start
before_washing= monthly_df[monthly_df['date'] < start_handwashing]
after_washing= monthly_df[monthly_df['date'] >= start_handwashing]

#befor handwashing
fig, ax= plt.subplots(figsize= (10,4))
x=before_washing['date']
y=before_washing["proportion of deaths"]
plt.plot(x ,y, color= "orange")
plt.title("Before Handwashing", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Proportion of Deaths", fontsize=14)

#after handwashing
fig, ax= plt.subplots(figsize= (10,4))
x=after_washing['date']
y=after_washing["proportion of deaths"]
plt.plot(x ,y, color= "orange")
plt.title("After Handwashing", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Proportion of Deaths", fontsize=14)

#combined before and after handwashing
ax= before_washing.plot(x="date", y="proportion of deaths", label="Before Handwashing", color="orange")
after_washing.plot(x="date", y="proportion of deaths", label="After Handwashing", ax=ax, ylabel="Proportion of Deaths", color="green")



plt.show()  #shows all plots



#Calculation of change of proportion rate before and after hand washing
before_proportion= before_washing["proportion of deaths"]
after_proportion= after_washing["proportion of deaths"]

mean_diff= after_proportion.mean() - before_proportion.mean()
print(mean_diff)