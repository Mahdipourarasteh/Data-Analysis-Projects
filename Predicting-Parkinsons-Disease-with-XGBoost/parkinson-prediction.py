import os, sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import joblib

parkinson_df= pd.read_csv('parkinsons.csv')
pd.set_option("display.max_columns", None)

#print(parkinson_df.head(5))
#print(parkinson_df.shape)
#rint(parkinson_df.info())
#print(parkinson_df.describe())
#print(parkinson_df.corr())

### Dataset attributes 
#### • MDVP:Fo(Hz) - Average vocal fundamental frequency
#### • MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
#### • MDVP:Flo(Hz) - Minimum vocal fundamental frequency
#### • MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency
#### • MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
#### • NHR,HNR - Two measures of ratio of noise to tonal components in the voice
#### • status - Health status of the subject (one) - Parkinson's, (zero) - healthy
#### • RPDE,D2 - Two nonlinear dynamical complexity measures
#### • DFA - Signal fractal scaling exponent
#### • spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation

sns.countplot(data= parkinson_df, x='status')

fig, ax= plt.subplots(figsize=(12,8))
corr= parkinson_df.corr()
ax= sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap= sns.diverging_palette(20, 220, n=200))

#####feature selection

#Rearrange the columns
parkinson_df = parkinson_df[["name", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE", "status"]]

#Create a copy of the original dataset
df2= parkinson_df.copy()

#Assign numeric values to the binary and categorical columns
number= LabelEncoder()
df2['name']= number.fit_transform(df2['name'])
print(df2.head(5))

plt.show()