import os, sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
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

fig, ax= plt.subplots(figsize=(5,4))
sns.countplot(data= parkinson_df, x='status')

fig, ax= plt.subplots(figsize=(10,7))
corr= parkinson_df.corr()
ax= sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap= sns.diverging_palette(20, 220, n=200))

##################### feature selection ########################

#Rearrange the columns
parkinson_df = parkinson_df[["name", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE", "status"]]

#Create a copy of the original dataset
df2= parkinson_df.copy()

#Assign numeric values to the binary and categorical columns
number= LabelEncoder()
df2['name']= number.fit_transform(df2['name'])
#print(df2.head(10))

x= df2.iloc[:, 0:11]  #all features
y= df2.iloc[:, -1]   #target (status of parkinson)

best_features= SelectKBest(score_func=chi2, k=3)  #function that select the top 3 features.
fit= best_features.fit(x, y)

#Creating dataframes for the features and the score of each feature.
parkinson_scores= pd.DataFrame(fit.scores_)
parkinson_columns= pd.DataFrame(x.columns)

#Create a dataframe that combines all the features and their corresponding scores.
feature_scores= pd.concat([parkinson_scores, parkinson_scores], axis=1)
feature_scores.columns= ['Features', 'Score']
feature_scores.sort_values(by= 'Score')
#print(feature_scores)

######################## Build the Model ################################

x= parkinson_df[["MDVP:Flo(Hz)", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]]
y= parkinson_df[["status"]]
x_train,x_test,y_train,y_test= train_test_split(x, y, test_size=0.2, random_state=7)

model= XGBClassifier()
model.fit(x_train,y_train)

######################## Evaluate the Model #############################

y_pred= model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

#define metrics
y_pred_proba= model.predict_proba(x_test) [::,1]

#Calculate true positive and false positive rates
false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba)

#Calculate the area under curve to see the model performance
auc= metrics.roc_auc_score(y_test, y_pred_proba)

#Create ROC curve
fig, ax= plt.subplots(figsize=(5,4))
plt.plot(false_positive_rate, true_positive_rate,label="AUC="+str(auc))
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('false Positive Rate')
plt.legend(loc=4)

# Save the trained model to a file to be used in future predictions 
joblib.dump(model, 'XG.pkl')

plt.show()