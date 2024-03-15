# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:32:57 2024

@author: Ashish Chincholikar
Multiclass_Logistic_regression
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

claimants = pd.read_csv("C:/Linear_Regression/Multiple_Correlation_Regression/Dataset/claimants.csv")
#There are CLMAGE and LOSS are having continious data rest are discrete data
#Verify the dataset, where CASENUM is not really useful so dropping it
c1 = claimants.drop('CASENUM' , axis = 1)
c1.head(11)
c1.describe()
#Let us check whether their are NULL values
c1.isna().sum()
#There are several null values
# if we will use dropna() function we will lose almost 290 datapoints
# hence we will go for imputation
c1.dtypes
mean_value = c1.CLMAGE.mean()
mean_value
#Now let us impute the same
c1.CLMAGE = c1.CLMAGE.fillna(mean_value)
c1.CLMAGE.isna().sum()
# hence all null values of CLMAGE has been filled by mena value
# for columns where there are discrete values, we will apply mode imputation
mode_CLMSEX = c1.CLMSEX.mode()
mode_CLMSEX
c1.CLMSEX = c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()
# CLMINSUR is also categorical data hence mode imputation is applied
mode_CLMINSUR = c1.CLMINSUR.mode()
mode_CLMINSUR
c1.CLMINSUR = c1.CLMINSUR.fillna((mode_CLMINSUR)[0])
c1.CLMINSUR.isna().sum()
#SEATBELT is categorical data hence go for model imputation
mode_SEATBELT = c1.SEATBELT.mode()
mode_SEATBELT 
c1.SEATBELT = c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT.isna().sum()
# Now the person we meet an accident will hire the atterney or not
# Let us build the model
logit_model = sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT' , data=c1).fit()
logit_model.summary()
# in logistic regression we do not have the R squared values, only check 
# p=values
#SEATBELT is statistically insignificant ignore and proceed
logit_model.summary2()
# here going to check AIC value, it stands for Akaike information criterion
# is mathematical method for evalutaion how well a model fits the data
# A lower the score more the better model , AIC score are only useful in comparing with other 
# AIC scores for the same dataset

# Now let us go for predictions 
pred = logit_model.predict(c1.iloc[: , 1:])
# here we are applying all rows and columns from 1 , as column0 is ATTORNEY
# Target Value

# let us check the performance of the model
fpr , tpr , thresholds = roc_curve(c1.ATTORNEY , pred)
#we are applying actual values and predicted values are so as to get False
#positive rate , true positive rate and threshold
# the optimal Cutoff value is the point where there is high true positive
# you cand use the below code to get the values:

optimal_idx = np.argmax(tpr-fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#ROC : receiver operating characterstics cure is logistic regression are
# determing best cutoff / threshold values

import pylab as pl
i = np.arange(len(tpr))# index for df
# here tpr is of 559 so it will create a scale of 0 to 558
roc = pd.DataFrame({'fpr' : pd.Series(fpr , index = i),
                    'tpr' : pd.Series(tpr , index = i),
                    '1-fpr' : pd.Series(1-fpr , index = i),
                    'tf' : pd.Series(tpr - (1-fpr) , index = i ),
                    'thresholds' : pd.Series(thresholds , index = i)})
# we want to create a dataframe which comprises of columns fpr, tpr, 1-fpr
# tpr - (1 - fpr) = tf
# the optimal cut off would be where tpr is high and fpr is low
# plot ROC curve
plt.plot(fpr , tpr)
plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
roc_auc = auc(fpr , tpr)
print("Area under the curve : %f" % roc_auc)
#Area is 0.7601

# tpr vs 1 - fpr
# plot tpr vs 1 - fpr
fig , ax = pl.subplots()
pl.plot(roc['tpr'] , color = 'red')
pl.plot(roc['1-fpr'] , color = 'blue')
pl.xlabel('1-False positive rate')
pl.ylabel('True positive rate')
pl.title('Receiver Operating characterstics')
ax.set_xticklabels([])

# the optimal cut off point is one where tpr is high and fpr is low
# the optimal cut off point is 0.317623
# so anything above this can be labeled as 1 else 0
# you can see from the output / chart that where TPR is Crossing 1-fpr 
# the tpr is 63%
# fpr is 36% and tpr - (1- fpr) is nearest to zero
# in the current example

#filling all the cells with zeros
c1['pred'] = np.zeros(1340)
c1.loc[pred>optimal_threshold , "pred"] = 1
#let us check the classification report
classification = classification_report(c1["pred"], c1["ATTORNEY"])
classification

#Splitting the data into train and test
train_data , test_data = train_test_split(c1 , test_size = 0.3)
#Model building
model = sm.logit('ATTORNEY ~CLMAGE + LOSS + CLMINSURANCE + SEATBELT' , data = train_data).fit()
model.summary()

# p values are below the condition of 0.05
# but SEATBELT has got statstically insignificant
model.summary2()
#AIC value is 1110.3782 , AIC score are useful in comparision with other
# lower the AIC score , Better the model

# let us go for the predictions
test_pred = logit_model.predict(test_data)
#creating new columns for storing predicted class of ATTORNEY
test_data['test_pred'] = np.zeros(402)

test_data.loc[test_pred>optimal_threshold,"test_pred"]=1

#Confusion Matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.ATTORNEY)
confusion_matrix
accuracy_test=(143+151)/(402)#Add current Values
accuracy_test

#Classification report
classification_test=classification_report(test_data["test_pred"],test_data["ATTORNEY"])
classification_test

#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(test_data["ATTORNEY"],test_pred)

#plot ROC Curve
plt.plot(fpr,tpr);plt.xlabel("False Positive Rate");plt.ylabel("True Positive Rate")

#AUC
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test

#prediction on train data
train_pred=logit_model.predict(train_data)
#Creating new column for storing predicted class of ATTORNEY
train_data.loc[train_pred>optimal_threshold,"train_pred"]=1
#confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.ATTORNEY)
confusion_matrix
accuracy_train=(315+347)/(938)
accuracy_train
#0.072174, this is going to  change with everytime when you 

#Classifcation report
classification_train=classification_report(train_data["train_pred"],
                                           train_data["ATTORNEY"])
classification_train
#Accuracy=0.69

#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(train_data["ATTORNEY"],train_pred)

#plotROC Curve
plt.plot(fpr,tpr);plt.xlabel("False Positive Rate");plt.ylabel("True Psitive Rate")

#AUC
roc_auc_train=metrics.auc(fpr,tpr)
roc_auc_train
















