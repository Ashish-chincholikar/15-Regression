# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:36:09 2024

@author: Ashish Chincholikar
Linear regression - 1. gradient descent
2. Ordinary least square regression (OLS)


AT is y 
waist is x

"""
import pandas as pd
import numpy as np
import seaborn as sns

wcat = pd.read_csv("C:/Linear_Regression/DataSet/wc-at.csv")
#EDA 
#1. Measures the central tendency
#2. Measures of Dispersion
#3. Third Moment Business Decision
#4. Fourth Moment Business Decision
wcat.info()
wcat.describe()
#Graphical representation
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT ,x = np.arange(1,110,1))
plt.hist(wcat.AT)
plt.boxplot(wcat.AT)
#Data is right skewed
sns.distplot(wcat.AT)
sns.distplot(wcat.Waist)

#Scatter plot
plt.scatter(x = wcat['Waist'] , y=wcat['AT'] , color = 'green')
#direction : positive , linearity : moderate , strength : poor
#Now let us calculate correlation coeficient
np.corrcoef(wcat.Waist, wcat.AT)
#0.81<0.85 moderately coorelated

#Let us check the direction using covar factor
cov_output = np.cov(wcat.Waist, wcat.AT)[0,1]
cov_output
#it is positive means correlation will be positve


#now let us apply to linear regression model
import statsmodels.formula.api as smf
#ALl machine learning algorithms are implemented using sklearn, 
#but for this statsmodel packaage is begin used because it gives you u
#backend calculations of bita-0 and bita-1
model = smf.ols('AT~Waist' , data = wcat).fit()
model.summary()
#OLS helps to find the best fit model , which causes least square error
#First you check R squared value=0.670 , if R square = 0.8 means that 
#model is best fit
#if R-square =0.8 to 0.6 moderate correlation
#Next you check P>|t| =0 , it means less than aplha, alpa is 0.05
#Hence the model is accepted

#Regression line
pred1 = model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist , wcat.AT)
plt.plot(wcat.Waist , pred1 , "r")
plt.legend(['observed data' , 'predicated line'])
plt.show()

#error calculations
res1 = wcat.AT-pred1
np.mean(res1)
#it must be zero and hence it 10^-14 = ~0
res_sqr1 = res1*res1
msel = np.mean(res_sqr1)
rmsel = np.sqrt(msel)
rmsel
#32.76 lesser the value better the model
#How to improve this model ,  transformation of
plt.scatter(x = np.log(wcat['Waist']) , y = wcat['AT'] , color = 'brown')
#Data is lineraly scatter , direction is postive , strength poor
#let us check the coefficient of correlation
np.corrcoef(np.log(wcat.Waist) , wcat.AT)
#r value is 0.82<0.85 hence moderately correlation
model2 = smf.ols('AT~np.log(Waist)' , data = wcat).fit()
model2.summary()
#R-squared:                       0.675<0.85 hence there is scope of improvement
#Intercept     -1328.3420     95.923    -13.848      0.000   -1518.498   -1138.186
#Again check the R-square value = 0.67 which is less than 0.8
#P value is 0 less than 0.05

"""
R_square < 
p value
bita-0
bita-1 """

pred2 = model2.predict(pd.DataFrame(wcat['Waist']))
#check wcat and pred2 from variable explorer
#scatter diagram
plt.scatter(np.log(wcat.Waist) , wcat.AT)
plt.plot(np.log(wcat.Waist) , pred2 , "r")
plt.legend(['observed data' , 'predicated line'])

#error calculation
res2 = wcat.AT-pred2
res_sqr2 = res2*res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
#32.49

#there is no considerable changes
#Now let us change y value instead of x
#Now let us make logY and X as is
plt.scatter(x = wcat['Waist'] , y=np.log(wcat['AT']) ,color = 'orange')
#Data is lineraly scattered, direction positive,strength poor
np.corrcoef(wcat.Waist, np.log(wcat.AT))
#r = 0.84
#r value is 0.84<0.85 hence moderately linearity

model3 = smf.ols('np.log(AT)~Waist' , data = wcat).fit()
model3.summary()
"""
R_square < 
p value
bita-0
bita-1 """

#Again check the R-Square value = 0.707 which is less than 0.8 
#p value is 0.02 less than 0.05


pred3 = model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at = np.exp(pred3)
pred3_at
#check wcat and pred3_at from variable explorer
#scatter diagram , regression line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred3 , "r")
plt.legend(['observed data' , 'predicated line'])
plt.show()

#error calcualation
res3 = wcat.AT-pred3_at
res_sqr3 = res3*res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3
#Rmsel is 38.53
#there is no significant change in rmse r=0.8409 , R-squared=0.707 
#let us try another model

#polynomial transformation
#x=Waist,X^2=Waist*Wasit , y=log(at)
#when using polynomial tranformation , we cannot calculate the correlation cofficient
model4 = smf.ols('np.log(AT)~Waist+I(Waist*Waist)' , data=wcat).fit()
model4.summary()
#R-squared = 0.779<0.85 , there is scope of improvement
#p=0.000<0.05 hence acceptable
#bita-0 = -7.8241(Intercept)
#bita-1 = 0.2289(Waist)

pred4 = model4.predict(pd.DataFrame(wcat['Waist']))
pred4 
pred4_at = np.exp(pred4)
pred4_at

#Regression line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist , pred4 , 'r')
plt.legend(['Observed data_model3 ' , 'Predicted line'])
plt.show()

#Error Calculations
res4 = wcat.AT-pred4_at
res_sqr4 = res4*res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4
#32.244
#Among all the models , model4 is the best 
#--------------------------------------------
data = {"model" : pd.Series(['SLR' , "log_model" , 'Exp_model' , "Poly_model"])}
data
table_rmse = pd.DataFrame(data)
table_rmse
#---------------------------------------------
#We have to generalize the best model
from sklearn.model_selection import train_test_split
train, test = train_test_split(wcat, test_size = 0.2)
plt.scatter(train.Waist , np.log(train.AT))
plt.scatter(test.Waist , np.log(test.AT))
final_model = smf.ols('np.log(AT)~Waist+I(Waist*Waist)' , data=wcat).fit()
#Y is log(AT) and X = Waist
final_model.summary()
#R-squared = 0.779 <0.85 , there is scope of improvement
#p = 0.000<0.5 hence acceptable
#bita-0 = -7.8241
#bita-1 = 0.2289
test_pred = final_model.predict(pd.DataFrame(test))

test_pred_at = np.exp(test_pred)
test_pred_at
#--------------------------------------------
train_pred = final_model.predict(pd.DataFrame(train))

train_pred_at = np.exp(train_pred)
train_pred_at
#--------------------------------------------
#Evaluation on test data
test_err = test.AT-test_pred_at
test_sqr = test_err*test_err
test_mse = np.mean(test_sqr)
test_rmse = np.sqrt(test_mse)
test_mse

#Evaluation on train data
train_res = train.AT-train_pred_at
train_sqr = train_res*train_res
train_mse = np.mean(train_sqr)
train_rmse = np.sqrt(train_mse)
train_rmse
#RMSE =  34.44
#---------------------------------------------
#test_rmse > train_rmse
#The model is overfit


