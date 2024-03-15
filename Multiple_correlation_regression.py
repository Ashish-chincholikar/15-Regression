# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:59:47 2024

@author: Ashish Chincholikar
Multiple Correlation Regression Analysis

"""

import pandas as pd
import numpy as np
import seaborn as sns

cars = pd.read_csv("C:/Linear_Regression/Multiple_Correlation_Regression/Dataset/Cars.csv")
#Exploratory data analysis
#1. Measures of Centeral Tendency
#2. Measures of dispersion
#3. Third Moment Business decision 
#4. Fourth moment Business decision
#5. probablity distribution
#6. Graphical representation(Histogram , Boxplot)
cars.describe()

#Graphical Representation
import matplotlib.pyplot as plt
plt.bar(height = cars.HP , x = np.arange(1,82,1))
sns.displot(cars.HP)
# data is rightly skewed
plt.boxplot(cars.HP)
#There are several outliers in HP columns
#Similar operations are expected from other 3 columns
sns.displot(cars.MPG)
#data is slightly left distributed
plt.boxplot(cars.MPG)
#there are no outliers
sns.displot(cars.VOL)
sns.distplot(cars.SP)
#data is slightly right distriubted
plt.boxplot(cars.SP)
#there are several outliers
sns.distplot(cars.WT)
plt.boxplot(cars.WT)
#there are serveral outliers
#NOw let us plot joint plot, joint plot is to show scatter plot and histogram
import seaborn as sns
sns.jointplot(x = cars['HP'] , y = cars['MPG'])

#now let us plot count plot
plt.figure(1 , figsize=(16,10))
sns.countplot(cars['HP'])
#count plot shows how many times the each value occurence 
#92 HP value occured 7 times

##QQ plots
from scipy import stats
import pylab
stats.probplot(cars.MPG , dist="norm" , plot=pylab)
#MPG data is normally distributed
# There are 10 scatter plots need to be plotted, one by one 
#to plot , so we can use pair plots

import seaborn as sns
sns.pairplot(cars.iloc[:,  :])
#write linearity: direction: and strength: for each 
#Data is lineraly scattered, direction positive,strength poor
#you can check the collinerity problem between the input 
#you can check the plot between SP and HP , they are strongly correlated
#same way you can check WT and VOL , it is also strongly correlated

#Now let us check r value between variables
cars.corr()
#you can check SP and HP , r value is 0.97 and same way
#you can check WT and VOl , it has got 0.99 which is greater

#Now although we observed strongly correlated pairs , still 
#linear regression
import statsmodels.formula.api as smf
ml1 = smf.ols('MPG~WT+VOL+SP+HP' , data = cars).fit()
ml1.summary()
#R square value is obsreved is 0.771 < 0.85
#p-value of WT and VOL is 0.814 and 0.556 which is very high
#it means it is greater than 0.05 , WT and VOL columns
#we need to ignore
#or delete. Instead deleting 81 entreies,
#let us check row wise outliers
#identifying is there any influenctial value
#to check you can use influential index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
#76 is the value which has got outliers
#go to data frame and check 76 th entry
#let us delete that entry
cars_new = cars.drop(cars.index[[76]])

#again apply regression to cars_new
ml_new = smf.ols('MPG~WT+VOL+HP+SP' , data=cars_new).fit()
ml_new.summary()
#R-squared value is 0.819 but p values are same , hence not 
#now next option is delete the coulumns but
#question is which columns is to be deleted
#we have already checked correlation factor r
#VOL has got -0.529 and for WT = -0.526
#WT is less hence can be deleted

#another approch is to check the collinearity , rsquare is giving 
#that values
#we will have to apply  regression w.r.t x1 and input
#x2 , x3 and x4 so on and soforth
rsq_hp = smf.ols('HP~WT+VOL+SP' , data=cars_new).fit().rsquared
vif_hp = 1/(1-rsq_hp)
vif_hp

#VIF is variance influential factor , calculating VIF helps
#of x1 ,x2 , x3 and x4
rsq_wt = smf.ols('WT~HP+VOL+SP' , data=cars).fit().rsquared
vif_wt = 1/(1-rsq_wt)

rsq_vol = smf.ols('VOL~HP+WT+SP' , data=cars).fit().rsquared
vif_vol = 1/(1-rsq_vol)

rsq_sp = smf.ols('SP~HP+WT+VOL' , data=cars).fit().rsquared
vif_sp = 1/(1-rsq_sp)

#vif_wt = 639.53 and vif_vol  = 638.80 hence vif_wt
#is greater , thumb rule is vif should not be greater han 

#storing the values in dataframe
d1={'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame=pd.DataFrame(d1)
vif_frame

# let us drop WT and apply correlation to remailing three
final_ml = smf.ols('MPG~VOL+SP+HP' , data=cars).fit()
final_ml.summary()
#R square is 0.770 and P values 0.00 , 0.012 <0.05

#prediction
pred = final_ml.predict(cars)

##QQplot
res = final_ml.resid
sm.qqplot(res)
plt.show()

#This QQ plot is on residual which is obtained on training 
#errors are obtained on test data
stats.probplot(res, dist="norm" , plot = pylab)
plt.show()

#let us plot the residual plot , which takes the residuals 
#and the data
sns.residplot(x = pred , y=cars.MPG, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

#splitting the data into train and test data
from sklearn.model_selection import train_test_split
cars_train , cars_test = train_test_split(cars, test_size=0.2)
#preparing the model on train data
model_train = smf.ols('MPG~VOL+SP+HP' , data=cars_train).fit()
model_train.summary()
test_pred = model_train.predict(cars_test)
#test_errors
test_error =test_pred-cars_test.MPG
test_rmse = np.sqrt(np.mean(test_error*test_error))
test_rmse










