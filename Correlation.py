#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("C:/Linear_Regression/DataSet/SampleSuperstore.csv")


# In[5]:


data.head()


# In[4]:


#correlation using numpy
np.corrcoef(data['Sales'] , data['Profit'])


# # Correlation using scipy

# In[6]:


st.pearsonr(data['Sales'] , data['Profit'])


# In[8]:


sns.scatterplot(x = data['Sales'] , y = data['Profit'])


# the above scatter diagram show moderately positive correaltion is exist between sales and profit features of the dataframe

# In[9]:


sns.pairplot(data)


# In[10]:


data.corr()


# In[11]:


sns.heatmap(data.corr())


# In[16]:


plt.figure(figsize = (15,8))
plt.subplot(231)
sns.scatterplot(x = data['Sales'] , y = data['Profit'])
plt.subplot(232)
sns.scatterplot(x = data['Discount'] , y = data['Profit'])
plt.subplot(233)
sns.scatterplot(x = data['Quantity'] , y = data['Profit'])
plt.subplot(234)
sns.scatterplot(x = data['Discount'] , y = data['Sales'])
plt.subplot(235)
sns.scatterplot(x = data['Discount'] , y = data['Qunatity'])


# In[19]:


#Spearman r
from scipy.stats import spearmanr


# In[20]:


spearmanr(data['Sales'] , data['Profit'])


# In[21]:


st.pearsonr(data['Sales'] , data['Profit'])


# In[ ]:




