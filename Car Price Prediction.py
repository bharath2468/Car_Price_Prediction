#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


df=pd.read_csv("C:/Users/Bharath/Downloads/CarPrice_Assignment.csv")


# In[18]:


df.info()


# In[19]:


df.head()


# In[20]:


df.describe()


# In[21]:


print(df.corr())


# In[26]:


plt.figure(figsize=(20, 15))
correlations = df.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# In[32]:


predict = "price"
data = df[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
accuracy=model.score(xtest, predictions)*100
print("The accuracy of Car Price Prediction is {:.2f}".format(accuracy))

