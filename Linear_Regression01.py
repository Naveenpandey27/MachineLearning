#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


# Import the dataset

df = pd.read_csv(r'C:\Users\admin\Desktop\files\Salary_Data.csv')
df.head()


# In[16]:


X = df.iloc[:, 0].values
y = df.iloc[:,1].values
X = X.reshape(-1,1)
y = y.reshape(-1,1)


# In[17]:


#split the data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[18]:


#Feature scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[20]:


# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)


# In[22]:


# Predicting the Test set results

y_pred = reg.predict(X_test)


# In[24]:


# Visualising the Training set results

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[25]:


# Visualising the Test set results

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:




