#!/usr/bin/env python
# coding: utf-8

# In[30]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[31]:


#Reading training file
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
#Selecting numerical features
train_numerical = train.select_dtypes(include='int64')
y = train['SalePrice']
X = train_numerical.iloc[:,:-1]
list(X.columns)


# In[32]:


#Reading test file
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#Selecting numerical features
test_numerical = test[X.columns]
#Filling NA cells with mean values
test_numerical = test_numerical.fillna(test_numerical.mean())
test_numerical.shape


# In[33]:


#Modelling decision tree classifier
model = DecisionTreeClassifier()
model.fit(X, y)
#X = np.arange(0, 10, 0.1)[:, np.newaxis]
y_pred = model.predict(test_numerical)


# In[34]:


#Writing prediction values to CSV file
print(y_pred)
output_array = np.array(y_pred)
np.savetxt("sales_price_pred_decisiontree.csv", output_array, delimiter=",")

output_array = np.array(test_numerical)
np.savetxt("Id.csv", output_array, delimiter=",")

