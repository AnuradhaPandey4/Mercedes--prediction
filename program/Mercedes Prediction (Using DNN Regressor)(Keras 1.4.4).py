#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import tensorflow as tf

import os


# In[2]:


train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')


# In[3]:


num_train = len(train_df)
y_train_init = train_df['y']
id_test = test_df['ID']


# In[4]:


train_df.drop(['ID', 'y'], axis=1, inplace=True)
test_df.drop(['ID'],axis=1, inplace=True)


# In[5]:


train_df.head()


# In[6]:


test_df.head()


# In[7]:


test_df.isnull().values.any()


# In[8]:


train_df.isnull().values.any()


# In[9]:


df_all = pd.concat([train_df, test_df])


# In[10]:


df_all.head()


# In[11]:


df_all = pd.get_dummies(df_all, drop_first=True)


# In[12]:


df_all.head()


# In[13]:


feature_cols = [tf.feature_column.numeric_column(col) for col in df_all.columns]


# In[14]:


train_df.shape


# In[15]:


df_all.shape


# In[16]:


X_train_init=df_all.iloc[4209:,]


# In[17]:


X_train_init.shape


# In[18]:


X_test=df_all.iloc[4210:,]


# In[19]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(X_train_init,y_train_init,test_size=0.3)


# ## DNN Regressor from Tensorflow

# In[20]:


input_func= tf.estimator.inputs.pandas_input_fn(x=X_train.astype(np.float32), 
                                                y= y_train.astype(np.float32), 
                                                batch_size=32, 
                                                num_epochs=1000, 
                                                shuffle=True)


# In[21]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_val.astype(np.float32),
                                                      y=y_val.astype(np.float32), 
                                                      batch_size=32, 
                                                      num_epochs=1, 
                                                      shuffle=False)


# In[22]:


test_input_func = tf.estimator.inputs.pandas_input_fn(x= X_test,                                                   
                                                 batch_size=100, 
                                                 num_epochs=1, 
                                                 shuffle=False)


# In[23]:


opti = tf.train.AdamOptimizer(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')


# In[24]:


estimator = tf.estimator.DNNRegressor(hidden_units=[128,256], feature_columns=feature_cols, optimizer=opti)


# In[25]:


estimator.train(input_fn=input_func,steps=200)


# In[26]:


result_eval = estimator.evaluate(input_fn=eval_input_func)


# In[27]:


result_eval


# In[28]:


predictions=[]
for pred in estimator.predict(input_fn=test_input_func):
    predictions.append(np.array(pred['predictions']).astype(float))


# In[29]:


predictions

