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

import os


# In[2]:


train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')


# In[3]:


train_df.head()


# In[4]:


test_df.head()


# In[5]:


test_df.isnull().values.any()


# In[6]:


train_df.isnull().values.any()


# In[7]:


train_df.shape


# In[8]:


dummy1=pd.get_dummies(train_df.iloc[:,2:10])
dummy1.shape


# In[9]:


train_df.columns


# In[10]:


train_df=train_df.drop(['ID','X0','X1','X2','X3','X4','X5','X6','X8'],axis=1)


# In[11]:


train_df.head()


# In[12]:


train_df=pd.concat([dummy1,train_df],axis=1)


# In[13]:


train_df.head()


# In[14]:


dummy2=pd.get_dummies(test_df.iloc[:,1:9])
dummy2.shape


# In[15]:


test_df=test_df.drop(['ID','X0','X1','X2','X3','X4','X5','X6','X8'],axis=1)


# In[16]:


test_df.head()


# In[17]:


test_df=pd.concat([dummy2,test_df],axis=1)


# In[18]:


test_df.head()


# In[19]:


from sklearn.model_selection import train_test_split

train=train_df.drop(['y'],axis=1)
test=train_df['y']


# In[20]:


X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)


# In[21]:


from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler().fit(X_train)
X_train=scaler1.transform(X_train)
X_test=scaler1.transform(X_test);


# In[22]:


X_train.shape


# In[23]:


from keras.models import Sequential
from keras.layers import Dense
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128,kernel_initializer='normal',input_dim =X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256,activation='relu'))
NN_model.add(Dense(256,activation='relu'))

# The Output Layer :
NN_model.add(Dense(1,activation='linear'))

# Compile the network :
NN_model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])


# In[24]:


history=NN_model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))


# In[25]:


print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[26]:


test_df.head()


# In[27]:


test_df.shape


# In[28]:


train_df.shape


# In[32]:


output=NN_model.predict(test_df.iloc[:,0:563])


# In[33]:


print(output)


# In[ ]:





# In[ ]:




