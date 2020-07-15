import numpy as np 
import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

import os

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')

train_df.head()

test_df.head()

test_df.isnull().values.any()

train_df.isnull().values.any()

train_df.shape

dummy1=pd.get_dummies(train_df.iloc[:,2:10])
dummy1.shape

train_df.columns


train_df=train_df.drop(['ID','X0','X1','X2','X3','X4','X5','X6','X8'],axis=1)

train_df.head()


# In[12]:


train_df=pd.concat([dummy1,train_df],axis=1)

train_df.head()

dummy2=pd.get_dummies(test_df.iloc[:,1:9])
dummy2.shape

test_df=test_df.drop(['ID','X0','X1','X2','X3','X4','X5','X6','X8'],axis=1)

test_df.head()

test_df=pd.concat([dummy2,test_df],axis=1)

test_df.head()

from sklearn.model_selection import train_test_split

train=train_df.drop(['y'],axis=1)
test=train_df['y']

X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler().fit(X_train)
X_train=scaler1.transform(X_train)
X_test=scaler1.transform(X_test);

X_train.shape

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

history=NN_model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

test_df.head()
test_df.shape

train_df.shape

output=NN_model.predict(test_df.iloc[:,0:563])

print(output)





