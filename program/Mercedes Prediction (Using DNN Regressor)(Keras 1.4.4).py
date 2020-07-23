import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import tensorflow as tf

import os

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')

num_train = len(train_df)
y_train_init = train_df['y']
id_test = test_df['ID']

train_df.drop(['ID', 'y'], axis=1, inplace=True)
test_df.drop(['ID'],axis=1, inplace=True)

train_df.head()
test_df.head()

test_df.isnull().values.any()
train_df.isnull().values.any()
df_all = pd.concat([train_df, test_df])
df_all.head()

df_all = pd.get_dummies(df_all, drop_first=True)

df_all.head()

feature_cols = [tf.feature_column.numeric_column(col) for col in df_all.columns]

train_df.shape

df_all.shape

X_train_init=df_all.iloc[4209:,]

X_train_init.shape

X_test=df_all.iloc[4210:,]

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(X_train_init,y_train_init,test_size=0.3)


# ## DNN Regressor from Tensorflow

input_func= tf.estimator.inputs.pandas_input_fn(x=X_train.astype(np.float32), 
                                                y= y_train.astype(np.float32), 
                                                batch_size=32, 
                                                num_epochs=1000, 
                                                shuffle=True)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_val.astype(np.float32),
                                                      y=y_val.astype(np.float32), 
                                                      batch_size=32, 
                                                      num_epochs=1, 
                                                      shuffle=False)

test_input_func = tf.estimator.inputs.pandas_input_fn(x= X_test,                                                   
                                                 batch_size=100, 
                                                 num_epochs=1, 
                                                 shuffle=False)

opti = tf.train.AdamOptimizer(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

estimator = tf.estimator.DNNRegressor(hidden_units=[128,256], feature_columns=feature_cols, optimizer=opti)

estimator.train(input_fn=input_func,steps=200)

result_eval = estimator.evaluate(input_fn=eval_input_func)

result_eval

predictions=[]
for pred in estimator.predict(input_fn=test_input_func):
    predictions.append(np.array(pred['predictions']).astype(float))
    
predictions

