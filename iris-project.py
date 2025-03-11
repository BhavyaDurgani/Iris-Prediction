import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

i_df = pd.read_csv('iris.csv')

i_df.head()

i_df.tail()

i_df.isnull().sum()

i_df.describe()

i_df.info()

i_df['Species'].value_counts()

i_df = i_df.replace({'Species':{'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}})

print(i_df)

i_df.info()

x = i_df.drop(columns='Species',axis=1)

y = i_df['Species']

print(x)

print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

print(x.shape,x_train.shape,x_test.shape)

model = LogisticRegression()

model.fit(x_train,y_train)

x_train_pred = model.predict(x_train)

train_data_acc = accuracy_score(x_train_pred,y_train)

print(train_data_acc)

x_test_pred = model.predict(x_test)

test_data_acc = accuracy_score(x_test_pred,y_test)

print(test_data_acc)

input_data = (1,5.1,3.5,1.4,0.2)

input_data_np = np.asarray(input_data)

input_reshape = input_data_np.reshape(1,-1)

pred = model.predict(input_reshape)

print(pred)

if pred[0] == 0:
  print('Species : Iris-setosa')
elif pred[0] == 1:
  print('Species : Iris-versicolor')
else:
  print('Species : Iris-virginica')

import pickle

filename = 'iris_model.sav'

pickle.dump(model,open(filename,'wb'))

loaded_model = pickle.load(open(filename,'rb'))

input_data = (1,5.1,3.5,1.4,0.2)

input_data_np = np.asarray(input_data)

input_reshape = input_data_np.reshape(1,-1)

pred = loaded_model.predict(input_reshape)

print(pred)

if pred[0] == 0:
  print('Species : Iris-setosa')
elif pred[0] == 1:
  print('Species : Iris-versicolor')
else:
  print('Species : Iris-virginica')