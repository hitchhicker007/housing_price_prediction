import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

df = pd.read_csv('history.csv')
dataset = df.values

#Split the data set 
X = dataset[:,1:4]
Y = dataset[:,4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

with open("best-model.pkl", 'rb') as file:  
    model = pickle.load(file)

y_pred = model.predict(X_test[:][50].reshape(1,-1))
print(y_pred)
print(Y_test[50])