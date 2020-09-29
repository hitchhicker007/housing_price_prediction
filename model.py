import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from sklearn.metrics import accuracy_score
import os
from pathlib import Path
import shutil
from sklearn import preprocessing

print("\n ------------  loading data  ------------")

df = pd.read_csv('history.csv')

# print(df.head(7))

dataset = df.values
# print(dataset)

#Split the data set 
X = dataset[:,1:4]
Y = dataset[:,4]

# print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

print("\n ------------  training model  ------------")

model = linear_model.LinearRegression()
MSE = 0
MSE_list = []
print("\n ------------  Evaluating model  ------------")
# min_max_scaler = preprocessing.MinMaxScaler()
# X_scale = min_max_scaler.fit_transform(X)

for i in range(0,100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    MSE = accuracy_score(Y_test,y_pred)
    MSE_list.append(MSE)
    print("\n"+str(i)+". MSE of model is : "+str(MSE))

    filename = str(i)+'-model-bengluru-'+str(MSE)+'.pkl'
    with open('models/'+filename, 'wb') as file:  
        pickle.dump(model, file)

print("\n ------------  Getting the Best Model  ------------")

MSE_list.sort()

path = Path('models/')

for model in path.glob("*.pkl"):
    model_file_name = str(model).split("\\")[-1]
    if(str(MSE_list[0]) in model_file_name):
        print("\n The best model is "+model_file_name+" With MSE "+str(MSE_list[0]))
        shutil.move("models/"+model_file_name,"./")
        shutil.rmtree('models')
        os.rename(model_file_name,"best-model.pkl")

print("\n ------------  done  ------------")