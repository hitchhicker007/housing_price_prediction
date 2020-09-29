import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

# def main():
print("""
     _   _ _ _       _       _   _ _      _
    | | | (_) |_ ___| |__   | | | (_) ___| | _____ _ __
    | |_| | | __/ __| '_ \  | |_| | |/ __| |/ / _ \ '__|
    |  _  | | || (__| | | | |  _  | | (__|   <  __/ |
    |_| |_|_|\__\___|_| |_| |_| |_|_|\___|_|\_\___|_|  
""")

print("""
    ===================Housing Price Prediction (Boston)===================
""")

BHK = int(input("\n   Please enter BHK size : "))
TYPE = int(input(" \n   Please enter proporty type : "))
LOC = int(input("\n   Please enter proporty Location : "))

sample = [BHK,TYPE,LOC]
sample = np.array(sample).reshape(1,-1)
# print(sample.reshape(1,-1))

# min_max_scaler = preprocessing.MinMaxScaler()
# sample = min_max_scaler.fit_transform(sample)
with open('best-model.pkl', 'rb') as file:  
    model = pickle.load(file)

prediction = model.predict(sample)

print("\n> price of the house is : " + str(prediction))
    
# main()