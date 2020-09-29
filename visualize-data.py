import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv('history.csv')

min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['Size(#rooms)','Type','Location']
x = data.loc[:,column_sels]
y = data['Price']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()

for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()