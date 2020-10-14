from __future__ import absolute_import, division, print_function, unicode_literals
import sys

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
import tensorflow_docs as tfdoc
import tensorflow_docs.plots
'''

#dataset_path = keras.utils.get_file("housing.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data")
column_names = ['CRIM', 'ZN', 'INDUS','CHAS','NOX',
                'RM', 'AGE', 'DIS','RAD','TAX','PTRATION', 'B', 'LSTAT', 'MEDV']
raw_dataset = pd.read_csv('/home/smg/swork/PycharmProjects/housing.data', names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset.tail(n=10)

p=0.8
trainDataset = dataset.sample(frac=p, random_state=0)
testDataset = dataset.drop(trainDataset.index)

fig, ax = plt.subplots()

x = trainDataset['RM']
y = trainDataset['MEDV']
ax.scatter(x, y, edgecolors=(0, 0, 0))
ax.set_xlabel('RM')
ax.set_ylabel('MEDV')
plt.show()

if __name__ == '__main__':
    print("Linear Regression");