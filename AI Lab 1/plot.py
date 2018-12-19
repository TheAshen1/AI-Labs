#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('iris.data', header=None)

print(data);
data.tail()

y = data.iloc[50:150, 4].values

y = np.where(y == 'Iris-virginica', -1, 1)
X = data.iloc[50:150, [0, 2]].values 

plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='versicolor')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='virginica')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

perceptron = Perceptron(eta=0.1, numberOfIterations=15)
perceptron.fit(X, y)
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

perceptron.predict(X)