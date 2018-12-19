#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:13:18 2018

@author: user
"""

import numpy as np

class Perceptron(object):
    """Perceptron classifier
    Parameters
    ----------
    eta : float
        шаг обучения (между 0.0 and 1.0)
    n_iter : int
       Итерации для обучения на основе обучающей выборки.
    Attributes
    ----------
    w_ : 1d-array
        Веса после обучения.
    errors_ : list
        Ошибка на каждой эпохе.
    """

    def __init__(self, eta=0.01, numberOfIterations=10):
        self.eta = eta
        self.numberOfIterations = numberOfIterations

    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Обучающий вектор - вход, где n_samples количество примеров и
            n_features - количество свойств.
        y : array-like, shape = [n_samples]
            Выходное целевое значение.
        Returns
        -------
        self : object
        """
        self.weights = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.numberOfIterations):
            errors = 0
            for xi, target in zip(X, y):
                """ Используется простаяя формула расчета прирощения веса:
                 прирощение веса = шаг обучения * (целевое значение - выходное значение персептрона)
                 выходное значение = 1 если больше 0, -1 если меньше 0 и 0, если =0
                 
                """     
                update = self.eta * (target - self.predict(xi))
                
                self.weights[1:] += update * xi
                """ Новое значение веса = старое значение веса + прирощение веса * входное значение     
                """     
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        net_input = np.dot(X, self.weights[1:]) + self.weights[0]
        return net_input

    def predict(self, X):
        """Return class label after unit step"""
        prediction = np.where(self.net_input(X) >= 0.0, 1, -1)
        return prediction