#! /usr/bin/env/ python3
# -*- coding: utf-8 -*-

import numpy as np


class NeuralNetwork:

    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        # Arch 2-4-4-1
        self.__neural_arch = np.array([3, 5, 3, 1])
        self.__neural_len = len(self.__neural_arch)

        # array of weights
        self.__W = []
        for i in range(self.__neural_len - 1):
            self.__W.append(np.random.randn(self.__neural_arch[i], self.__neural_arch[i + 1]) * 0.1)

        self.__sidmoid = []
        self.__d_sigmoid = []
        for i in range(self.__neural_len - 1):
            self.__sidmoid.append(lambda x: 1 / (1 + np.exp(-x)))
            self.__d_sigmoid.append(lambda y: y - np.square(y))

        self.__sidmoid.append(lambda x: x)
        self.__d_sigmoid.append(lambda x: np.ones(x.shape))

        # sigmoid vector
        self.__sigmoid_mapper = np.vectorize(self.__sigmoid_func)
        # learning rate
        self.__learning_rate = learning_rate
		
	# activate func to predict
    def __sigmoid_func(self, x):
        return 1 / (1 + np.exp(-x))
		
	# network prediction list
    def predict(self, inputs):
        result = []
        for input in inputs:
            data = input
            for i in range(self.__neural_len - 1):
                data = self.__sigmoid_mapper(np.dot(data, self.__W[i]))
            result.append(data)
        return result
		
	# network prediction 
    def predict_once(self, input_):
        data = input_
        for i in range(self.__neural_len - 1):
            data = self.__sigmoid_mapper(np.dot(data, self.__W[i]))
        return data
		
	# train network
    def train(self, data, classes, out, gradient, bias):
        for i in range(len(data)):
            t = classes[i, :]
            out[0] = data[i, :]
            for j in range(self.__neural_len - 1):
                out[j + 1] = self.__sidmoid[j](np.dot(out[j], self.__W[j]) + bias[j])

            gradient[-1] = np.multiply((t - out[-1]), self.__d_sigmoid[-1](out[-1]))

            for j in range(self.__neural_len - 2, 0, -1):
                gradient[j - 1] = np.multiply(np.dot(gradient[j], self.__W[j].T), self.__d_sigmoid[j](out[j]))

            for j in range(self.__neural_len - 1):
                self.__W[j] = self.__W[j] + self.__learning_rate * np.outer(out[j], gradient[j])
                bias[j] = bias[j] + self.__learning_rate * gradient[j]
				
	# return network arch
    def return_arch(self):
        return self.__neural_arch

    def return_weigh(self):
        return self.__W