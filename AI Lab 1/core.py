#! /usr/bin/env/ python3
# -*- coding: utf-8 -*-

from sys import exit, stdout
from net import NeuralNetwork
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Core:

    def __init__(self, file_path):
        self.__file_path = file_path
        self.__network = None
        self.__train_data = []
        self.__train_class = []
        self.__test_data = []
        self.__test_class = []
        self.__outs = []
        self.__gradients = []
        self.__bias = []
        self.__learning_rate = 0.023
        self.__network = NeuralNetwork(self.__learning_rate)

        for i in range(len(self.__network.return_arch())):
            self.__outs.append(np.zeros([self.__network.return_arch()[i]]))

        for i in range(1, len(self.__network.return_arch())):
            self.__gradients.append(np.zeros(self.__network.return_arch()[i]))

        for i in range(1, len(self.__network.return_arch())):
            self.__bias.append(np.random.randn(self.__network.return_arch()[i]) * 0.1)

        if self.__check_file():
            data_frame = pd.read_csv(self.__file_path, header=None)[:100]
            data_frame = data_frame.sample(frac=1).reset_index(drop=True)
            classes = data_frame[4]
            clss = data_frame[4][0]
            y = data_frame.iloc[:80, 0:3].values
            X = data_frame.iloc[80:, 0:3].values

            for row, cls in zip(y, classes[:80]):
                self.__train_data.append(list(row))
                self.__train_class.append([int(cls == clss)])

            self.__data = self.__train_data
            self.__classes = self.__train_class
            self.__train_data = np.matrix(self.__train_data)
            self.__train_class = np.matrix(self.__train_class)

            for row, cls in zip(X, classes[80:]):
                self.__test_data.append(list(row))
                self.__test_class.append(int(cls == clss))

            # self.__network = NeuralNetwork(0.1)
            self.__start(250, data_frame[80:])
        else:
            print("TypeError; file must have .csv type")
            exit(1)

    # check type of file
    def __check_file(self):
        try:
            if self.__file_path.split('.')[1] == "csv":
                return 1
            else:
                return 0
        except IndexError:
            print("Error: Name of file must be whole or file doesn`t have type")
            exit(1)


    def __MSE(self, y, Y):
        return np.mean((y - Y) ** 2)

    def __start(self, epochs, test_frame):
        print("Train:")
        errors_ = []
        for e in range(epochs):
            self.__network.train(self.__train_data, self.__train_class, self.__outs, self.__gradients, self.__bias)
            train_loss = self.__MSE(self.__network.predict(np.array(self.__data)), np.array(self.__classes[:80]))
            errors_.append(train_loss)
            stdout.write("\rProgress: {}, Train loss: {}".format(str(100 * e / float(epochs))[:4], str(train_loss)[:5]))

        # error graph

        plt.plot(range(1, len(errors_) + 1), errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of misclassifications')
        plt.show()

        print(self.__network.return_weigh())

        print('\n\n')
        print("Test data")
        print(test_frame)

        print('\n\nResult:')
        for input_stat, correct_predict in zip(self.__test_data, self.__test_class):
            print("For inputs: {} predict is: ({}) {}, expected: {}".format(
                str(np.array(input_stat)),
                str(self.__network.predict_once(np.array(input_stat))),
				str(self.__network.predict_once(np.array(input_stat)) > .5),
                str(correct_predict == 1)
            ))
		
        print('\n\nPregictions:')
        while True:
            print('Write your dots, i predict their class:')
            arg1 = float(input("First arg: "))
            arg2 = float(input("Second arg: "))
            arg3 = float(input("Third arg: "))

            result = self.__network.predict_once([arg1, arg2, arg3])
            print("Result: {1}({0})\n\n".format(result, result > .5))
