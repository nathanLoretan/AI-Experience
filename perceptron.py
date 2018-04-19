# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

import pickle
from math import *
import numpy as np
import matplotlib.pyplot as plt
from generate_dataset import display, colorlist, markerlist

# ------------------------------------------------------------------------------
class Input:
    """ """

    # Constructor
    def __init__(self, _x):
        self.x = _x

    def __mul__(self, _w=0):
        return self.x * _w

class Perceptron:
    """ """

    # x = input
    # y = ouput
    # f = activation function
    # w = weights
    # t = training data
    # b = bias
    # a = alpha, smoothing factor
    # n = learning factor
    # dw = delta weight
    # db = delta bias

    # General attributes
    a   = 0 # 0.9
    n   = 0.5
    b   = 0
    out = 0

    dw = 0
    db = 0

    # Constructor
    def __init__(self, _x):

        self.x = _x
        self.y = np.zeros(len(self.x))

        # Need to assign rand weight to train a network with hidden layers
        self.w = np.random.uniform(low=-1, high=1, size=len(self.x))

        # Used to plot the evolution of the weights of the neurone
        self.w_buffer = [np.zeros(len(self.x))]

    def __mul__(self, _w=0):
        return self.out * _w

    def plot_weight(self, _l, _n):
        """ Plot the evolution of the weight value of the neurone over the
        training set."""

        plt.figure("Weights, neurone " + str(_l) + "-" + str(_n))
        plt.grid(True)
        plt.xlabel('set', fontsize=16)
        plt.ylabel('w',   fontsize=16)
        plt.plot(np.arange(len(self.w_buffer)), self.w_buffer,"-")
        plt.show()

    def activation(self, _y):

        # Sigmoid, returns value between 0 and 1
        return (1.0 / (1.0 + exp(-_y)))

    def delta(self, _y, _t, _a, _x, _dw):

        # Use the derivate of square loss error and smoothing weight adjustement
        return np.add(_a * _dw,
                      np.multiply(_x, (1.0 - _a) * _y * (_t - _y) * (1.0 -_y)))

    def delta_bk(self, _y, _t, _a, _x, _dw):

        # Use the derivate of square loss error and smoothing weight adjustement
        return np.add(_a * _dw,
                      np.multiply(_x, (1.0 - _a) * _y * _t * (1.0 -_y)))


    def backpropagation(self, _dw, _db):

        self.dw = self.delta_bk(self.out, _dw, self.a, self.x, self.dw)
        self.db = self.delta_bk(self.out, _db, self.a, 1,      self.db)

        # wNew = wOld + a*dw*x
        self.w = self.w + self.n * self.dw

        # bNew = bOld + a*db*1
        self.b = self.b + self.n * self.db

        # Add the new value of weights to plot their evolution
        self.w_buffer.append(self.w)

    def train(self, _t):


        if _t != self.out:

            self.dw = self.delta(self.out, _t, self.a, self.x, self.dw)
            self.db = self.delta(self.out, _t, self.a, 1,      self.db)

            # wNew = wOld + a*dw*x
            self.w = self.w + self.n * self.dw

            # bNew = bOld + a*db*1
            self.b = self.b + self.n * self.db

            # Add the new value of weights to plot their evolution
            self.w_buffer.append(self.w)

    def run(self):

        # Adder
        self.y = self.b + np.sum(self.x * self.w)

        # Activation function
        self.out = self.activation(self.y)

        return self.out

class Neural_network:
    """ """

    # x = input
    # y = ouput
    # t = training dataset
    # l = [nbr of nodes of layer 1, nbr of nodes of layer 2, ...]
    # f = activation function

    # Constructor
    def __init__(self, _nbr_param, _l):

        # List of the layers (input=0, hidden, output=last one)
        self.layers = list()

        for i in range(len(_l)+1):

            self.layers.append(list())

            if i == 0:

                # The nodes of the input layer are directly connected to a
                # parameter
                for y in range(_nbr_param):
                    self.layers[i].append(Input(0))

            else:

                # Create the nodes of the layers and connect them to all the
                # other from the previous layer
                for y in range(_l[i-1]):
                    self.layers[i].append(Perceptron(self.layers[i-1]))

        # Creat the array for the output values
        self.y = np.zeros(_l[len(_l)-1])

    def getNeuron(self, _l, _n):

        # _l+1 because first layer has only the input parameters
        return self.layers[_l+1][_n]

    def plot_weights(self, _l, _n):
        """ Plot the evolution of the weight value of the neurone over the
        training set."""

        # _l+1 because first layer has only the input parameters
        self.layers[_l+1][_n].plot_weight(_l, _n)

    def plot(self, _l, _n, _disx, _disy, _dataset, _cat):
        """Plot the output of a specific neurone of the neural networkself.
        /!\ It is only implemented with 2 parameters NN"""

        # /!\ _l+1 because first layer has only the input parameters

        # Result for graphical display
        plt.figure("neurone " + str(_l) + "-" + str(_n))

        r = 0
        x = [np.linspace(_disx[0], _disx[1], 30),
             np.linspace(_disy[0], _disy[1], 30)]

        for x1 in x[0]:

            self.layers[0][0].x = x1

            for x2 in x[1]:

                self.layers[0][1].x = x2

                for l in range(1, _l+2, 1):

                    # Activate output layers
                    if l == _l+1:
                        r = self.layers[l][_n].run()

                    # Activate hidden layers
                    else:
                        for n in range(len(self.layers[l])):
                            self.layers[l][n].run()

                if r >= 0.5:
                    plt.plot(x1, x2, 'rs', markersize=10, alpha=0.1, mec=None)
                else:
                    plt.plot(x1, x2, 'bs', markersize=10, alpha=0.1, mec=None)

        for data in _dataset:
            if data[2] == 1:
                plt.plot(data[0], data[1], 'ro')
            else:
                plt.plot(data[0], data[1], 'bo')

        plt.grid(True)
        plt.xlabel('x1', fontsize=16)
        plt.ylabel('x2', fontsize=16)
        plt.show()

    def train(self, _x, _t):

        for l in range(len(self.layers)):

            # Set Input to new value
            if l == 0:
                for n in range(len(self.layers[l])):
                    self.layers[l][n].x = _x[n]

            # Activate hidden layers and output layer
            else:
                for n in range(len(self.layers[l])):
                    self.layers[l][n].run()

        # Training
        for l in range(len(self.layers)-1, 0, -1):

            # Output layer
            if l == len(self.layers)-1:
                for n in range(len(self.layers[l])):
                    self.layers[l][n].train(_t[n])

            # BackPropagation
            else:
                dw = list() # dw = delta * weight of each node
                db = list()

                # Node of the layer
                for n in range(len(self.layers[l])):

                    # nn = node of next layer
                    for nn in range(len(self.layers[l+1])):

                        w = self.layers[l+1][nn].w[n]
                        dw.append(w * self.layers[l+1][nn].dw)
                        db.append(w * self.layers[l+1][nn].db)

                    self.layers[l][n].backpropagation(np.sum(dw), np.sum(db))

                    # reset the lists
                    dw = list()
                    db = list()

    def run(self, _x):

        for l in range(len(self.layers)):

            # Set Input to new value
            if l == 0:
                for n in range(len(self.layers[l])):
                    self.layers[l][n].x = _x[n]

            # Activate output layers
            elif l == len(self.layers)-1:
                for n in range(len(self.layers[l])):
                    self.y[n] = self.layers[l][n].run()

            # Activate hidden layers
            else:
                for n in range(len(self.layers[l])):
                    self.layers[l][n].run()

        return self.y

def classification2(type=1):

        if type == 1:
            (dataset, cat, disx, disy) = \
                pickle.load(file('dataset/classification2_1'))
        elif type == 2:
            (dataset, cat, disx, disy) = \
                pickle.load(file('dataset/classification2_2'))
        elif type == 3:
            (dataset, cat, disx, disy) = \
                pickle.load(file('dataset/classification2_3'))
        elif type == 4:
            (dataset, cat, disx, disy) = \
                pickle.load(file('dataset/classification2_4'))

        # Create Neural network with 2 input parameters and 1 neurone
        nn = Neural_network(2, [1])

        # Train the neural network, select randomly the data in the dataset
        for x in np.random.permutation(np.arange(len(dataset))):
            nn.train(dataset[x][:2], [dataset[x][2]])

        # Result for graphical display
        plt.figure("Dataset_solution")

        x1 = np.linspace(disx[0], disx[1], 30)
        x2 = np.linspace(disy[0], disy[1], 30)

        for i in x1:
            for y in x2:
                out = nn.run([i,y])

                if out < 0.5:
                    c = colorlist[0]
                else:
                    c = colorlist[1]

                plt.plot(i, y, c + 's', markersize=10, alpha=0.1, mec=None)

        for data in dataset:
            for i in range(len(cat)):
                if data[2] == cat[i] and i < len(colorlist):
                    plt.plot(data[0], data[1], colorlist[i] + 'o')

        plt.grid(True)
        plt.xlabel('x1', fontsize=16)
        plt.ylabel('x2', fontsize=16)
        plt.show()

def xor():

    # Create Neural network with 2 input parameters and 1 neurone
    nn = Neural_network(2, [2, 1])

    disx    = [0, 1]
    disy    = [0, 1]
    cat     = [0, 1]
    dataset = [[0, 0, 0],
               [0, 1, 1],
               [1, 0, 1],
               [1, 1, 0]]

    # Train the neural network, select randomly the data in the dataset
    for i in range(10000):
        for x in np.random.permutation(np.arange(len(dataset))):
            nn.train(dataset[x][:2], [dataset[x][2]])

    # Result for graphical display
    plt.figure("Dataset_solution")

    x1 = np.linspace(disx[0], disx[1], 30)
    x2 = np.linspace(disy[0], disy[1], 30)

    for i in x1:
        for y in x2:
            out = nn.run([i,y])

            if out < 0.5:
                c = colorlist[0]
            else:
                c = colorlist[1]

            plt.plot(i, y, c + 's', markersize=10, alpha=0.1, mec=None)

    for data in dataset:
        for i in range(len(cat)):
            if data[2] == cat[i] and i < len(colorlist):
                plt.plot(data[0], data[1], colorlist[i] + 'o')

    plt.grid(True)
    plt.xlabel('x1', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.show()

def xor_advanced():

    (dataset, cat, disx, disy) = pickle.load(file('dataset/xor_problem'))

    # Create Neural network with 2 input parameters and 1 neurone
    nn = Neural_network(2, [5, 3, 1])

    # Train the neural network, select randomly the data in the dataset
    for i in range(100):
        for x in np.random.permutation(np.arange(len(dataset))):
            nn.train(dataset[x][:2], [dataset[x][2]])

    # Result for graphical display
    plt.figure("Dataset_solution")

    x1 = np.linspace(disx[0], disx[1], 30)
    x2 = np.linspace(disy[0], disy[1], 30)

    for i in x1:
        for y in x2:
            out = nn.run([i,y])

            if out < 0.5:
                c = colorlist[0]
            else:
                c = colorlist[1]

            plt.plot(i, y, c + 's', markersize=10, alpha=0.1, mec=None)

    for data in dataset:
        for i in range(len(np.unique(cat))):
            if data[2] == cat[i] and i < len(colorlist):
                plt.plot(data[0], data[1], colorlist[i] + 'o')

    plt.grid(True)
    plt.xlabel('x1', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.show()

def classification4():

    (dataset, cat, disx, disy) = pickle.load(file('dataset/classification4'))

    # Create Neural network with 2 input parameters and 1 neurone
    nn = Neural_network(2, [4, 2])

    # Train the neural network, select randomly the data in the dataset
    for i in range(100):
        for x in np.random.permutation(np.arange(len(dataset))):
                nn.train(dataset[x][:2], dataset[x][2])

    # Result for graphical display
    plt.figure("Dataset_solution")

    x1 = np.linspace(disx[0], disx[1], 30)
    x2 = np.linspace(disy[0], disy[1], 30)

    for i in x1:
        for y in x2:
            out = nn.run([i,y])

            if out[0] < 0.5 and out[1] < 0.5:       # 0
                c = colorlist[0]
            elif out[0] >= 0.5 and out[1] < 0.5:    # 1
                c = colorlist[1]
            elif out[0] < 0.5 and out[1] >= 0.5:    # 2
                c = colorlist[2]
            elif out[0] >= 0.5 and out[1] >= 0.5:   # 3
                c = colorlist[3]

            plt.plot(i, y, c + 's', markersize=10, alpha=0.1, mec=None)

    for data in dataset:
        for i in range(len(cat)):
            if data[2] == cat[i] and i < len(colorlist):
                plt.plot(data[0], data[1], colorlist[i] + 'o')

    plt.grid(True)
    plt.xlabel('x1', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.show()

def kernel():

    (dataset, cat, disx, disy) = pickle.load(file('dataset/kernel'))

    # Create Neural network with 2 input parameters and 1 neurone
    nn = Neural_network(2, [8, 1])

    # Train the neural network, select randomly the data in the dataset
    for i in range(100):
        for x in np.random.permutation(np.arange(len(dataset))):
            nn.train(dataset[x][:2], [dataset[x][2]])

    # Result for graphical display
    plt.figure("Dataset_solution")

    x1 = np.linspace(disx[0], disx[1], 30)
    x2 = np.linspace(disy[0], disy[1], 30)

    for i in x1:
        for y in x2:
            out = nn.run([i,y])

            if out < 0.5:
                c = colorlist[0]
            else:
                c = colorlist[1]

            plt.plot(i, y, c + 's', markersize=10, alpha=0.1, mec=None)

    for data in dataset:
        for i in range(len(cat)):
            if data[2] == cat[i] and i < len(colorlist):
                plt.plot(data[0], data[1], colorlist[i] + 'o')

    plt.grid(True)
    plt.xlabel('x1', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.show()

if __name__ == "__main__":

    # classification2(1)
    # classification2(2)
    # classification2(3)
    # classification2(4)
    # xor()
    # xor_advanced()
    # classification4()
    kernel()
