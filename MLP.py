# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

import pickle
from math import *
import numpy as np
import matplotlib.pyplot as plt
from datasetGenerator import display, colorlist, markerlist

class Perceptron:
    """Perceptron Neurone"""

    # x     = input
    # z     = adder output
    # y     = output
    # w     = weights
    # t     = training data
    # b     = bias
    # gamma = smoothing factor
    # alpha = learning factor
    # dw    = delta weight
    # db    = delta bias
    # dnw   = delta node weight for backpropagation
    # dnb   = delta node bias for backpropagation

    b = 0
    y = 0

    # phiw = 0
    # phib = 0
    # phiw  = value used for backpropagation of the weight
    # phib  = value used for backpropagation of the bias

    dw = 0
    db = 0

    dnw = 0
    dnb = 0

    # Constructor
    def __init__(self, nbrIn, gamma=0.5, alpha=0.4):

        self.x = np.zeros(nbrIn)
        self.gamma = gamma
        self.alpha = alpha

        # Need to assign rand weight to train a network with hidden layers
        self.w = np.random.uniform(low=-2, high=2, size=nbrIn)

    def activation(self, y):
        return (1.0 / (1.0 + exp(-y))) # Sigmoid

    def backpropagation(self, dn, w):

        gamma = self.gamma
        alpha = self.alpha

        # Error probagate from layer +1 and using Sigmoid as activation
        # function
        #
        # dE/dw   = dE/dy * dy/dz * dz/dw
        # dE/dy   = sum{i, dE/dz+1i * dz+1i/dy} where i = ith node of layer +1
        # dE/dz+1 = dE/dy+1i * dy+1i/dz+1i = dn+1i
        # dz+1/dy = w+1i
        # dy/dz   = y * (1 - y)
        # dz/dw   = x
        #
        # Delta node:
        # dn = dE/dy * dy/dz = sum{i, dE/dz+1i * dz+1i/dy} * y * (1 - y)

        # Calculate the delta node used in backpropagation
        self.dn = self.y * (1.0 - self.y) * np.sum(dn * w)

        # Calculate dw and db using smoothing factor
        # dw = gamma * dw + (1 - gamma) * dn * x
        self.db = gamma * self.db + (1.0 - gamma) * self.dn
        self.dw = np.add(gamma * self.dw,\
                         np.multiply((1.0 - gamma) * self.dn, self.x))

        # wNew = wOld + alpha * dw
        self.w = self.w + self.alpha * self.dw

        # bNew = bOld + a * db
        self.b = self.b + self.alpha * self.db

    def train(self, t):

        if t != self.y:

            gamma = self.gamma
            alpha = self.alpha

            # Squared Error is used to train the neural network and using
            # sigmoid as activation function
            #
            # dE/dw = dE/dy * dy/dz * dz/dw
            # dE/dy = (t - y)
            # dy/dz = y * (1 - y)
            # dz/dw = x
            #
            # Delta node:
            # dn = dE/dy * dy/dz = (t - y) * y * (1 - y)

            # Calculate the delta node used in backpropagation
            self.dn = self.y * (1.0 - self.y) * (t   - self.y)

            # Calculate dw and db using smoothing factor
            # dw = gamma * dw + (1 - gamma) * dn * x
            self.db = gamma * self.db + (1.0 - gamma) * self.dn
            self.dw = np.add(gamma * self.dw,\
                             np.multiply((1.0 - gamma) * self.dn, self.x))

            # wNew = wOld + alpha * dw
            self.w = self.w + alpha * self.dw

            # bNew = bOld + a * db
            self.b = self.b + alpha * self.db

    def run(self, x):

        # Save the input for backpropagation
        self.x = x

        # Adder
        z = self.b + np.sum(self.x * self.w)

        # Activation function
        self.y = self.activation(z)

        return self.y

class MLP:
    """Multilayers Perceptron"""

    # x = input
    # y = ouput
    # t = training dataset
    # layers = [nbr of nodes of layer 1, nbr of nodes of layer 2, ...]

    # Constructor
    def __init__(self, nbrIn, layers, gamma=0.5, alpha=0.4):

        # List of the layers (input=0, hidden, output=last one)
        self.layers = list()

        for l in range(len(layers)):

            # Add a new layer
            self.layers.append(list())

            # Determine the number of input
            if l == 0:
                n = nbrIn
            else:
                n = layers[l-1]

            # Create the neurone
            for _ in range(layers[l]):
                self.layers[l].append(Perceptron(n, gamma, alpha))

        # Creat the array for the output values
        self.y = np.zeros(layers[len(layers)-1])

    def getNeuron(self, l, n):

        # _l+1 because first layer has only the input parameters
        return self.layers[l+1][n]

    def train(self, x, t):

        self.run(x)

        # Training
        for l in range(len(self.layers)-1, -1, -1):

            # Output layer
            if l == len(self.layers)-1:
                for n in range(len(self.layers[l])):
                    self.layers[l][n].train(t[n])

            # BackPropagation
            else:

                # Node of the layer
                for n in range(len(self.layers[l])):

                    # Number of neurones in the previous layer
                    pLen = len(self.layers[l+1])

                    # Create/reset the lists
                    w  = np.zeros(pLen)
                    dn = np.zeros(pLen)

                    # nn = node of next layer
                    for nn in range(pLen):

                        w[nn]  = self.layers[l+1][nn].w[n]
                        dn[nn] = self.layers[l+1][nn].dn

                        self.layers[l][n].backpropagation(dn, w)

        return self.y

    def run(self, x):

        for l in range(len(self.layers)):

            # Number of neurones in the layer
            lLen = len(self.layers[l])

            # Layer's input value
            if l == 0:
                lIn = x
            else:
                lIn = lOut

            # Layer's output value
            lOut = np.zeros(lLen)

            for n in range(lLen):
                lOut[n] = self.layers[l][n].run(lIn)

        self.y = lOut

        return self.y

# ------------------------------------------------------------------------------

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
        nn = MLP(2, [1])

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

    (dataset, cat, disx, disy) = pickle.load(file('dataset/xor_problem'))

    # Create Neural network with 2 input parameters and 1 neurone
    nn = MLP(2, [5, 3, 1])

    # Train the neural network, select randomly the data in the dataset
    for i in range(200):
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
    nn = MLP(2, [4, 2])

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
    nn = MLP(2, [8, 1])

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

    classification2(1)
    classification2(2)
    classification2(3)
    classification2(4)
    xor()
    classification4()
    kernel()
