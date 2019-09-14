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
    dw = 0
    db = 0
    dn = 0

    # Constructor
    def __init__(self, nbrIn, gamma=0.5, alpha=0.4):

        self.x = np.zeros(nbrIn)
        self.gamma = gamma
        self.alpha = alpha

        # Need to assign rand weight to train a network with hidden layers
        self.w = np.random.uniform(low=-2, high=2, size=nbrIn)

    def back(self, dn):

        gamma = self.gamma
        alpha = self.alpha

        # dE/dwi  = dE/dy * dy/dz * dz/dwi
        # dy/dz   = y * (1 - y)
        # dz/dwi  = xi
        # dE/dy   = dE/dx+1
        #         = sum{n, dE/dx+1nj}
        #         = sum{n, dn+1nj}
        #         = sum{n, dE/dy+1nj * dy+1n/dz+1nj * dz+1nj/dx+1nj}
        #
        # where l = lth layer,
        #       n = nth node of l+1,
        #       j = jth node of l == jth connection of node n == Current node
        #
        # Delta node:
        # dni = dE/dxi
        #     = dE/dy * dy/dz * dz/dxi
        #     = sum{n, dn+1nj} * y * (1 - y) * wi

        # dE/dy * dE/dz
        temp = self.y * (1.0 - self.y) * np.sum(dn)

        # Calculate the delta node used in backpropagation
        self.dn = np.multiply(temp, self.w)

        # Calculate dw and db using smoothing factor
        # dwi = gamma * dwi + (1 - gamma) * dE/dwi
        self.db = gamma * self.db + (1.0 - gamma) * temp
        self.dw = np.add(gamma * self.dw,\
                         np.multiply((1.0 - gamma) * temp, self.x))

        # wNew = wOld + alpha * dw
        self.w = self.w + self.alpha * self.dw

        # bNew = bOld + a * db
        self.b = self.b + self.alpha * self.db

    def train(self, t):

        if t != self.y:

            gamma = self.gamma
            alpha = self.alpha

            # E = 1/2 * (t - y)^2
            #
            # dE/dwi = dE/dy * dy/dz * dz/dwi
            # dE/dy  = (t - y)
            # dy/dz  = y * (1 - y)
            # dz/dwi = xi
            #
            # dE/dwi = (t - y) * y * (1 - y) * xi
            #
            # Delta node:
            # dni = dE/dxi
            #     = dE/dy * dE/dz * dz/dxi
            #     = (t - y) * y * (1 - y) * wi

            # dE/dy * dy/dz
            temp = self.y * (1.0 - self.y) * (t   - self.y)

            # Calculate the delta node used in backpropagation
            self.dn = np.multiply(temp, self.w)

            # Calculate dw and db using smoothing factor
            # dwi = gamma * dwi + (1 - gamma) * dE/dwi
            self.db = gamma * self.db + (1.0 - gamma) * temp
            self.dw = np.add(gamma * self.dw,\
                             np.multiply((1.0 - gamma) * temp, self.x))

            # wNew = wOld + alpha * dw
            self.w = self.w + alpha * self.dw

            # bNew = bOld + a * db
            self.b = self.b + alpha * self.db

    def run(self, x):

        # z = sum{i, xi * wi} + b
        # y = sigmoid(z) = 1 / (1 + e^-z)

        # Save the input for backpropagation
        self.x = x

        # Adder
        z = self.b + np.sum(self.x * self.w)

        # Activation function
        self.y = (1.0 / (1.0 + exp(-z)))

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

                    # Number of neurones in the next layer
                    nLen = len(self.layers[l+1])

                    # Create/reset the lists
                    dn = np.zeros(nLen)

                    # nn = node of next layer
                    for nn in range(nLen):
                        dn[nn] = self.layers[l+1][nn].dn[n]

                    self.layers[l][n].back(dn)

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
                pickle.load(open('dataset/classification2_1', 'rb'))
        elif type == 2:
            (dataset, cat, disx, disy) = \
                pickle.load(open('dataset/classification2_2', 'rb'))
        elif type == 3:
            (dataset, cat, disx, disy) = \
                pickle.load(open('dataset/classification2_3', 'rb'))
        elif type == 4:
            (dataset, cat, disx, disy) = \
                pickle.load(open('dataset/classification2_4', 'rb'))

        # Create Neural network with 2 input parameters and 1 neurone
        nn = MLP(2, [1])

        train_data = dataset[:int(0.8 * len(dataset))]
        test_data  = dataset[int(0.8 * len(dataset)):]

        # Train the neural network, select randomly the data in the dataset
        for x in np.random.permutation(np.arange(len(train_data))):
            nn.train(train_data[x][:2], [train_data[x][2]])

        accuracy = 0

        for x in np.random.permutation(np.arange(len(test_data))):

            out = nn.run(test_data[x][:2])

            if out < 0.5 and test_data[x][2] == 0 or \
               out >= 0.5 and test_data[x][2] == 1:
                accuracy += 1

        print("accuracy of", accuracy / len(test_data))

def xor():

    (dataset, cat, disx, disy) = pickle.load(open('dataset/xor_problem', 'rb'))

    # Create Neural network with 2 input parameters and 1 neurone
    nn = MLP(2, [5, 3, 1])

    train_data = dataset[:int(0.8 * len(dataset))]
    test_data  = dataset[int(0.8 * len(dataset)):]

    # Train the neural network, select randomly the data in the dataset
    for i in range(len(train_data)):
        for x in np.random.permutation(np.arange(len(train_data))):
            nn.train(train_data[x][:2], [train_data[x][2]])

    accuracy = 0

    for x in np.random.permutation(np.arange(len(test_data))):

        out = nn.run(test_data[x][:2])

        if out < 0.5 and test_data[x][2] == 0 or \
           out >= 0.5 and test_data[x][2] == 1:
            accuracy += 1

    print("accuracy of", accuracy / len(test_data))

def classification4():

    (dataset, cat, disx, disy) = pickle.load(open('dataset/classification4', 'rb'))

    # Create Neural network with 2 input parameters and 1 neurone
    nn = MLP(2, [4, 2])

    train_data = dataset[:int(0.8 * len(dataset))]
    test_data  = dataset[int(0.8 * len(dataset)):]

    # Train the neural network, select randomly the data in the dataset
    for i in range(len(train_data)):
        for x in np.random.permutation(np.arange(len(train_data))):
                nn.train(train_data[x][:2], train_data[x][2])

    accuracy = 0

    for x in np.random.permutation(np.arange(len(test_data))):

        out = nn.run(test_data[x][:2])

        if out[0] < 0.5  and out[1] < 0.5  and np.array_equal(test_data[x][2], [0, 0]) or \
           out[0] >= 0.5 and out[1] < 0.5  and np.array_equal(test_data[x][2], [1, 0]) or \
           out[0] < 0.5  and out[1] >= 0.5 and np.array_equal(test_data[x][2], [0, 1]) or \
           out[0] >= 0.5 and out[1] >= 0.5 and np.array_equal(test_data[x][2], [1, 1]):
            accuracy += 1

    print("accuracy of", accuracy / len(test_data))

def kernel():

    (dataset, cat, disx, disy) = pickle.load(open('dataset/kernel', 'rb'))

    # Create Neural network with 2 input parameters and 1 neurone
    nn = MLP(2, [8, 1])

    train_data = dataset[:int(0.8 * len(dataset))]
    test_data  = dataset[int(0.8 * len(dataset)):]

    # Train the neural network, select randomly the data in the dataset
    for i in range(len(train_data)):
        for x in np.random.permutation(np.arange(len(train_data))):
            nn.train(train_data[x][:2], [train_data[x][2]])

    accuracy = 0

    for x in np.random.permutation(np.arange(len(test_data))):

        out = nn.run(test_data[x][:2])

        if out < 0.5 and test_data[x][2] == 0 or \
           out >= 0.5 and test_data[x][2] == 1:
            accuracy += 1

    print("accuracy of", accuracy / len(test_data))

if __name__ == "__main__":

    classification2(1)
    classification2(2)
    classification2(3)
    classification2(4)
    xor()
    classification4()
    kernel()
