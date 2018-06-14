# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

import gzip
import pickle
import numpy as np
from search import *
from collections import defaultdict

class Filter:
    """ Filter for Convolutional layer"""

    # w = weight
    # b = bias
    # dw = delta weight
    # db = delta bias
    # alpha = learning factor

    b = 0
    db = 0

    # Constructor
    def __init__(self, fShape, alpha=0.1):

        self.w  = np.random.uniform(low=0.0, high=0.01, size=fShape)
        self.dw = np.zeros(fShape)

        self.alpha = alpha

    def fUpdate(self, dw=0):
        self.w = self.w - self.alpha * dw

    def bUpdate(self, db=0):
        self.b = self.b - self.alpha * db

class Softmax:
    """Softmax Neurone"""

    # x = input
    # y = output
    # t = training data
    # phi = back propagation value

    y   = 0
    phi = 0

    # Constructor
    def __init__(self):
        pass

    def train(self, t):
        self.phi = self.y - t

    def run(self, x):

        e = np.zeros(x.shape)
        max = np.max(x)

        # Calculate exponentiel of each value
        for i in range(len(x)):
    	       e[i] = np.exp(x[i] - max, dtype=np.float)

        self.y = e / sum(e)

        return self.y

class FC:
    """ Fully Connected Neurone"""

    # x = input
    # y = output
    # b = bias
    # w = weight
    # alpha = learning factor
    # phiw = value used for backpropagation of the weight
    # phib = value used for backpropagation of the bias

    y   = 0
    b   = 0
    dw  = 0
    db  = 0
    phiw = 0

    # Constructor
    def __init__(self, nbrIn, alpha=0.1):
        self.x = np.zeros(nbrIn)
        self.w = np.random.uniform(low=0.0, high=0.01, size=nbrIn)

        self.alpha = alpha

    def backpropagation(self, phiw, phib):
        self.phiw = phiw * self.w.T
        self.w = self.w - self.alpha * phiw * self.x
        self.b = self.b - self.alpha * phib

    def run(self, x):

        # Save for backpropagation
        self.x = x

        # Adder
        self.y = self.b + np.sum(x * self.w)

        return self.y

class ReLu:
    """Rectified Linear Unit Neurone"""

    # x = input
    # y = output
    # phi = value used for backpropagation

    y   = 0
    phi = 0

    # Constructor
    def __init__(self):
        self.x = 0

    def backpropagation(self, phi):
        if self.x < 0:
            self.phi = 0
        else:
            self.phi = phi

    def run(self, x):

        # Save for backpropagation
        self.x = x

        self.y = np.max([0, x])

        return self.y

class MaxPool:
    """Max Pooling Neurone"""

    # x = input
    # y = output
    # i = keep the index of the max value, used for backpropagation
    # phi = value used for backpropagation

    y   = 0
    i   = 0

    # Constructor
    def __init__(self, inShape):
        self.phi = np.zeros(inShape)

    def backpropagation(self, phi):
        self.phi.fill(0)
        self.phi[self.i] = phi

    def run(self, x):

        self.y = np.max(x)

        # Save the index of the max value for backpropagation
        self.i = np.unravel_index(np.argmax(x, axis=None), x.shape)

        return self.y

class Conv:
    """Convolutional Neurone"""

    # x = input
    # y = output
    # phi = back propagation value
    # w = weight
    # b = bias
    # dw = delta weight

    y   = 0
    df  = 0
    phi = 0

    # Constructor
    def __init__(self, inShape, filter):
        self.x = np.zeros(inShape)
        self.filter = filter

    def backpropagation(self, phi):
        self.dw  = np.sum(self.x * phi, axis=0)
        self.phi = self.filter.w * phi

    def run(self, x):

        # Save for backpropagation
        self.x = x

        # Convolution
        self.y = self.filter.b + np.sum(x * self.filter.w)

        return self.y

class CNN:
    """Convolutional Neural Network"""

    # f = filter / kernel size
    # s = Stride
    # p = Padding
    # d = depth
    # w = width of features map
    # n = number of neurones / neurones
    # k = nbr filters for a Conv layer
    # l = {"Conv": (w, f, p, s), "ReLu": (), "MaxPool": ..., "FC":...}

    def __init__(self, layers, inShape):

        self.lInfo   = []   # All the layers type and shape
        self.layers  = []   # All the layers of neurones
        self.filters = {}   # All the filter for the Conv neurones

        # Parse the layers indicated and create the neural network
        for l in range(len(layers)):

            type = layers[l][0]
            data = layers[l][1]

            # Add a new layer
            self.lInfo.append(layers[l])
            self.layers.append([])

            ll = len(self.layers)-1  # last layer index
            pl = len(self.layers)-2  # previous layer index

            if type == "Conv":

                (nbrD, inShape, fShape, pShape, sShape, alpha) = data

                # Determine the number of neurones
                n1, n2 = self.nbr_neurones(inShape, fShape, pShape, sShape)

                # Each depth of the layer has one filter which is common to all
                # the neurones of the depth. The bias is part of the class Filter
                self.filters[ll] = []

                # Create the neurones and add them to the layer
                for d in range(nbrD):

                    # Add depth to the layer
                    self.layers[ll].append([])

                    # Add filter to the layer for the new depth
                    self.filters[ll].append(Filter(fShape, alpha))

                    for i in range(n1):

                        self.layers[ll][d].append([])

                        for y in range(n2):

                            # Create new Conv neurone
                            self.layers[ll][d][i].append(Conv(inShape, \
                                                              self.filters[ll][d]))

            elif type == "ReLu":

                # Each ReLu node is directly connected to the previous one.
                # Hence, there is as much ReLu neurone than the number of
                # neurones in the previous layer.
                n1, n2 = len(self.layers[pl][0]), len(self.layers[pl][0][0])

                # Create the neurones and add them to the layer
                for d in range(len(self.layers[pl])):

                    # Add depth to the layer
                    self.layers[ll].append([])

                    for i in range(n1):

                        self.layers[ll][d].append([])

                        for y in range(n2):

                            # Create new ReLu neurone
                            self.layers[ll][d][i].append(ReLu())

            elif type == "MaxPool":

                (inShape, poolShape, sShape) = data

                # Determine the number of neurones
                n1, n2 = self.nbr_neurones(inShape, poolShape, (0,0), sShape)

                # Create the neurones and add them to the layer
                for d in range(len(self.layers[pl])):

                    # Add depth to the layer
                    self.layers[ll].append([])

                    for i in range(n1):

                        self.layers[ll][d].append([])

                        for y in range(n2):

                            # Create MaxPool neurone
                            self.layers[ll][d][i].append(MaxPool(poolShape))

            elif type == "FC":

                # Get number of neurones
                n, alpha = data

                nbrIn = len(self.layers[pl]) *\
                        len(self.layers[pl][0]) *\
                        len(self.layers[pl][0][0])

                # Create the full Connected layer
                for _ in range(n):

                    # Create New Neurone
                    self.layers[ll].append(FC(nbrIn, alpha))

            elif type == "Softmax":

                # Create New Neurone
                self.layers[ll] = Softmax()

        # Creat the array for the output values
        self.y = None

    def nbr_neurones(self, w, f, p, s):

        # (w - f + 2 * p) / s + 1 = number of neurones along each row
        # w = input width
        # f = filter dimension
        # p = padding
        # s = stride

        return ((w[0] - f[0] + 2 * p[0]) / s[0] + 1), \
               ((w[1] - f[1] + 2 * p[1]) / s[1] + 1)

    def train(self, x, t):

        self.run(x)

        nbrLayers = len(self.layers)

        # Training
        for l in range(nbrLayers-1, 0, -1):

            # Last layer (Softmax) -> training with expected output
            if l == nbrLayers-1:
                self.layers[l].train(t)

            # BackPropagation
            else:

                if self.lInfo[l][0]  == "Conv":

                    if self.lInfo[l+1][0] == "FC": # TODO
                        pass

                    elif self.lInfo[l+1][0]  == "ReLu":

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):

                            dw = np.zeros(self.filters[l][d].w.shape)
                            db = 0

                            for n1 in range(_n1):
                                for n2 in range(_n2):

                                    phi = self.layers[l+1][d][n1][n2].phi

                                    self.layers[l][d][n1][n2].backpropagation(phi)

                                    # Update layer's filter
                                    dw = np.add(dw, self.layers[l][d][n1][n2].dw)
                                    db += phi

                            self.filters[l][d].fUpdate(dw)
                            self.filters[l][d].bUpdate(db)

                    elif self.lInfo[l+1][0]  == "MaxPool": # TODO
                        pass

                    elif self.lInfo[l+1][0]  == "Conv": # TODO
                        pass

                elif self.lInfo[l][0]  == "ReLu":

                    if self.lInfo[l+1][0] == "FC": # TODO
                        pass

                    elif self.lInfo[l+1][0] == "MaxPool":

                        phi = np.zeros((len(self.layers[l]), \
                                        len(self.layers[l][0]), \
                                        len(self.layers[l][0][0])))

                        _d  = len(self.layers[l+1])
                        _n1 = len(self.layers[l+1][0])
                        _n2 = len(self.layers[l+1][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):

                                    (_, f, s)   = self.lInfo[l+1][1]
                                    (x1, x2) = (n1*s[0], n1*s[0]+f[0])
                                    (y1, y2) = (n2*s[1], n2*s[1]+f[1])

                                    temp = self.layers[l+1][d][n1][n2].phi
                                    phi[d, x1:x2, y1:y2] = temp

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):
                                    self.layers[l][d][n1][n2].backpropagation(phi[d][n1][n2])

                    elif self.lInfo[l+1][0] == "Conv": # TODO padding

                        phi = np.zeros((len(self.layers[l][0]), \
                                        len(self.layers[l][0][0])))

                        _d  = len(self.layers[l+1])
                        _n1 = len(self.layers[l+1][0])
                        _n2 = len(self.layers[l+1][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):

                                    (_, _, f, p, s, _) = self.lInfo[l+1][1]
                                    (x1, x2) = (n1*s[0], n1*s[0]+f[0])
                                    (y1, y2) = (n2*s[1], n2*s[1]+f[1])

                                    temp = self.layers[l+1][d][n1][n2].phi
                                    phi[x1:x2, y1:y2] = np.add(phi[x1:x2, y1:y2], temp)

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):
                                    self.layers[l][d][n1][n2].backpropagation(phi[n1][n2])

                elif self.lInfo[l][0]  == "MaxPool":

                    if self.lInfo[l+1][0] == "FC":

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):

                                    phiw = list()

                                    # nn = node of next layer
                                    for nn in range(len(self.layers[l+1])):

                                        n = d  * len(self.layers[l]) + \
                                            n1 * len(self.layers[l][d]) + n2

                                        # Get the backpropagation value from
                                        # the previous layer
                                        phiw.append(self.layers[l+1][nn].phiw[n])

                                    self.layers[l][d][n1][n2].backpropagation(np.sum(phiw))

                    elif self.lInfo[l+1][0] == "MaxPool": # TODO
                        pass

                    elif self.lInfo[l+1][0] == "Conv":  # TODO
                        pass

                elif self.lInfo[l][0]  == "FC":

                    if self.lInfo[l+1][0]  == "Softmax":

                        # Get the backpropagation value from the previous layer
                        phiw = self.layers[l+1].phi
                        phib = np.sum(phiw)

                        for n in range(len(self.layers[l])):
                            self.layers[l][n].backpropagation(phiw[n], phib)

                    elif self.lInfo[l+1][0]  == "FC":  # TODO
                        pass

        return self.y

    def run(self, x):

        nbrLayers = len(self.layers)

        # Parse each layer
        for l in range(nbrLayers):

            # Layer's input value
            if l == 0:
                lIn = x
            else:
                lIn = lOut

            # Layer's output value
            lOut = np.zeros(np.asarray(self.layers[l]).shape)

            if self.lInfo[l][0] == "Softmax":
                self.y = self.layers[l].run(lIn)

            elif self.lInfo[l][0] == "FC":
                for n in range(len(self.layers[l])):
                    lOut[n] = self.layers[l][n].run(lIn.reshape(-1))

            else:   # Conv, ReLu, MaxPool

                _d  = len(self.layers[l])
                _n1 = len(self.layers[l][0])
                _n2 = len(self.layers[l][0][0])

                for d in range(_d):
                    for n1 in range(_n1):
                        for n2 in range(_n2):

                            if self.lInfo[l][0] == "Conv": # TODO padding

                                # Determine the input matrix of the neurone
                                (_, _, f, p, s, _) = self.lInfo[l][1]
                                (x1, x2) = (n1*s[0], n1*s[0]+f[0])
                                (y1, y2) = (n2*s[1], n2*s[1]+f[1])

                                nIn = lIn[:, x1:x2, y1:y2]

                            elif self.lInfo[l][0] == "ReLu":
                                nIn = lIn[d, n1, n2]

                            elif self.lInfo[l][0] == "MaxPool":

                                # Determine the input matrix of the neurone
                                (_, f, s)   = self.lInfo[l][1]
                                (x1, x2) = (n1*s[0], n1*s[0]+f[0])
                                (y1, y2) = (n2*s[1], n2*s[1]+f[1])

                                nIn = lIn[d, x1:x2, y1:y2]

                            lOut[d, n1, n2] = self.layers[l][d][n1][n2].run(nIn)

        return self.y

# ------------------------------------------------------------------------------

if __name__ == "__main__":

    nbr_images = 10000

    w = 28
    d = 1

    # Extract image from MNIST dataset
    with gzip.open("dataset/train-images-idx3-ubyte.gz") as bytestream:

        # Throw away the 16 first bytes
        bytestream.read(16)

        # Get the images
        buf = bytestream.read(w * w * nbr_images)

        # Encode the data
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(nbr_images, w * w)

    # Extract label from MNIST dataset
    with gzip.open("dataset/train-labels-idx1-ubyte.gz") as bytestream:

        # Throw away the 8 first bytes
        bytestream.read(8)

        # Get the labels
        buf = bytestream.read(nbr_images)

        # Encode the labels
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    # Create the convolutional neural network, w = input
    # Conv    -> (depth, inShape, fShape, padding, stride, alpha)
    # ReLu    -> ()
    # MaxPool -> (inShape, fShape, stride)
    # FC      -> (nbrIn, alpha)
    # Softmax -> ()

    # (w - f + 2 * p) / s + 1 = number of neurones along each row

    l = [
            ("Conv",     (8, (28,28), (5,5), (0,0), (1,1), 0.000001)),
            ("ReLu",     ()),
            ("Conv",     (8, (24,24), (5,5), (0,0), (1,1), 0.000001)),
            ("ReLu",     ()),
            ("MaxPool",  ((20,20), (2,2), (2,2))),
            ("FC",       (10, 0.01)),
            ("Softmax",  ()),
        ]

    output = [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0),   # 0
              (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),   # 1
              (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),   # 2
              (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),   # 3
              (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),   # 4
              (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),   # 5
              (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),   # 6
              (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),   # 7
              (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),   # 8
              (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)]   # 9

    try:
        cnn = pickle.load(file('saving/cnn'))
        print "CNN reloaded"
    except:
        cnn = CNN(l, (d, w, w))
        pickle.dump(cnn, file('saving/cnn','w'))

    # Train the convolutional neural network
    for i in range(len(data)):
        result = cnn.train(data[i].reshape(d, w, w), output[labels[i]])

        print "Training set:" + str(i)
        print "Answer:", np.argmax(result)
        print "Right Answer:", labels[i]
        print "Prob:", np.max(result)
        print

    pickle.dump(cnn, file('saving/cnn','w'))

    cnt_tested  = 0.0
    cnt_correct = 0.0

    # Train the convolutional neural network
    for i in range(50):
        result = cnn.run(data[i].reshape(d, w, w))

        cnt_tested += 1.0

        if np.argmax(result) == labels[i]:
            cnt_correct += 1.0

        print "Answer:", np.argmax(result)
        print "Right Answer:", labels[i]
        print "Prob:", np.max(result)
        print "Ratio:", (cnt_correct/cnt_tested)*100.0, "%"
        print
