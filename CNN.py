    # -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

import gzip
import pickle
import numpy as np

class Filter:
    """ Filter for Convolutional layer"""

    # w     = weight
    # b     = bias
    # dw    = delta weight
    # db    = delta bias
    # alpha = learning factor

    b = 0
    db = 0

    # Constructor
    def __init__(self, fShape, alpha=0.1):

        self.w  = np.random.uniform(low=0.0, high=0.01, size=fShape)
        self.dw = np.zeros(fShape)

        self.alpha = alpha

    def update(self):
        self.w = self.w - self.alpha * self.dw
        self.b = self.b - self.alpha * self.db

        # Reset the detla weight and bias
        self.dw = np.zeros(self.dw.shape)
        self.db = 0

class Softmax:
    """Softmax Neurone"""

    # x  = input, vector
    # y  = output, vector
    # t  = training data
    # dn = delta node used for back propagation

    y  = 0
    dn = 0

    # Forward:
    # --------
    # yi = e^zi / sum{k, e^zk}
    # zi = xi - max(x), used for more stability
    #
    # Error:
    # -----
    # t is a vector with 1 at the correct expected result and the rest is 0
    # E = - sum{k, tk * log(yk)}
    #
    # Delta node:
    # -----------
    # dni =  dE/dxi
    #     = - sum{k, d(tk * log(yk))/dyk * dyk/dxi}
    # dlog(yk)/dyk = 1 / yk
    #              _ yk * (1 - yi) , if i = k
    # dyk/dxi  = /                              -> yk = e^xk / sum{j, e^xj}
    #            \ _ - yk * yi    , if i != k
    # dE/dxi = - sum{k, tk * 1/yk * dyk/dxi}
    #        = -tk * 1/yi * yi * (1 - yi) - sum{k not i, tk * 1/yk * -yk * yi}
    #        = -tk + tk * yi + tk * sum{k not i, yi}
    #        =  tk * sum(k, yk) - yi  -> sum(k, yk) = 1
    #        =  yi - tk

    # Constructor
    def __init__(self):
        pass

    def train(self, t):
        self.dn = self.y - t

    def run(self, x):

        e   = np.zeros(x.shape)
        max = np.max(x)

        # Calculate exponentiel of each value
        for i in range(len(x)):
    	       e[i] = np.exp(x[i] - max, dtype=np.float)

        self.y = e / sum(e)

        return self.y

class FC:
    """ Fully Connected Neurone"""

    # x     = input, vector
    # y     = output, real number
    # b     = bias
    # w     = weight
    # dn    = delta node used for backpropagation
    # alpha = learning factor

    y   = 0
    b   = 0
    dw  = 0
    db  = 0
    dn  = 0

    # Forward:
    # --------
    # y = sum{i, xi * wi} + b
    #
    # BackPropagation:
    # ----------------
    # dE/dwi = dE/dy * dy/dwi
    # dy/dwi = xi
    # dE/dy = dE/dx+1j                  -> l+1 = Softmax
    #       = dn+1j
    #       = dE/y+1j * dy+1j/dy
    # dE/dy = sum{n, dE/dx+1nj}         -> l+1 = FC
    #       = sum{n, dn+1nj}
    #       = sum{n, dE/dy+1nj * dy+1nj/dy}
    #
    # where l = lth layer,
    #       n = nth node of l+1,
    #       j = jth node of l == jth connection of node n == Current node
    #
    # Delta node:
    # -----------
    # dni = dE/dxi
    #    = dE/dy * dy/dxi
    #    = dE/dy * wi

    # Constructor
    def __init__(self, nbrIn, alpha=0.1):
        self.x = np.zeros(nbrIn)
        self.w = np.random.uniform(low=-0.01, high=0.01, size=nbrIn)

        self.alpha = alpha

    def back(self, dn, next="Softmax"):

        if next == "Softmax":
            temp = dn
        elif next == "FC":
            temp = np.sum(dn)

        self.w  = self.w - self.alpha * temp * self.x
        self.b  = self.b - self.alpha * temp
        self.dn = temp * self.w

    def run(self, x):

        # Save for backpropagation
        self.x = x

        # Adder
        self.y = self.b + np.sum(x * self.w)

        return self.y

class ReLu:
    """Rectified Linear Unit Neurone"""

    # x  = input, real number
    # y  = output, real number
    # dn = delta node used for backpropagation

    y  = 0
    dn = 0

    # Forward:
    # --------
    # y = x if x > 0 else 0
    #
    # Delta node:
    # -----------
    # dn = dE/dx
    #    = dE/dy * dy/dx
    # dy/dx = 1 if x > 0 else 0
    # dE/dy = dE/dx+1nj = dn+1nj       -> l+1 = MaxPool
    #
    # where l = lth layer,
    #     n = nth node of l+1
    #     j = jth node of l == jth connection of node n == current node

    # Constructor
    def __init__(self):
        self.x = 0

    def back(self, dn, prev="MaxPool"):

        if prev == "FC":
            temp = np.sum(dn)
        else:
            temp = dn

        if self.x < 0:
            self.dn = 0
        else:
            self.dn = temp

    def run(self, x):

        # Save for backpropagation
        self.x = x

        self.y = np.max([0, x])

        return self.y

class MaxPool:
    """Max Pooling Neurone"""

    # x  = input, matrix
    # y  = output, real number
    # i  = keep the index of the max value, used for backpropagation
    # dn = delta node used for backpropagation

    y = 0
    i = 0

    # Forward:
    # --------
    # y = max(x)
    #
    # Delta node:
    # -----------
    # dni = dE/dxi
    #     = dE/y * dy/dxi
    # dy/dxi = 1 if xi = max(x) else 0
    # dE/dy = dE/dx+1
    #       = sum{n, dE/dx+1nj}
    #       = sum{n, dn+1nj}
    #       = sum{n, dE/dy+1nj * dy+1nj/dx+1nj}   -> l+1 = FC
    #
    # where l = lth layer,
    #       n = nth node of l+1,
    #       j = jth node of l == jth connection of node n == Current node

    # Constructor
    def __init__(self, inShape):
        self.dn = np.zeros(inShape)

    def back(self, dn, next="FC"):

        if next == "FC":
            self.dn.fill(0)
            self.dn[self.i] = np.sum(dn)

        elif next == "Conv":
            pass

    def run(self, x):

        self.y = np.max(x)

        # Save the index of the max value for backpropagation
        self.i = np.unravel_index(np.argmax(x, axis=None), x.shape)

        return self.y

class Conv:
    """Convolutional Neurone"""

    # x  = input, matrix
    # y  = output, real number
    # w  = weight
    # b  = bias
    # dw = delta weight
    # dn = delta node for back propagation

    y  = 0
    dn = 0

    # Forward:
    # --------
    # The input matrix of the neurone is the same size than the weight used
    # for the convolution and is a subset of the matrix in the input of the layer.
    #
    # yij = conv(xij, w) = sum{nm, x(i+n)(j+m) * wnm} + b
    #
    #       Layer inputs     filter     layer outputs
    # E.g:  x00 x01 x02     w00 w01       y00 y01
    #       x10 x11 x12     w10 w11       y10 y11
    #       x20 x21 x22
    #
    # 1st Neurone: y00 = conv([(x00, x01), (x10, x11)], w)
    # 2nd Neurone: y01 = conv([(x01, x02), (x11, x12)], w)
    # 3rd Neurone: y10 = conv([(x10, x11), (x20, x21)], w)
    # 4th Neurone: y11 = conv([(x11, x12), (x21, x22)], w)
    #
    # Backpropagation:
    # ----------------
    # dE/dwnm   = sum{ij, dE/dyij * dyij/dwnm}
    # dE/dyij   = dn+1ij
    # dyij/dwnm = x(i+n)(j+m)
    #
    #       Layer inputs     filter         delta node
    # E.g:  x00 x01 x02     w00 w01       dn+100 dn+101
    #       x10 x11 x12     w10 w11       dn+110 dn+111
    #       x20 x21 x22
    #
    # dE/dw00 = dE/dy00 * x00 + dE/dy01 * x01 + dE/dy10 * x10 + dE/dy11 * x11
    # dE/dw01 = dE/dy00 * x01 + dE/dy01 * x02 + dE/dy10 * x11 + dE/dy11 * x12
    #
    # But a neurone get only one dE/dyij corresponding to its output. Hence:
    # 1st Neurone: dE/dy00 * x00, dE/dy00 * x01, dE/dy00 * x10, dE/dy00 * x11
    # 2nd Neurone: dE/dy01 * x01, dE/dy01 * x02, dE/dy01 * x11, dE/dy01 * x12
    # 3rd Neurone: dE/dy10 * x10, dE/dy10 * x11, dE/dy10 * x20, dE/dy10 * x21
    # 4th Neurone: dE/dy11 * x11, dE/dy12 * x01, dE/dy21 * x10, dE/dy22 * x11
    #           + -----------------------------------------------------------
    #              dE/dw00      , dE/dw01      , dE/dw10      , dE/dw11
    #
    # Delta node:
    # -----------
    # dnij = full_conv(dE/dyij * w.T)
    #
    # dn00 = dE/dy00 * w00
    # dn01 = dE/dy00 * w01 + dE/dy01 * w00
    # dn02 =               + dE/dy01 * w01
    #
    # But a neurone get only one dE/dyij corresponding to its output. Hence:
    # 1st Neurone: dE/dy00 * w00, dE/dy00 * w01, dE/dy00 * w10, dE/dy00 * w11
    # 2nd Neurone: dE/dy01 * w00, dE/dy01 * w01, dE/dy01 * w10, dE/dy01 * w11
    # 3rd Neurone: dE/dy10 * w00, dE/dy10 * w01, dE/dy10 * w10, dE/dy10 * w11
    # 4th Neurone: dE/dy11 * w00, dE/dy11 * w01, dE/dy11 * w10, dE/dy11 * w11
    #
    # The addition of the different part of the dnij is done in the previous
    # layer during the backPropagation

    # Constructor
    def __init__(self, inShape, filter):
        self.x = np.zeros(inShape)
        self.filter = filter

    def back(self, dn, next="ReLu"):

        if next == "ReLu":
            self.filter.dw += np.sum(dn * self.x, axis=0)
            self.filter.db += dn

        elif next == "MaxPool":
            pass

        elif next == "FC":
            temp = np.sum(dn)
            self.filter.dw += np.sum(temp * self.x, axis=0)
            self.filter.db += temp

        self.dn = dn * self.filter.w

    def run(self, x):

        # Save for backpropagation
        self.x = x

        # Multiply the input by the filter
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

                    if self.lInfo[l+1][0] == "FC":

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):

                                    # Number of neurones in the next layer
                                    _nn = len(self.layers[l+1])

                                    # Create/reset the lists
                                    dn = np.zeros(_nn)

                                    # nn = node of next layer
                                    for nn in range(_nn):
                                        n = d * _d + n1 * _n1 + n2
                                        dn[nn] = self.layers[l+1][nn].dn[n]

                                    self.layers[l][d][n1][n2].back(dn, w, "FC")

                    elif self.lInfo[l+1][0]  == "ReLu":

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):

                                    dn = self.layers[l+1][d][n1][n2].dn
                                    self.layers[l][d][n1][n2].back(dn)

                            # Update the weight of the filter
                            self.filters[l][d].update()

                    elif self.lInfo[l+1][0]  == "MaxPool": # TODO
                        pass

                    elif self.lInfo[l+1][0]  == "Conv": # TODO
                        pass

                elif self.lInfo[l][0]  == "ReLu":

                    if self.lInfo[l+1][0] == "FC": # TODO

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):

                                    # Number of neurones in the next layer
                                    _nn = len(self.layers[l+1])

                                    # Create/reset the lists
                                    dn = np.zeros(_nn)

                                    # nn = node of next layer
                                    for nn in range(_nn):
                                        n = d * _d + n1 * _n1 + n2
                                        dn[nn] = self.layers[l+1][nn].dn[n]

                                    self.layers[l][d][n1][n2].back(dn, "FC")

                    elif self.lInfo[l+1][0] == "MaxPool":

                        dn = np.zeros((len(self.layers[l]), \
                                        len(self.layers[l][0]), \
                                        len(self.layers[l][0][0])))

                        _nd  = len(self.layers[l+1])
                        _nn1 = len(self.layers[l+1][0])
                        _nn2 = len(self.layers[l+1][0][0])

                        for nd in range(_nd):
                            for nn1 in range(_nn1):
                                for nn2 in range(_nn2):

                                    (_, f, s)   = self.lInfo[l+1][1]
                                    (x1, x2) = (nn1*s[0], nn1*s[0]+f[0])
                                    (y1, y2) = (nn2*s[1], nn2*s[1]+f[1])

                                    temp = self.layers[l+1][nd][nn1][nn2].dn
                                    dn[d, x1:x2, y1:y2] = temp

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):
                                    self.layers[l][d][n1][n2].back(dn[d][n1][n2])

                    elif self.lInfo[l+1][0] ==  "Conv": # TODO padding

                        dn = np.zeros((len(self.layers[l][0]), \
                                        len(self.layers[l][0][0])))

                        _nd  = len(self.layers[l+1])
                        _nn1 = len(self.layers[l+1][0])
                        _nn2 = len(self.layers[l+1][0][0])

                        for nd in range(_nd):
                            for nn1 in range(_nn1):
                                for nn2 in range(_nn2):

                                    (_, _, f, p, s, _) = self.lInfo[l+1][1]
                                    (x1, x2) = (nn1*s[0], nn1*s[0]+f[0])
                                    (y1, y2) = (nn2*s[1], nn2*s[1]+f[1])

                                    temp = self.layers[l+1][nd][nn1][nn2].dn
                                    dn[x1:x2, y1:y2] = np.add(dn[x1:x2, y1:y2],\
                                                              temp)

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):
                                    self.layers[l][d][n1][n2].back(dn[n1][n2])

                elif self.lInfo[l][0]  == "MaxPool":

                    if self.lInfo[l+1][0] == "FC":

                        _d  = len(self.layers[l])
                        _n1 = len(self.layers[l][0])
                        _n2 = len(self.layers[l][0][0])

                        for d in range(_d):
                            for n1 in range(_n1):
                                for n2 in range(_n2):

                                    # Number of neurones in the next layer
                                    _nn = len(self.layers[l+1])

                                    # Create/reset the lists
                                    dn = np.zeros(_nn)

                                    # nn = node of next layer
                                    for nn in range(_nn):
                                        n = d * _n1 * _n2 + n1 * _n2 + n2
                                        dn[nn] = self.layers[l+1][nn].dn[n]

                                    self.layers[l][d][n1][n2].back(dn)

                    elif self.lInfo[l+1][0] == "MaxPool": # TODO
                        pass

                    elif self.lInfo[l+1][0] == "Conv":  # TODO
                        pass

                elif self.lInfo[l][0]  == "FC":

                    if self.lInfo[l+1][0]  == "Softmax":

                        # Get the backpropagation value from the previous layer
                        dn = self.layers[l+1].dn

                        for n in range(len(self.layers[l])):
                            self.layers[l][n].back(dn[n])

                    elif self.lInfo[l+1][0]  == "FC":

                        # Number of neurones in the next layer
                        nLen = len(self.layers[l+1])

                        for n in range(len(self.layers[l])):

                            # Create/reset the lists
                            dn = np.zeros(nLen)

                            # nn = node of next layer
                            for nn in range(nLen):

                                dn[nn] = self.layers[l+1][nn].dn[n]

                            self.layers[l][n].back(dn, "FC")

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
                lOut = self.layers[l].run(lIn)

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

        self.y = lOut
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
