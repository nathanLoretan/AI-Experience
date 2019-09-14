# -*- coding: utf-8 -*-
"""
Nathan Loretan
Convolutional Neural Network
04.09.18
"""

import gzip
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
    def __init__(self, f, alpha=0.1):

        # Xavier initialization
        self.w  = np.random.randn(f[0], f[1]) / np.sqrt(f[0] * f[1])
        self.dw = np.zeros(f)

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
            e[i] = np.exp(x[i], dtype=np.float)

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
    def __init__(self, inp, alpha=0.1):
        self.x = np.zeros(inp)

        # Xavier Initialization
        self.w = np.random.randn(inp) / np.sqrt(inp)
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
    """Rectified Linear Unit"""

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

        # if prev == "FC":
        #     temp = np.sum(dn)
        # else:
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
    def __init__(self, inp):
        self.dn = np.zeros(inp)

    def back(self, dn):

            self.dn.fill(0)
            self.dn[self.i] = np.sum(dn)

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
    def __init__(self, inp, f):
        self.x = np.zeros(inp)
        self.filter = f

    def back(self, dn):

        self.filter.dw = np.add(self.filter.dw, np.sum(self.x * dn, axis=0))
        self.filter.db += dn

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

    def __init__(self, layers):

        self.l_info  = []   # All the layers type and shape
        self.layers  = []   # All the layers of neurones
        self.filters = {}   # All the filter for the Conv neurones

        # Parse the layers indicated and create the neural network
        for l in range(len(layers)):

            type = layers[l][0]
            data = layers[l][1]

            # Add a new layer
            self.l_info.append(layers[l])
            self.layers.append([])

            ll = len(self.layers)-1  # last layer index
            pl = len(self.layers)-2  # previous layer index

            if type == "Conv":

                (depth, inp, f, p, s, alpha) = data

                # Determine the number of neurones
                n1, n2 = self.nbr_neurones(inp, f, p, s)

                # Each depth of the layer has one filter which is common to all
                # the neurones of the depth. The bias is part of the class Filter
                self.filters[ll] = []

                # Create the neurones and add them to the layer
                for d in range(depth):

                    # Add depth to the layer
                    self.layers[ll].append([])

                    # Add filter to the layer for the new depth
                    self.filters[ll].append(Filter(f, alpha))

                    for i in range(n1):

                        self.layers[ll][d].append([])

                        for y in range(n2):

                            # Create new Conv neurone
                            self.layers[ll][d][i].append(Conv(inp, \
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

                (inp, pool, s) = data

                # Determine the number of neurones
                n1, n2 = self.nbr_neurones(inp, pool, (0,0), s)

                # Create the neurones and add them to the layer
                for d in range(len(self.layers[pl])):

                    # Add depth to the layer
                    self.layers[ll].append([])

                    for i in range(n1):

                        self.layers[ll][d].append([])

                        for y in range(n2):

                            # Create MaxPool neurone
                            self.layers[ll][d][i].append(MaxPool(pool))

            elif type == "FC":

                # Get number of neurones
                n, alpha = data

                inp = len(self.layers[pl]) *\
                      len(self.layers[pl][0]) *\
                      len(self.layers[pl][0][0])

                # Create the full Connected layer
                for _ in range(n):

                    # Create New Neurone
                    self.layers[ll].append(FC(inp, alpha))

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

        return int(((w[0] - f[0] + 2 * p[0]) / s[0] + 1)), \
               int(((w[1] - f[1] + 2 * p[1]) / s[1] + 1))

    def train(self, x, t):

        self.run(x)

        nbr_layers = len(self.layers)

        # Training
        for l in range(nbr_layers-1, -1, -1):

            # Last layer (Softmax) -> training with expected output
            if self.l_info[l][0]  == "Softmax":
                self.layers[l].train(t)

            elif self.l_info[l][0]  == "Conv":

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

            elif self.l_info[l][0]  == "ReLu":

                if self.l_info[l+1][0] == "MaxPool":

                    dn = np.zeros((len(self.layers[l]), \
                                    len(self.layers[l][0]), \
                                    len(self.layers[l][0][0])))

                    _nd  = len(self.layers[l+1])
                    _nn1 = len(self.layers[l+1][0])
                    _nn2 = len(self.layers[l+1][0][0])

                    for nd in range(_nd):
                        for nn1 in range(_nn1):
                            for nn2 in range(_nn2):

                                (_, f, s)   = self.l_info[l+1][1]
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

                elif self.l_info[l+1][0] ==  "Conv":

                    dn = np.zeros((len(self.layers[l][0]), \
                                   len(self.layers[l][0][0])))

                    _nd  = len(self.layers[l+1])
                    _nn1 = len(self.layers[l+1][0])
                    _nn2 = len(self.layers[l+1][0][0])

                    for nd in range(_nd):
                        for nn1 in range(_nn1):
                            for nn2 in range(_nn2):

                                (_, _, f, p, s, _) = self.l_info[l+1][1]
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

            elif self.l_info[l][0]  == "MaxPool":

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

            elif self.l_info[l][0]  == "FC":

                if self.l_info[l+1][0]  == "Softmax":

                    # Get the backpropagation value from the previous layer
                    dn = self.layers[l+1].dn

                    for n in range(len(self.layers[l])):
                        self.layers[l][n].back(dn[n])

                elif self.l_info[l+1][0]  == "FC":

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

        nbr_layers = len(self.layers)

        # Parse each layer
        for l in range(nbr_layers):

            # Layer's input value
            if l == 0:
                l_in = x
            else:
                l_in = l_out

            # Layer's output value
            l_out = np.zeros(np.asarray(self.layers[l]).shape)

            if self.l_info[l][0] == "Softmax":
                l_out = self.layers[l].run(l_in)

            elif self.l_info[l][0] == "FC":
                for n in range(len(self.layers[l])):
                    l_out[n] = self.layers[l][n].run(l_in.reshape(-1))

            else:   # Conv, ReLu, MaxPool

                _d  = len(self.layers[l])
                _n1 = len(self.layers[l][0])
                _n2 = len(self.layers[l][0][0])

                for d in range(_d):
                    for n1 in range(_n1):
                        for n2 in range(_n2):

                            if self.l_info[l][0] == "Conv":

                                # Determine the input matrix of the neurone
                                (_, _, f, p, s, _) = self.l_info[l][1]
                                (x1, x2) = (n1*s[0], n1*s[0]+f[0])
                                (y1, y2) = (n2*s[1], n2*s[1]+f[1])

                                nIn = l_in[:, x1:x2, y1:y2]

                            elif self.l_info[l][0] == "ReLu":
                                nIn = l_in[d, n1, n2]

                            elif self.l_info[l][0] == "MaxPool":

                                # Determine the input matrix of the neurone
                                (_, f, s)   = self.l_info[l][1]
                                (x1, x2) = (n1*s[0], n1*s[0]+f[0])
                                (y1, y2) = (n2*s[1], n2*s[1]+f[1])

                                nIn = l_in[d, x1:x2, y1:y2]

                            l_out[d, n1, n2] = self.layers[l][d][n1][n2].run(nIn)

        self.y = l_out
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
        data = data.reshape(nbr_images, w * w) / np.float32(255)

    # Extract label from MNIST dataset
    with gzip.open("dataset/train-labels-idx1-ubyte.gz") as bytestream:

        # Throw away the 8 first bytes
        bytestream.read(8)

        # Get the labels
        buf = bytestream.read(nbr_images)

        # Encode the labels
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    # Create the convolutional neural network, w = input
    # Conv    -> (depth, input, filter, padding, stride, alpha)
    # ReLu    -> ()
    # MaxPool -> (input, filter, stride)
    # FC      -> (input, alpha)
    # Softmax -> ()

    # (w - f + 2 * p) / s + 1 = number of neurones along each row

    l = [
            ("Conv",     (8, (28,28), (5,5), (0,0), (1,1), 1e-4)),
            ("ReLu",     ()),
            ("Conv",     (8, (24,24), (5,5), (0,0), (1,1), 1e-4)),
            ("ReLu",     ()),
            ("MaxPool",  ((20,20), (2,2), (2,2))),
            ("FC",       (10, 1e-4)),
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

    cnn = CNN(l)

    # Train the convolutional neural network
    for i in range(len(data)):
        result = cnn.train(data[i].reshape(d, w, w), output[labels[i]])

        print("Training set:" + str(i))
        print("Answer:", np.argmax(result))
        print("Right Answer:", labels[i])
        print("Prob:", np.max(result))
        print()

    cnt_tested  = 0.0
    cnt_correct = 0.0

    # Train the convolutional neural network
    for i in range(50):
        result = cnn.run(data[i].reshape(d, w, w))

        cnt_tested += 1.0

        if np.argmax(result) == labels[i]:
            cnt_correct += 1.0

        print("Answer:", np.argmax(result))
        print("Right Answer:", labels[i])
        print("Prob:", np.max(result))
        print("Ratio:", (cnt_correct/cnt_tested)*100.0, "%")
        print()
