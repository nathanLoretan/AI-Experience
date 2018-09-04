# -*- coding: utf-8 -*-
"""
Nathan Loretan
Convolutional Neural Network with parallel programming
04.09.18
"""

import gzip
import pickle
import numpy as np
import numba
from numba import cuda

# Rectified l_inear Unit Neurone ===============================================

# x  = input, real number
# y  = output, real number
# dn = delta node used for backpropagation

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

# If next layer is Conv
@cuda.jit
def relu_back(dn1, dn0, x):

    d  = cuda.blockIdx.x
    n1 = cuda.blockIdx.y
    n2 = cuda.threadIdx.x

    ln2 = cuda.blockDim.x
    ln1 = cuda.gridDim.y

    if x[d, n1, n2] < 0:
        dn0[d, n1, n2] = 0
    else:
        dn0[d, n1, n2] = dn1[n1, n2]

# If next layer is MaxPool
@cuda.jit
def relu2_back(dn1, dn0, x):

    d  = cuda.blockIdx.x
    n1 = cuda.blockIdx.y
    n2 = cuda.threadIdx.x

    ln2 = cuda.blockDim.x
    ln1 = cuda.gridDim.y

    if x[d, n1, n2] < 0:
        dn0[d, n1, n2] = 0
    else:
        dn0[d, n1, n2] =  dn1[d, n1, n2]

@cuda.jit
def relu_run(x, y):

    d  = cuda.blockIdx.x
    n1 = cuda.blockIdx.y
    n2 = cuda.threadIdx.x

    y[d, n1, n2] = max(0, x[d, n1, n2])

# Fully Connected Neurone ======================================================

# x     = input, vector
# y     = output, real number
# b     = bias
# w     = weight
# dn    = delta node used for backpropagation
# alpha = learning factor

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
# dE/dy = sum{n, dE/dx+1nj}         -> l+1 = FC or DQN
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
#     = dE/dy * dy/dxi
#     = dE/dy * wi

# If next layer is another Software
@cuda.jit
def fc_back(dn1, dn0, x, w, b, alpha):

    n = cuda.threadIdx.x

    temp = dn1[n]

    b[n] -= alpha[0] * temp

    for i in range(len(x)):
        w[n, i] -= alpha[0] * temp * x[i]
        dn0[n, i] = temp * w[n, i]

# If next layer is another FC
@cuda.jit
def fc2_back(dn1, dn0, x, w, b, alpha):

    n = cuda.threadIdx.x

    temp = 0

    for i in range(len(dn1)):
        temp += dn1[i, n]

    b[n] -= alpha[0] * temp

    for i in range(len(x)):
        w[n, i] -= alpha[0] * temp * x[i]
        dn0[n, i] = temp * w[n, i]

@cuda.jit
def fc_run(x, y, w, b):

    n = cuda.threadIdx.x

    sum = 0

    for i in range(len(x)):
        sum += x[i] * w[n, i]

    # Adder
    y[n] = b[n] + sum

# Convolutional Neurone ========================================================

# x  = input, matrix
# y  = output, real number
# w  = weight
# b  = bias
# dw = delta weight
# db    = delta bias
# dn = delta node for back propagation
# alpha = learning factor

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

@cuda.jit
def conv_back(dn1, dn0, x, w, dw, b, s, f, alpha):

    d  = cuda.blockIdx.x
    n1 = cuda.blockIdx.y
    n2 = cuda.threadIdx.x

    # s = stride
    # f = filter
    s1, s2 = s
    f1, f2 = f

    for _d in range(len(x)):
        for _f1 in range(f1):
            for _f2 in range(f2):
                dw[d, n1, n2, _f1, _f2] = \
                    alpha[0] * dn1[d, n1, n2] * x[_d, n1*s1 + _f1, n2*s2 + _f2]

                dn0[d, n1, n2, n1*s1 + _f1, n2*s2 + _f2] = \
                                                dn1[d, n1, n2] * w[d, _f1, _f2]

@cuda.jit
def conv_run(x, y, w, b, s, f):

    d  = cuda.blockIdx.x
    n1 = cuda.blockIdx.y
    n2 = cuda.threadIdx.x

    # s = stride
    # f = filter
    s1, s2 = s
    f1, f2 = f

    sum = 0

    # d of x is not necessarily d of y
    for _d in range(len(x)):
        for _f1 in range(f1):
            for _f2 in range(f2):
                sum += x[_d, n1*s1+_f1, n2*s2+_f2] * w[d, _f1, _f2]

    y[d, n1, n2] = b[d] + sum

# Softmax Neurone ==============================================================

# x  = input, vector
# y  = output, vector
# t  = training data
# dn = delta node used for back propagation

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

def softmax_train(y, t):
    return y - t

def softmax_run(x):

    y = np.zeros(x.shape)
    e = np.zeros(x.shape)
    max = np.max(x)

    # Calculate exponentiel of each value
    for i in range(len(x)):
	       e[i] = np.exp(x[i], dtype=np.float)

    y = e / np.sum(e)

    return y

# MaxPool Neurone ==============================================================

# x  = input, matrix
# y  = output, real number
# i  = keep the index of the max value, used for backpropagation
# dn = delta node used for backpropagation

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

@cuda.jit
def max_back(dn1, dn0, i, p):

    d  = cuda.blockIdx.x
    n1 = cuda.blockIdx.y
    n2 = cuda.threadIdx.x

    ln2 = cuda.blockDim.x
    ln1 = cuda.gridDim.y

    # p = pool
    # i = index
    p1, p2 = p
    i1, i2 = i[(d, n1, n2)]

    sum = 0

    for y in range(len(dn1)):
        sum += dn1[y, d * ln1 * ln2 + n1 * ln2 + n2]

    for _p1 in range(p1):
        for _p2 in range(p2):
            dn0[d, n1*p1 + _p1, n2*p2 + _p2] = 0

    dn0[d, n1*p1 + i1, n2*p2 + i2] = sum

@cuda.jit
def max_run(x, y, p, i):

    d  = cuda.blockIdx.x
    n1 = cuda.blockIdx.y
    n2 = cuda.threadIdx.x

    # p = pool
    p1, p2 = p

    max = 0

    for _p1 in range(p1):
        for _p2 in range(p2):

            if _p1 == 0 and _p2 == 0:
                max = x[d, n1 + _p1, n2 + _p2]
                i[d, n1, n2] = (_p1, _p2)

            elif max < x[d, n1 + _p1, n2 + _p2]:
                max = x[d, n1 + _p1, n2 + _p2]
                i[d, n1, n2] = (_p1, _p2)

    y[d, n1, n2] = max

class CNN:
    """Convolutional Neural Network"""

    # f = filter / kernel size
    # s = Stride
    # p = Padding or pool
    # d = depth
    # w = width of features map
    # n = number of neurones / neurones
    # k = nbr filters for a Conv layer
    # l = {"Conv": (w, f, p, s), "ReLu": (), "MaxPool": ..., "FC":...}

    def __init__(self, layers):

        self.l_info   = []   # All the layers type and shape
        self.layers  = []   # All the layers of neurones

        prev_out = 0

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

                d, inp, f, p, s, alpha, w, b = data

                # Determine the number of neurones
                n1, n2 = self.nbr_neurones(inp, f, p, s)

                prev_out = (d, n1, n2)

                # Each depth of the layer has one filter which is common to all
                # the neurones of the depth. The bias is part of the class Filter
                self.layers[ll] = \
                {
                    'x':  np.zeros((d, inp[0], inp[1])),
                    'y':  np.zeros((d, n1, n2)),
                    'dn': np.zeros((inp[0], inp[0])),
                    'w':  np.random.uniform(w[0], w[1], (d, f[0], f[1])),
                    'b':  np.full((d), b),
                    's':  s,
                    'f':  f,
                    'alpha': alpha,
                    'shape': (d, n1, n2)
                }

            elif type == "ReLu":

                # Each ReLu node is directly connected to the previous one.
                # Hence, there is as much ReLu neurone than the number of
                # neurones in the previous layer.
                d, n1, n2 = prev_out

                self.layers[ll] = \
                {
                    'x':  np.zeros((d, n1, n2)),
                    'y':  np.zeros((d, n1, n2)),
                    'dn': np.zeros((d, n1, n2)),
                    'shape': (d, n1, n2)
                }

            elif type == "MaxPool":

                d, inp, p = data

                # Determine the number of neurones
                n1, n2 = self.nbr_neurones(inp, p, (0,0), p)

                prev_out = (d, n1, n2)

                # Each depth of the layer has one filter which is common to all
                # the neurones of the depth. The bias is part of the class Filter
                self.layers[ll] = \
                {
                    'x':  np.zeros((d, inp[0], inp[1])),
                    'y':  np.zeros((d, n1, n2)),
                    'i':  np.zeros((d, n1, n2), dtype=(np.int32, 2)),
                    'dn': np.zeros((d, inp[0], inp[1])),
                    'p':  p,
                    'shape': (d, n1, n2)
                }

            elif type == "FC":

                # Get number of neurones
                n, alpha, w, b = data

                if self.l_info[pl][0] == "FC":
                    inp = prev_out
                else:
                    d, n1, n2 = prev_out
                    inp = d * n1 * n2

                prev_out = n

                self.layers[ll] = \
                {
                    'x':  np.zeros(inp),
                    'y':  np.zeros(n),
                    'dn': np.zeros((n, inp)),
                    'w':  np.random.uniform(w[0], w[1], (n, inp)),
                    'b':  np.full(n, b),
                    'alpha': alpha,
                    'shape': n
                }

            elif type == "Softmax":

                n = prev_out

                self.layers[ll] = \
                {
                    'x':  np.zeros(n),
                    'y':  np.zeros(n),
                    'dn': np.zeros(n),
                    'shape': n,
                }

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
        for l in range(nbrLayers-1, -1, -1):

            # Last layer (Softmax) -> training with expected output
            if self.l_info[l][0]  == "Softmax":
                self.layers[l]['dn'] = softmax_train(self.layers[l]['y'], t)

            elif self.l_info[l][0]  == "Conv":

                d, n1, n2  = self.layers[l]['shape']
                _, f1, f2  = self.layers[l]['w'].shape
                inp1, inp2 = self.layers[l]['dn'].shape

                dw  = np.zeros((d, n1, n2, f1, f2))
                dn0 = np.zeros((d, n1, n2, inp1, inp2))

                # Move data from host to device
                d_dw    = cuda.to_device(dw)
                d_dn0   = cuda.to_device(dn0)
                d_dn1   = cuda.to_device(self.layers[l+1]['dn'])
                d_x     = cuda.to_device(self.layers[l]['x'])
                d_w     = cuda.to_device(self.layers[l]['w'])
                d_b     = cuda.to_device(self.layers[l]['b'])
                d_s     = cuda.to_device(self.layers[l]['s'])
                d_f     = cuda.to_device(self.layers[l]['f'])
                d_alpha = cuda.to_device(self.layers[l]['alpha'])

                # One thread per element, assuming no more than n2 < 1024
                bl = (d, n1)
                th = n2

                conv_back[bl, th](d_dn1, d_dn0, d_x, d_w, d_dw, d_b, \
                                                    d_s, d_f, d_alpha)

                # Move data from device to host
                d_dn0.copy_to_host(dn0)
                d_dw.copy_to_host(dw)

                for _d in range(d):
                    self.layers[l]['b'][_d] -= self.layers[l]['alpha'] * \
                                    np.sum(self.layers[l+1]['dn'][_d, : , :])

                for _d in range(d):
                    for _f1 in range(f1):
                        for _f2 in range(f2):
                            self.layers[l]['w'][_d, _f1, _f2] -= \
                                                np.sum(dw[_d, :, :, _f1, _f2])

                if l == 0:
                    continue

                self.layers[l]['dn'] = np.zeros(self.layers[l]['dn'].shape)

                for _d in range(d):
                    for _n1 in range(n1):
                        for _n2 in range(n2):
                            self.layers[l]['dn'] += dn0[_d, _n1, _n2]

            elif self.l_info[l][0]  == "ReLu":

                # Move data from host to device
                d_dn1   = cuda.to_device(self.layers[l+1]['dn'])
                d_dn0   = cuda.to_device(self.layers[l]['dn'])
                d_x     = cuda.to_device(self.layers[l]['x'])

                # One thread per element, assuming no more than n2 < 1024
                d, n1, n2 = self.layers[l]['shape']
                bl = (d, n1)
                th = n2

                if self.l_info[l+1][0] == "Conv":
                    relu_back[bl, th](d_dn1, d_dn0, d_x)
                elif self.l_info[l+1][0] == "MaxPool":
                    relu2_back[bl, th](d_dn1, d_dn0, d_x)

                # Move data from device to host
                d_dn0.copy_to_host(self.layers[l]['dn'])

            elif self.l_info[l][0]  == "MaxPool":

                # Move data from host to device
                d_dn1   = cuda.to_device(self.layers[l+1]['dn'])
                d_dn0   = cuda.to_device(self.layers[l]['dn'])
                d_i     = cuda.to_device(self.layers[l]['i'])
                d_p     = cuda.to_device(self.layers[l]['p'])

                # One thread per element, assuming no more than n < 1024
                n = self.layers[l]['shape']
                bl = 1
                th = n

                max_back[bl, th](d_dn1, d_dn0, d_i, d_p)

                # Move data from device to host
                d_dn0.copy_to_host(self.layers[l]['dn'])

            elif self.l_info[l][0]  == "FC":

                # Move data from host to device
                d_dn1   = cuda.to_device(self.layers[l+1]['dn'])
                d_dn0   = cuda.to_device(self.layers[l]['dn'])
                d_x     = cuda.to_device(self.layers[l]['x'])
                d_w     = cuda.to_device(self.layers[l]['w'])
                d_b     = cuda.to_device(self.layers[l]['b'])
                d_alpha = cuda.to_device(self.layers[l]['alpha'])

                # One thread per element, assuming no more than n < 1024
                n = self.layers[l]['shape']
                bl = 1
                th = n

                fc_back[bl, th](d_dn1, d_dn0, d_x, d_w, d_b, d_alpha)

                # Move data from device to host
                d_dn0.copy_to_host(self.layers[l]['dn'])
                d_w.copy_to_host(self.layers[l]['w'])
                d_b.copy_to_host(self.layers[l]['b'])

        return self.y

    def run(self, x):

        nbrLayers = len(self.layers)

        l_out = 0

        # Parse each layer
        for l in range(nbrLayers):

            # Layer's input value
            if l == 0:
                l_in = np.copy(x)
            else:
                l_in = np.copy(l_out)

            # Layer's output value
            l_out = np.zeros(self.layers[l]['shape'])

            if self.l_info[l][0] == "Softmax":

                self.layers[l]['x'] = np.copy(l_in)
                self.layers[l]['y'] = softmax_run(l_in)
                l_out = self.layers[l]['y']

            elif self.l_info[l][0] == "FC":

                self.layers[l]['x'] = np.copy(l_in.reshape(-1))

                # Move data from host to device
                d_x = cuda.to_device(self.layers[l]['x'])
                d_y = cuda.to_device(self.layers[l]['y'])
                d_w = cuda.to_device(self.layers[l]['w'])
                d_b = cuda.to_device(self.layers[l]['b'])

                # One thread per element, assuming no more than n < 1024
                n = self.layers[l]['shape']
                bl = 1
                th = n

                fc_run[bl, th](d_x, d_y, d_w, d_b)

                # Move data from device to host
                d_y.copy_to_host(self.layers[l]['y'])
                l_out = np.copy(self.layers[l]['y'])

            elif self.l_info[l][0] == "Conv":

                self.layers[l]['x'] = np.copy(l_in)

                # Move data from host to device
                d_x = cuda.to_device(self.layers[l]['x'])
                d_y = cuda.to_device(self.layers[l]['y'])
                d_w = cuda.to_device(self.layers[l]['w'])
                d_b = cuda.to_device(self.layers[l]['b'])
                d_s = cuda.to_device(self.layers[l]['s'])
                d_f = cuda.to_device(self.layers[l]['f'])

                # One thread per element, assuming no more than n2 < 1024
                d, n1, n2 = self.layers[l]['shape']
                bl = (d, n1)
                th = n2

                conv_run[bl, th](d_x, d_y, d_w, d_b, d_s, d_f)

                # Move data from device to host
                d_y.copy_to_host(self.layers[l]['y'])
                l_out = np.copy(self.layers[l]['y'])

            elif self.l_info[l][0] == "ReLu":

                self.layers[l]['x'] = np.copy(l_in)

                # Move data from host to device
                d_x = cuda.to_device(self.layers[l]['x'])
                d_y = cuda.to_device(self.layers[l]['y'])

                # One thread per element, assuming no more than n2 < 1024
                d, n1, n2 = self.layers[l]['shape']
                bl = (d, n1)
                th = n2

                relu_run[bl, th](d_x, d_y)

                # Move data from device to host
                d_y.copy_to_host(self.layers[l]['y'])
                l_out = np.copy(self.layers[l]['y'])

            elif self.l_info[l][0] == "MaxPool":

                self.layers[l]['x'] = np.copy(l_in)

                # Move data from host to device
                d_x = cuda.to_device(self.layers[l]['x'])
                d_y = cuda.to_device(self.layers[l]['y'])
                d_i = cuda.to_device(self.layers[l]['i'])
                d_p = cuda.to_device(self.layers[l]['p'])

                # One thread per element, assuming no more than n2 < 1024
                d, n1, n2 = self.layers[l]['shape']
                bl = (d, n1)
                th = n2

                max_run[bl, th](d_x, d_y, d_p, d_i)

                # Move data from device to host
                d_y.copy_to_host(self.layers[l]['y'])
                d_i.copy_to_host(self.layers[l]['i'])
                l_out = np.copy(self.layers[l]['y'])

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
    # Conv    -> (depth, input, filter, padding, stride, alpha, weights, bias)
    # ReLu    -> ()
    # MaxPool -> (depth, input, pool)
    # FC      -> (input, alpha, weights, bias)
    # Softmax -> ()

    # (w - f + 2 * p) / s + 1 = number of neurones along each row

    l = [
            ("Conv",     (8, (28,28), (5,5), (0,0), (1,1), 1e-7, (-0.01, 0.01), 0)),
            ("ReLu",     ()),
            ("Conv",     (8, (24,24), (5,5), (0,0), (1,1), 1e-7, (-0.01, 0.01), 0)),
            ("ReLu",     ()),
            ("MaxPool",  (8, (20,20), (2,2))),
            ("FC",       (10, 1e-2, (-0.1, 0.1), 0)),
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
        cnn = pickle.load(file('saving/cnn-v2'))
        print "CNN reloaded"
    except:
        cnn = CNN(l)
        pickle.dump(cnn, file('saving/cnn-v2','w'))

    # Train the convolutional neural network
    for i in range(len(data)):
        result = cnn.train(data[i].reshape(d, w, w), output[labels[i]])

        print "Training set:" + str(i)
        print "Answer:", np.argmax(result)
        print "Right Answer:", labels[i]
        print "Prob:", np.max(result)
        print

    pickle.dump(cnn, file('saving/cnn-v2','w'))

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
