# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

# TODO: use cross entropy as loss function, work with softmax

import sys
import gzip
import time
import pygame
import pickle
import numba
from numba import cuda
import threading
import numpy as np
from copy import deepcopy, copy
from random import randint
from math import sqrt, exp
from pygame.locals import *
import matplotlib.pyplot as plt
from collections import defaultdict

# Rectified linear Unit Neurone ================================================

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

    # ln2 = cuda.blockDim.x
    # ln1 = cuda.gridDim.y

    if x[d, n1, n2] <= 0:
        dn0[d, n1, n2] = 0
    else:
        dn0[d, n1, n2] = dn1[n1, n2]

@cuda.jit
def relu_run(x, y):

    d  = cuda.blockIdx.x
    n1 = cuda.blockIdx.y
    n2 = cuda.threadIdx.x

    y[d, n1, n2] = max(0, x[d, n1, n2])

# If next layer is DQN or FC
@cuda.jit
def relu2_back(dn1, dn0, x):

    n = cuda.threadIdx.x

    temp = 0

    for i in range(len(dn1)):
        temp += dn1[i, n]

    if x[n] <= 0:
        dn0[n] = 0
    else:
        dn0[n] = temp

@cuda.jit
def relu2_run(x, y):

    n = cuda.threadIdx.x

    y[n] = max(0, x[n])

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
#    = dE/dy * dy/dxi
#    = dE/dy * wi

@cuda.jit
def fc_back(dn1, dn0, x, w, b, alpha):

    n = cuda.threadIdx.x

    temp = dn1[n]

    # for i in range(len(dn1)):
    #     temp += dn1[i, n]

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
# Delta node: dE/dx
# -----------
# dnij = full_conv(dE/dyij * w.T)
#
# dn00 = dE/dy00 * w00
# dn01 = dE/dy00 * w01 + dE/dy01 * w00
# dn02 =               + dE/dy01 * w01
# dn10 = dE/dy00 * w10 + dE/dy10 * w00
# dn11 = dE/dy00 * w11 + dE/dy10 * w01 + dE/dy01 * w10 + dE/dy11 * w00
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

    # b[d, n1, n2] -= alpha[0] * dn1[d, n1, n2]
    # for _d in range(d):
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
                sum += x[_d, n1*s1 + _f1, n2*s2 + _f2] * w[d, _f1, _f2]

    # Multiply the input by the filter
    # y[d, n1, n2] = b[d, n1, n2] + sum
    y[d, n1, n2] = b[d] + sum

# Deep Q Neurone ===============================================================

# x     = input
# z     = adder output
# Q     = output
# w     = weight
# b     = bias
# a     = action defined for the neurone
# t     = training data
# alpha = learning factor
# gamma = discount factor
# dw    = delta weight
# db    = delta bias
# dn    = delta node used for backpropagation
# selected = if the action was selected

# Forward:
# --------
# Q = sum{i, xi * wi} + b
#
# Error:
# ------
# Qplus = max(a, Q(s', a))
# E = 1/2 * (R + gamma * Qplus - Q)^2
#
# Train:
# ------
# dE/dwi = dE/dQ * dQ/dwi
# dE/dQ = 1/2 * 2 * (R + gamma * QPlus - Q) * -1
# dQ/dwi = xi
# -> wi = wi + alpha * (R + gamma * Qplus - Q) * 1 * xi
#
# Delta node:
# -----------
# dni = dE/dxi = dE/dQ * dQ/dxi
#    = (R + gamma * QPlus - Q) * -1 * wi

@cuda.jit
def dqn_train(dn, x, Q, Qplus, r, w, b, alpha, gamma, sel):

    n = cuda.threadIdx.x

    # dE/dQ
    temp = (r[n] + gamma[0] * Qplus[n] - Q[n]) * -1 #(-Q[n])

    # No update if the action was not selected
    # dw = alpha * dE/dQ * dQ/dwi
    b[n] -= alpha[0] * temp * sel[n]
    for i in range(len(x)):
            w[n, i] -= alpha[0] * temp * x[i] * sel[n]
            dn[n, i] = temp * w[n, i] * sel[n]

@cuda.jit
def dqn_run(x, Q, w, b):

    n = cuda.threadIdx.x

    sum = 0

    for i in range(len(x)):
        sum += x[i] * w[n, i]

    # Adder
    Q[n] = b[n] + sum

# Convolutional Neural Network =================================================

# create mutex global to save the DQL with pickle
layers_mutex = threading.Lock()
rply_mutex = threading.Lock()

# Let the replay dictionary outside of the object to not save it with pickle
rply_save = defaultdict()

class DQL:
    """Convolutional Neural Network"""

    # f = filter / kernel size
    # s = Stride
    # p = Padding
    # d = depth
    # w = width of features map
    # n = number of neurones / neurones
    # k = nbr filters for a Conv layer
    # l = {"Conv": (w, f, p, s), "ReLu": (), "MaxPool": ..., "FC":...}
    # r = rewards
    # x = state
    # a = action
    # e = experiance replay

    a = 0       # Save action for experiance replay
    x = None    # Save state for experiance replay

    rply_limit    = 100
    rply_samples  = 10000
    rply_stop     = False
    rply_cnt      = 0

    update_counter = 0
    explore_cnt = 0

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
                    'dn': np.zeros((inp[0], inp[1])),
                    # 'w':  np.random.uniform(w[0], w[1], (d, f[0], f[1])),
                    # # 'b':  np.full((d, n1, n2), b),
                    # 'b':  np.full(d, b),
                    'w':  np.random.normal(0, w, (d, f[0], f[1])),
                    'b':  np.random.normal(0, w, d),
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

            elif type == "ReLu2":

                # Each ReLu node is directly connected to the previous one.
                # Hence, there is as much ReLu neurone than the number of
                # neurones in the previous layer.
                n = prev_out

                self.layers[ll] = \
                {
                    'x':  np.zeros(n),
                    'y':  np.zeros(n),
                    'dn': np.zeros(n),
                    'shape': n
                }

            elif type == "FC":

                # Get number of neurones
                n, alpha, w, b = data

                if self.l_info[pl][0] == "FC" or self.l_info[pl][0] == "ReLu2":
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
                    # 'w':  np.random.uniform(w[0], w[1], (n, inp)),
                    # 'b':  np.full(n, b),
                    'w':  np.random.normal(0, w, (n, inp)),
                    'b':  np.random.normal(0, w, n),
                    'alpha': alpha,
                    'shape': n
                }

            elif type == "DQN":

                # Get number of neurones
                n, alpha, gamma, w, b = data

                if self.l_info[pl][0] == "FC" or self.l_info[pl][0] == "ReLu2":
                    inp = prev_out
                else:
                    d, n1, n2 = prev_out
                    inp = d * n1 * n2

                self.layers[ll] = \
                {
                    'x':  np.zeros(inp),
                    'Q':  np.zeros(n),
                    'dn': np.zeros((n, inp)),
                    # 'w':  np.random.normal(w[0], w[1], (n, inp)),
                    # 'b':  np.full(n, b),
                    'w':  np.random.normal(0, w, (n, inp)),
                    'b':  np.random.normal(0, w, n),
                    'alpha': alpha,
                    'gamma': gamma,
                    'shape': n,
                }

        # Copy the network for experiance replay
        self.rply_layers  = deepcopy(self.layers)

        # Create the array for the output values
        self.Q = None

    def __call__(self, x, r, first, train):

        str = cuda.stream()

        if train:
            return self.train(x, r, str, first)
        else:
            self.a, _ = self.run(x, self.layers, str)
            return self.a

    def nbr_neurones(self, w, f, p, s):

        # (w - f + 2 * p) / s + 1 = number of neurones along each row
        # w = input width
        # f = filter dimension
        # p = padding
        # s = stride

        return ((w[0] - f[0] + 2 * p[0]) / s[0] + 1), \
               ((w[1] - f[1] + 2 * p[1]) / s[1] + 1)

    def experiance_replay(self):

        global rply_save
        global rply_mutex
        global layers_mutex

        global DQN_ALPHA
        global CONV_ALPHA
        global FC_ALPHA

        str = cuda.stream()

        exp = 0
        replay_length = 0

        self.rply_stop = False

        while not self.rply_stop:

            while replay_length == 0 and not agent.rply_stop:
                rply_mutex.acquire()
                replay_length = len(rply_save)
                rply_mutex.release()

            if agent.rply_stop:
                return

            rply_mutex.acquire()
            if len(rply_save) == 0:
                rply_mutex.release()
                continue

            # Get an experiance randomly
            exp = np.random.randint(low=0, high=len(rply_save))
            x, a, r, xPlus = rply_save.keys()[exp]
            rply_save[x ,a, r, xPlus] += 1
            rply_mutex.release()

            print "Experiance chosen: ", exp, self.rply_cnt

            # Calculate the next Q with explorationsel
            # NOTE: It seems than my grahpic card is not capable to execute
            #       the kernel in the same time than the main thread.
            if xPlus is None:
                QPlus = np.zeros(self.rply_layers[len(self.rply_layers)-1]['shape'])
            else:
                layers_mutex.acquire()
                _, QPlus = self.run(np.asarray(xPlus), self.rply_layers, str)
                layers_mutex.release()
            layers_mutex.acquire()
            _, Q = self.run(np.asarray(x), self.rply_layers, str)
            layers_mutex.release()

            print "Q:    ", Q, r
            print "QPlus:", QPlus, r

            nbrLayers = len(self.rply_layers)

            # Training
            for l in range(nbrLayers-1, -1, -1):

                # Last layer (DQN)
                if self.l_info[l][0]  == "DQN":

                    r_tmp     = np.zeros(self.rply_layers[l]['shape'])
                    QPlus_tmp = np.zeros(self.rply_layers[l]['shape'])
                    sel_tmp   = np.zeros(self.rply_layers[l]['shape'])

                    r_tmp[a] = r
                    sel_tmp[a] = 1
                    QPlus_tmp[a] = max(QPlus)

                    # DEBUG:
                    self.rply_layers[l]['alpha'] = DQN_ALPHA

                    # Move data from host to device
                    d_dn    = cuda.to_device(self.rply_layers[l]['dn'], str)
                    d_x     = cuda.to_device(self.rply_layers[l]['x'], str)
                    d_Q     = cuda.to_device(self.rply_layers[l]['Q'], str)
                    d_w     = cuda.to_device(self.rply_layers[l]['w'], str)
                    d_b     = cuda.to_device(self.rply_layers[l]['b'], str)
                    d_alpha = cuda.to_device(self.rply_layers[l]['alpha'], str)
                    d_gamma = cuda.to_device(self.rply_layers[l]['gamma'], str)
                    d_r     = cuda.to_device(r_tmp, str)
                    d_Qplus = cuda.to_device(QPlus_tmp, str)
                    d_sel   = cuda.to_device(sel_tmp, str)

                    # One thread per element, assuming no more than n < 1024
                    n = self.rply_layers[l]['shape']
                    bl = 1
                    th = n

                    dqn_train[bl, th, str](d_dn, d_x, d_Q, d_Qplus, d_r, \
                                            d_w, d_b, d_alpha, d_gamma, d_sel)

                    # Move data from device to host
                    d_dn.copy_to_host(self.rply_layers[l]['dn'], str)
                    d_w.copy_to_host(self.rply_layers[l]['w'], str)
                    d_b.copy_to_host(self.rply_layers[l]['b'], str)

                elif self.l_info[l][0]  == "Conv":

                    d, n1, n2  = self.rply_layers[l]['shape']
                    _, f1, f2  = self.rply_layers[l]['w'].shape
                    inp1, inp2 = self.rply_layers[l]['dn'].shape

                    dw   = np.zeros((d, n1, n2, f1, f2))
                    dn0 = np.zeros((d, n1, n2, inp1, inp2))

                    # DEBUG:
                    self.rply_layers[l]['alpha'] = CONV_ALPHA

                    # Move data from host to device
                    d_dw    = cuda.to_device(dw, str)
                    d_dn0   = cuda.to_device(dn0, str)
                    d_dn1   = cuda.to_device(self.rply_layers[l+1]['dn'], str)
                   # d_dn0   = cuda.to_device(self.rply_layers[l]['dn'], str)
                    d_x     = cuda.to_device(self.rply_layers[l]['x'], str)
                    d_w     = cuda.to_device(self.rply_layers[l]['w'], str)
                    d_b     = cuda.to_device(self.rply_layers[l]['b'], str)
                    d_s     = cuda.to_device(self.rply_layers[l]['s'], str)
                    d_f     = cuda.to_device(self.rply_layers[l]['f'], str)
                    d_alpha = cuda.to_device(self.rply_layers[l]['alpha'], str)

                    # One thread per element, assuming no more than n2 < 1024
                    bl = (d, n1)
                    th = n2

                    conv_back[bl, th, str](d_dn1, d_dn0, d_x, d_w, d_dw, d_b, \
                                                        d_s, d_f, d_alpha)

                    # Move data from device to host
                    # d_dn0.copy_to_host(self.rply_layers[l]['dn'], str)
                    # d_w.copy_to_host(self.rply_layers[l]['w'], str)
                    d_dn0.copy_to_host(dn0, str)
                    d_dw.copy_to_host(dw, str)
                    # d_b.copy_to_host(self.rply_layers[l]['b'], str)

                    for _d in range(d):
                        self.rply_layers[l]['b'][_d] -= self.rply_layers[l]['alpha'] * \
                                        np.sum(self.rply_layers[l+1]['dn'][_d, : , :])

                    for _d in range(d):
                        for _f1 in range(f1):
                            for _f2 in range(f2):
                                self.rply_layers[l]['w'][_d, _f1, _f2] -= \
                                                np.sum(dw[_d, :, :, _f1, _f2])

                    if l == 0:
                        continue

                    self.rply_layers[l]['dn'] = np.zeros((inp1, inp2))

                    for _d in range(d):
                        for _n1 in range(n1):
                            for _n2 in range(n2):
                                self.rply_layers[l]['dn'] += dn0[_d, _n1, _n2]

                    # for _i1 in range(inp1):
                    #     for _i2 in range(inp2):
                    #         self.rply_layers[l]['dn'][_i1, _i2] += \
                    #                             np.sum(dn0[:, :, :, _i1, _i2])

                elif self.l_info[l][0]  == "ReLu":

                    # Move data from host to device
                    d_dn1   = cuda.to_device(self.rply_layers[l+1]['dn'], str)
                    d_dn0   = cuda.to_device(self.rply_layers[l]['dn'], str)
                    d_x     = cuda.to_device(self.rply_layers[l]['x'], str)

                    # One thread per element, assuming no more than n2 < 1024
                    d, n1, n2 = self.rply_layers[l]['shape']
                    bl = (d, n1)
                    th = n2

                    relu_back[bl, th, str](d_dn1, d_dn0, d_x)

                    # Move data from device to host
                    d_dn0.copy_to_host(self.rply_layers[l]['dn'], str)

                elif self.l_info[l][0]  == "ReLu2":

                    # Move data from host to device
                    d_dn1   = cuda.to_device(self.rply_layers[l+1]['dn'], str)
                    d_dn0   = cuda.to_device(self.rply_layers[l]['dn'], str)
                    d_x     = cuda.to_device(self.rply_layers[l]['x'], str)

                    # One thread per element, assuming no more than n2 < 1024
                    n = self.rply_layers[l]['shape']
                    bl = 1
                    th = n

                    relu2_back[bl, th, str](d_dn1, d_dn0, d_x)

                    # Move data from device to host
                    d_dn0.copy_to_host(self.rply_layers[l]['dn'], str)

                elif self.l_info[l][0]  == "FC":

                    # DEBUG:
                    self.rply_layers[l]['alpha'] = FC_ALPHA

                    # Move data from host to device
                    d_dn1   = cuda.to_device(self.rply_layers[l+1]['dn'], str)
                    d_dn0   = cuda.to_device(self.rply_layers[l]['dn'], str)
                    d_x     = cuda.to_device(self.rply_layers[l]['x'], str)
                    d_w     = cuda.to_device(self.rply_layers[l]['w'], str)
                    d_b     = cuda.to_device(self.rply_layers[l]['b'], str)
                    d_alpha = cuda.to_device(self.rply_layers[l]['alpha'], str)

                    # One thread per element, assuming no more than n < 1024
                    n = self.rply_layers[l]['shape']
                    bl = 1
                    th = n

                    fc_back[bl, th, str](d_dn1, d_dn0, d_x, d_w, d_b, \
                                                                    d_alpha)

                    # Move data from device to host
                    d_dn0.copy_to_host(self.rply_layers[l]['dn'], str)
                    d_w.copy_to_host(self.rply_layers[l]['w'], str)
                    d_b.copy_to_host(self.rply_layers[l]['b'], str)

            self.update_counter += 1
            print "Update Counter:", self.update_counter

            # Set an update during the next training session
            if self.rply_cnt >= self.rply_limit:

                print "NEW WEIGHT LOADED --------------------------------------"
                self.rply_cnt  = 0

                layers_mutex.acquire()
                self.layers = deepcopy(self.rply_layers)
                layers_mutex.release()

                # # Reset dictionary of experiance
                # rply_mutex.acquire()
                # rply_save = defaultdict()
                # replay_length = 0
                # rply_mutex.release()

            else:
                self.rply_cnt += 1

    def train(self, x, r, str, first):

        global rply_save
        global rply_mutex
        global layers_mutex

        # Save experiance et = (st,at,rt,st+1)
        if not first and len(rply_save) < self.rply_samples:

            tuple_x = self.x.tolist()

            if x is not None:
                tuple_xPlus = x.tolist()

            # Convert the numpy array to tuple to save them in the dict
            for i in range(len(tuple_x)):
                tuple_x[i] = tuple(map(tuple, tuple_x[i]))

                if x is not None:
                    tuple_xPlus[i] = tuple(map(tuple, tuple_xPlus[i]))

            tuple_x = tuple(map(tuple, tuple_x))

            if x is not None:
                tuple_xPlus = tuple(map(tuple, tuple_xPlus))
            else:
                tuple_xPlus = None

            if (tuple_x, self.a, r, tuple_xPlus) not in rply_save:
                rply_mutex.acquire()
                rply_save[tuple_x, self.a, r, tuple_xPlus] = 0
                rply_mutex.release()
                print "NEW SAMPLES:", len(rply_save)

            # rply_mutex.acquire()
            # rply_save[tuple_x, self.a, r, tuple_xPlus] = 0
            # rply_mutex.release()

        # # Reset experiance dictionary
        # elif not first:
        #     rply_mutex.acquire()
        #     rply_save = defaultdict()
        #     replay_length = 0
        #     rply_mutex.release()

        # Select the next action
        self.x = x

        if x is None:
            return

        # Avoid the replay thread to upload the new weight during a pass
        layers_mutex.acquire()
        self.a, Q = self.run(x, self.layers, str, True)
        layers_mutex.release()

        # print
        # print x
        # print
        print "TRAIN:", Q

        return self.a

    def run(self, x, layers, str, explore=False):

        nbrLayers = len(layers)

        l_out = 0

        # Parse each layer
        for l in range(nbrLayers):

            # Layer's input value
            if l == 0:
                l_in = np.copy(x)
            else:
                l_in = np.copy(l_out)

            # Layer's output value
            l_out = np.zeros(layers[l]['shape'])

            if self.l_info[l][0] == "DQN":

                layers[l]['x'] = np.copy(l_in.reshape(-1))

                # Move data from host to device
                d_x = cuda.to_device(layers[l]['x'], str)
                d_Q = cuda.to_device(layers[l]['Q'], str)
                d_w = cuda.to_device(layers[l]['w'], str)
                d_b = cuda.to_device(layers[l]['b'], str)

                # One thread per element, assuming no more than n < 1024
                n = layers[l]['shape']
                bl = 1
                th = n

                dqn_run[bl, th, str](d_x, d_Q, d_w, d_b)

                # Move data from device to host
                d_Q.copy_to_host(layers[l]['Q'], str)
                l_out = np.copy(layers[l]['Q'])

            elif self.l_info[l][0] == "FC":

                layers[l]['x'] = np.copy(l_in.reshape(-1))

                # Move data from host to device
                d_x = cuda.to_device(layers[l]['x'], str)
                d_y = cuda.to_device(layers[l]['y'], str)
                d_w = cuda.to_device(layers[l]['w'], str)
                d_b = cuda.to_device(layers[l]['b'], str)

                # One thread per element, assuming no more than n < 1024
                n = layers[l]['shape']
                bl = 1
                th = n

                fc_run[bl, th, str](d_x, d_y, d_w, d_b)

                # Move data from device to host
                d_y.copy_to_host(layers[l]['y'], str)
                l_out = np.copy(layers[l]['y'])

            elif self.l_info[l][0] == "Conv":

                layers[l]['x'] = np.copy(l_in)

                # Move data from host to device
                d_x = cuda.to_device(layers[l]['x'], str)
                d_y = cuda.to_device(layers[l]['y'], str)
                d_w = cuda.to_device(layers[l]['w'], str)
                d_b = cuda.to_device(layers[l]['b'], str)
                d_s = cuda.to_device(layers[l]['s'], str)
                d_f = cuda.to_device(layers[l]['f'], str)

                # One thread per element, assuming no more than n2 < 1024
                d, n1, n2 = layers[l]['shape']
                bl = (d, n1)
                th = n2

                conv_run[bl, th, str](d_x, d_y, d_w, d_b, d_s, d_f)

                # Move data from device to host
                d_y.copy_to_host(layers[l]['y'], str)
                l_out = np.copy(layers[l]['y'])

            elif self.l_info[l][0] == "ReLu":

                layers[l]['x'] = np.copy(l_in)

                # Move data from host to device
                d_x = cuda.to_device(layers[l]['x'], str)
                d_y = cuda.to_device(layers[l]['y'], str)

                # One thread per element, assuming no more than n2 < 1024
                d, n1, n2 = layers[l]['shape']
                bl = (d, n1)
                th = n2

                relu_run[bl, th, str](d_x, d_y)

                # Move data from device to host
                d_y.copy_to_host(layers[l]['y'], str)
                l_out = np.copy(layers[l]['y'])

            elif self.l_info[l][0] == "ReLu2":

                layers[l]['x'] = np.copy(l_in)

                # Move data from host to device
                d_x = cuda.to_device(layers[l]['x'], str)
                d_y = cuda.to_device(layers[l]['y'], str)

                # One thread per element, assuming no more than n2 < 1024
                n = layers[l]['shape']
                bl = 1
                th = n

                relu2_run[bl, th, str](d_x, d_y)

                # Move data from device to host
                d_y.copy_to_host(layers[l]['y'], str)
                l_out = np.copy(layers[l]['y'])

        # Greedy selection, P(explore) to not choose the given action
        if explore:
            e = 5000.0 / (5000.0 + self.explore_cnt)
            greedy = np.full(len(l_out), e / (len(l_out)-1))
            greedy[np.argmax(l_out)] = 1.0 - e
            a = np.random.choice(len(l_out), 1, p=greedy.flatten())[0]
            self.explore_cnt += 1
            print "P(explore):", e
            return a, l_out
        else:
            return np.argmax(l_out), l_out
        # return np.argmax(l_out), l_out

# ------------------------------------------------------------------------------

# General Parameters
SQUARE_SIZE    = 10
WINDOW_WIDTH   = 10
WINDOW_HEIGHT  = 10
WINDOW_COLOR   = (0, 0, 0)

CONV_ALPHA  = 1e-8
FC_ALPHA    = 1e-6
DQN_ALPHA   = 1e-6
DQN_GAMMA   = 99e-1
TRAINING    = True

MOVE_REWARD_POS  = -1e-2 # Reward for good move
MOVE_REWARD_NEG  = -1e-2 # Reward for bad move
LOSE_REWARD      = -1e-1
FRUIT_REWARD     =  5e-1

ACTION_TIME      = 500 #ms

FRUIT_COLOR  = (255, 255, 255)

SNAKE_INIT_LENGTH = 5 # SQUARES
SNAKE_GROWING     = 1 # SQUARES
SNAKE_INIT_POSX   = WINDOW_WIDTH  / 2 # SQUARES
SNAKE_INIT_POSY   = WINDOW_HEIGHT / 2 # SQUARES
SNAKE_COLOR       = (255, 255, 255)
SNAKE_COLOR_HEAD  = (255, 255, 255)

INIT_DIRECTION = "UP"

DIRECTIONS = {
    "UP":    ( 0, -1),
    "DOWN":  ( 0,  1),
    "LEFT":  (-1,  0),
    "RIGHT": ( 1,  0),
}
                   # TURN: RIGHT=0  |   LEFT = 1   |  STRAIGHT = 2
DIRS_ACTION = {"UP":    [ "RIGHT",      "LEFT",         "UP"],
               "DOWN":  [ "LEFT",       "RIGHT",        "DOWN"],
               "LEFT":  [ "UP",         "DOWN",         "LEFT"],
               "RIGHT": [ "DOWN",       "UP",           "RIGHT"]}

EVENT = {
    "TIMER":    USEREVENT + 0,
    "START":    USEREVENT + 1,
}

pd = 0 # previous distance snake and fruit

class Snake:

    dir        = ""
    body       = []      # index 0 is the head of the snake
    snake_body = None
    snake_head   = None

    def __init__(self):

        self.dir = INIT_DIRECTION

        for i in range(SNAKE_INIT_LENGTH):
            x = SQUARE_SIZE * (SNAKE_INIT_POSX - DIRECTIONS[self.dir][0]*i)
            y = SQUARE_SIZE * (SNAKE_INIT_POSY - DIRECTIONS[self.dir][1]*i)

            self.body.append([x, y])

        self.snake_body = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
        self.snake_body.fill(SNAKE_COLOR)

        self.snake_head = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
        self.snake_head.fill(SNAKE_COLOR_HEAD)

    def reset(self):

        self.body = []

        self.dir = INIT_DIRECTION

        for i in range(SNAKE_INIT_LENGTH):
            x = SQUARE_SIZE * (SNAKE_INIT_POSX - DIRECTIONS[self.dir][0]*i)
            y = SQUARE_SIZE * (SNAKE_INIT_POSY - DIRECTIONS[self.dir][1]*i)

            self.body.append([x, y])

    def movement(self):

        # Move the snake
        for i in range(len(self.body)-1, 0, -1):
            self.body[i][0] = self.body[i-1][0]
            self.body[i][1] = self.body[i-1][1]

        x = DIRECTIONS[self.dir][0] * SQUARE_SIZE
        y = DIRECTIONS[self.dir][1] * SQUARE_SIZE

        self.body[0][0] += x
        self.body[0][1] += y

        # Check if the snake went out of the window
        if self.body[0][0] < 0:
            # --print "Snake left wall"
            return False

        elif self.body[0][0] >= WINDOW_WIDTH  * SQUARE_SIZE:
            # --print "Snake right wall"
            return False

        elif self.body[0][1] < 0:
            # --print "Snake bottom walls"
            return False

        elif self.body[0][1] >= WINDOW_HEIGHT * SQUARE_SIZE:
            # --print "Snake top wall"
            return False

        # Check if snake eats itself
        for i in range(len(self.body)-1):
            if self.body[0][0] == self.body[i+1][0] and\
               self.body[0][1] == self.body[i+1][1]:
               # --print "Snake ate itself"
               return False

        return True

    def checkWall(self):

        wall = [0,0,0]
        actions = DIRS_ACTION[self.dir]

        # Check for each possible next direction if a collision may occured
        for a in range(len(actions)):

            dir = DIRECTIONS[actions[a]]

            if self.body[0][0] + (dir[0] * SQUARE_SIZE) < 0\
            or self.body[0][0] + (dir[0] * SQUARE_SIZE) >= \
                    WINDOW_WIDTH * SQUARE_SIZE\
            or self.body[0][1] + (dir[1] * SQUARE_SIZE) < 0\
            or self.body[0][1] + (dir[1] * SQUARE_SIZE) >= \
                    WINDOW_HEIGHT * SQUARE_SIZE:
                wall[a] = 1

        return wall

    def checkBody(self):

        body_part = [0,0,0]
        actions = DIRS_ACTION[self.dir]

        # Check if snake eats itself for each next direction
        for a in range(len(actions)):

            dir = DIRECTIONS[actions[a]]

            for i in range(len(self.body)-1):
                    if  self.body[0][0] + (dir[0] * SQUARE_SIZE) == \
                            self.body[i+1][0]\
                    and self.body[0][1] + (dir[1] * SQUARE_SIZE) == \
                            self.body[i+1][1]:
                        body_part[a] = 1;

        return body_part

    def newDir(self, dir):
        self.dir = dir

    def getPosition(self):
        return (self.body[0][0], self.body[0][1])

    def getDir(self):
        return self.dir

    def growing(self):

        last  = self.body[len(self.body)-1] # last part of body
        bLast = self.body[len(self.body)-1] # Before last part of body

        dir = np.asarray(last) - np.asarray(bLast)

        self.body.append([last[0] + dir[0],
                          last[1] + dir[1]])

    def draw(self, w):
        for b in self.body:
            if b == self.body[0]:
                w.blit(self.snake_head, (b[0], b[1]))
            else:
                w.blit(self.snake_body, (b[0], b[1]))

class Fruit:

    posx      = 0
    posy      = 0
    snake     = None
    fruit_img = None

    def __init__(self, snake):

        self.snake = snake

        self.newPosition()

        self.fruit_img = pygame.Surface((SQUARE_SIZE,SQUARE_SIZE))
        self.fruit_img.fill(FRUIT_COLOR)

    def reset(self):
        self.newPosition()

    def newPosition(self):
        self.posx = SQUARE_SIZE * randint(0, WINDOW_WIDTH-1)
        self.posy = SQUARE_SIZE * randint(0, WINDOW_HEIGHT-1)

        # Check if the new fruit is on the position than the snake
        for i in range(len(self.snake.body)):
            if  self.snake.body[i][0] == self.posx and \
                self.snake.body[i][1] == self.posy:
                self.newPosition()

    def getPosition(self):
        return (self.posx, self.posy)

    def draw(self, w):
        w.blit(self.fruit_img, (self.posx, self.posy))

def getReward(snake, fruit):

    global pd

    # The rewards is positif if the snake is nearer from the fruit and negatif
    # if it is further from the fruit
    snake_pos = snake.getPosition()
    fruit_pos = fruit.getPosition()

    # Calcul the distance between fruit and snake
    d = sqrt((snake_pos[0] - fruit_pos[0])**2 +\
             (snake_pos[1] - fruit_pos[1])**2)

    # If new distance smaller than previous, positif reward
    if (pd - d) > 0:
        r = MOVE_REWARD_POS
    else:
        r = MOVE_REWARD_NEG

    # Save distance
    pd = d

    return r

def getState(snake, fruit):

    surface  = pygame.display.get_surface()
    # shape    = pygame.surfarray.pixels_red(surface).shape
    # state    = np.zeros((3, shape[0], shape[0]))
    # state[0] = pygame.surfarray.pixels_red(surface)
    # state[1] = pygame.surfarray.pixels_green(surface)
    # state[2] = pygame.surfarray.pixels_blue(surface)
    # state[0] = np.divide(state[0], np.max((np.max(state), 1)) * 100)

    surface  = pygame.display.get_surface()
    # shape    = pygame.surfarray.pixels_red(surface).shape
    state    = np.zeros((3, WINDOW_WIDTH, WINDOW_HEIGHT))
    color = [pygame.surfarray.pixels_red(surface),\
             pygame.surfarray.pixels_green(surface),\
             pygame.surfarray.pixels_blue(surface)]

    for i in range(3):
        for x in range(WINDOW_WIDTH):
            for y in range(WINDOW_HEIGHT):
                state[i, x, y] = color[i][x * SQUARE_SIZE][y * SQUARE_SIZE]

    return state

def run(agent, snake, fruit):

    action = "UP"
    state_cnt = 0
    first = True

    # Redraw snake and screen
    window.fill(WINDOW_COLOR)
    snake.draw(window)
    fruit.draw(window)

    # Update display on screen
    pygame.display.flip()

    # Get the initial state
    state = getState(snake, fruit)
    next  = agent(state, 0, True, TRAINING)
    action = DIRS_ACTION[action][next]

    print "state:", state_cnt, "action:", DIRS_ACTION[action][next]
    state_cnt += 1

    # Loop til doomsday
    while True:

        rewards = 0
        state = 0

        # Wait an event from pygame
        event = pygame.event.wait()

        # Close windows
        if event.type == QUIT:
            print("Quit")
            pygame.quit()

        # Timer to move
        elif event.type is not EVENT["TIMER"]:
            continue

        # Perform action recommended by the agent
        snake.newDir(action)

        # Move the snake and check if it touch itself or the wall
        if not snake.movement():
            # Update state rewards for the agent
            rewards += LOSE_REWARD

            # Redraw snake and screen
            window.fill(WINDOW_COLOR)
            snake.draw(window)
            fruit.draw(window)

            # Update display on screen
            pygame.display.flip()

            state = getState(snake, fruit)
            print "state:", state_cnt, "rewards:", rewards, "DEAD"
            state_cnt += 1

            # Add the final losing state
            agent(None, rewards, False, TRAINING)

            return False

        # Check if the snake has eaten a fruit
        if snake.getPosition() == fruit.getPosition():
            snake.growing()
            fruit.newPosition()

            rewards += FRUIT_REWARD

        # Rewards regarding the distance of the snake
        rewards += getReward(snake, fruit)

        # Redraw snake and screen
        window.fill(WINDOW_COLOR)
        snake.draw(window)
        fruit.draw(window)

        # Update display on screen
        pygame.display.flip()

        # Update state rewards for the agent
        state = getState(snake, fruit)
        next = agent(state, rewards, False, TRAINING)
        action = DIRS_ACTION[action][next]

        print "state cnt:", state_cnt, "action:", DIRS_ACTION[action][next], \
                                                            "rewards:", rewards
        state_cnt += 1

if __name__ == "__main__":

    # Create the Deep Q Learning, w = input
    # Conv    -> (depth, input, filter, padding, stride, alpha)
    # ReLu    -> ()
    # FC      -> (neurones, alpha, gamma)
    # DQN     -> (neurones, alpha, gamma)

    # (w - f + 2 * p) / s + 1 = number of neurones along each row

    in1 = (WINDOW_WIDTH, WINDOW_HEIGHT)
    d1 = 32
    f1 = (2, 2)
    s1 = (2, 2)
    p1 = (0, 0)
    w1 = 0.01
    b1 = 0.01

    in2 = (5, 5)
    d2 = 64
    f2 = (3, 3)
    s2 = (2, 2)
    p2 = (0, 0)
    w2 = 0.01
    b2 = 0.01

    in3 = (2, 2)
    d3 = 64
    f3 = (1, 1)
    s3 = (1, 1)
    p3 = (0, 0)
    w3 = 0.01
    b3 = 0.01

    n4 = 512
    w4 = 0.01
    b4 = 0.01

    n5 = 3
    w5 = 0.01
    b5 = 0.01

    l = [
            ("Conv",     (d1, in1, f1, p1, s1, CONV_ALPHA, w1, b1)),
            ("ReLu",     ()),
            ("Conv",     (d2, in2, f2, p2, s2, CONV_ALPHA, w2, b2)),
            ("ReLu",     ()),
            ("Conv",     (d3, in3, f3, p3, s3, CONV_ALPHA, w3, b3)),
            ("ReLu",     ()),
            ("FC",       (n4, FC_ALPHA, w4, b4)),
            ("ReLu2",    ()),
            ("DQN",      (n5, DQN_ALPHA, DQN_GAMMA, w5, b5)),
        ]

    try:
        agent = pickle.load(open('saving/dql', 'rb'))
        print "DQL reloaded"
    except:
        agent = DQL(l)
        print "DQL created"

    # Thread for experiance replay
    if TRAINING:
        replay_thread = threading.Thread(target=agent.experiance_replay)
        replay_thread.daemon = True
        replay_thread.start()

        agent.explore_cnt = 0

    # init Pygame
    pygame.init()

    # initialize window
    window = pygame.display.set_mode((WINDOW_WIDTH  * SQUARE_SIZE,
                                      WINDOW_HEIGHT * SQUARE_SIZE))
    window.fill(WINDOW_COLOR)

    # Title of the window
    pygame.display.set_caption('Snake')

    # create snake
    snake = Snake()

    # place one fruit
    fruit = Fruit(snake)

    # Init timer
    clk = pygame.time.Clock()
    pygame.time.set_timer(EVENT["TIMER"], ACTION_TIME)

    game_cnt = 0

    print("Game initialized")

    # Loop til doomsday
    while True:

        game_cnt += 1
        print "NEW GAME: " + str(game_cnt)

        # Run snake
        run(agent, snake, fruit)

        # create snake
        snake.reset()

        # place one fruit
        fruit.reset()

        # Redraw snake and screen
        window.fill(WINDOW_COLOR)
        snake.draw(window)
        fruit.draw(window)

        # Update display on screen
        pygame.display.flip()

        # Save the object each 100 trials
        if game_cnt % 100 == 0 and TRAINING:

            agent.rply_stop = True
            replay_thread.join()
            pickle.dump(agent, open('saving/dql','wb'))
            agent.rply_stop = False

            print "Agent saved"

            # Restart Thread
            replay_thread = threading.Thread(target=agent.experiance_replay)
            replay_thread.daemon = True
            replay_thread.start()
