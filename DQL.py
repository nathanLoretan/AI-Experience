# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

import sys
import gzip
import pygame
from CNN import *
import threading
import numpy as np
from copy import deepcopy, copy
from random import randint
from math import sqrt, exp
from pygame.locals import *
from collections import defaultdict

class DQN:
    """ Deep Q Neurone"""

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

    b = 0
    Q = 0

    dw = 0
    db = 0
    dn = 0

    def __init__(self, nbrIn, a, alpha=0.5, gamma=0.9):

        self.gamma    = gamma
        self.alpha    = alpha
        self.a        = a

        self.x = np.zeros(nbrIn)
        self.w = np.random.uniform(low=-1e-4, high=1e-4, size=nbrIn)

    def train(self, Qplus, r, selected):

        # Qplus = max(a, Q(s', a))
        # E = 1/2(Q - (R + gamma * Qplus))^2
        #
        # dE/dwi = dE/dQ * dQ/dwi
        # dE/dQ = (Q - (R + gamma * Qplus))
        # dQ/dwi = xi
        #
        # dni = dE/dxi = dE/dQ * dQ/dz * dz/dxi
        #    = (Q - (R + gamma * Qplus)) * wi

        # dE/dQ
        temp = (self.Q - (r + self.gamma * Qplus))

        #  dw = alpha * (Q - (R + gamma * Qplus)) * dQ/dwi
        self.dw = self.alpha * temp * self.x
        self.db = self.alpha * temp

        # No update if the action was not selected
        self.w = self.w - self.dw * selected
        self.b = self.b - self.db * selected

        self.dn = temp * self.w * selected

    def run(self, x):

        # Q = sum{i, xi * wi} + b

        # Save for backpropagation
        self.x = x

        # Adder
        self.Q = self.b + np.sum(self.x * self.w)

        return self.Q

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

    a = 0
    x = None

    replay = defaultdict()
    replay_cnt = 0
    replay_limit = 30
    replay_stop = False

    def __init__(self, layers, inShape, explore):

        self.explore = explore

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

                if self.lInfo[pl][0] == "FC":
                    nbrIn = len(self.layers[pl])
                else:
                    nbrIn = len(self.layers[pl]) *\
                            len(self.layers[pl][0]) *\
                            len(self.layers[pl][0][0])


                # Create the full Connected layer
                for _ in range(n):

                    # Create New Neurone
                    self.layers[ll].append(FC(nbrIn, alpha))

            elif type == "DQN":

                # Get number of neurones
                n, alpha, gamma = data

                if self.lInfo[pl][0] == "FC":
                    nbrIn = len(self.layers[pl])
                else:
                    nbrIn = len(self.layers[pl]) *\
                            len(self.layers[pl][0]) *\
                            len(self.layers[pl][0][0])

                # Create the full Connected layer
                for _ in range(n):

                    # Create New Neurone
                    self.layers[ll].append(DQN(nbrIn, alpha, gamma))

        # Copy the network for experiance replay
        self.replay_layers  = deepcopy(self.layers)
        self.replay_filters = deepcopy(self.filters)

        # Update correctly the filter of the convolutional layers
        for l in range(len(layers)):

            if type == "Conv":

                _d  = len(self.replay_layers[l])
                _n1 = len(self.replay_layers[l][0])
                _n2 = len(self.replay_layers[l][0][0])

                for d in range(_d):
                    for n1 in range(_n1):
                        for n2 in range(_n2):
                                self.replay_layers[ll][d][i].filter = \
                                        self.replay_filters[ll][d]

        # Creat the array for the output values
        self.Q = None

        # create mutex
        self.replay_mutex = threading.Lock()
        self.layers_mutex = threading.Lock()

        # Start thread for experiance replay
        self.replay_thread = threading.Thread(target=self.experiance_replay)
        self.replay_thread.start()

    def __call__(self, x, r, first, train):

        if train:
            return self.train(x, r, first)
        else:
            return self.run(x, self.layers)

    def nbr_neurones(self, w, f, p, s):

        # (w - f + 2 * p) / s + 1 = number of neurones along each row
        # w = input width
        # f = filter dimension
        # p = padding
        # s = stride

        return ((w[0] - f[0] + 2 * p[0]) / s[0] + 1), \
               ((w[1] - f[1] + 2 * p[1]) / s[1] + 1)

    def experiance_replay(self):

        replay_length = 0

        while replay_length == 0:
            self.replay_mutex.acquire()
            replay_length = len(self.replay)
            self.replay_mutex.release()

        while not self.replay_stop:

            # Select the experiance to replay based on the number of time it has
            # been replayed
            self.replay_mutex.acquire()
            e    = np.zeros(len(self.replay))
            prob = np.zeros(len(self.replay))

            # Calculate exponentiel of each value
            for i in range(len(self.replay)):
        	       e[i] = np.exp(self.replay.values()[i], dtype=np.float)

            prob = e / sum(e)

            exp = np.random.choice(len(self.replay), 1, p=prob.flatten())[0]
            x ,a, r, xPlus = self.replay.keys()[exp]
            self.replay[x ,a, r, xPlus] += 1
            self.replay_mutex.release()

            # # Get one experiance in the replay list
            # self.replay_mutex.acquire()
            # x ,a, r, xPlus =\
            #         self.replay[np.random.randint(low=0, high=len(self.replay))]
            # self.replay_mutex.release()

            # Calculate the next Q with exploration
            _, QPlus = self.run(np.asarray(xPlus), self.replay_layers)
            _, Q     = self.run(np.asarray(x), self.replay_layers)

            print "Qplus:", QPlus
            print "Q:    ", Q

            nbrLayers = len(self.replay_layers)

            # Training
            for l in range(nbrLayers-1, 0, -1):

                # Last layer (DQN)
                if l == nbrLayers-1:

                    for n in range(len(self.replay_layers[l])):

                        if n == a:
                            self.replay_layers[l][n].train(max(QPlus), r, 1)
                        else:
                            self.replay_layers[l][n].train(0, 0, 0)

                # BackPropagation
                else:

                    if self.lInfo[l][0]  == "Conv":

                        if self.lInfo[l+1][0]  == "FC" or \
                           self.lInfo[l+1][0]  == "DQN":
                            pass

                        elif self.lInfo[l+1][0]  == "ReLu":

                            _d  = len(self.replay_layers[l])
                            _n1 = len(self.replay_layers[l][0])
                            _n2 = len(self.replay_layers[l][0][0])

                            for d in range(_d):
                                for n1 in range(_n1):
                                    for n2 in range(_n2):

                                        dn = self.replay_layers[l+1][d][n1][n2].dn
                                        self.replay_layers[l][d][n1][n2].back(dn)

                                # Update the weight of the filter
                                self.replay_filters[l][d].update()

                        elif self.lInfo[l+1][0]  == "MaxPool": # TODO
                            pass

                        elif self.lInfo[l+1][0]  == "Conv": # TODO
                            pass

                    elif self.lInfo[l][0]  == "ReLu":

                        if self.lInfo[l+1][0]  == "FC" or \
                           self.lInfo[l+1][0]  == "DQN":

                            _d  = len(self.replay_layers[l])
                            _n1 = len(self.replay_layers[l][0])
                            _n2 = len(self.replay_layers[l][0][0])

                            for d in range(_d):
                                for n1 in range(_n1):
                                    for n2 in range(_n2):

                                        # Number of neurones in the next layer
                                        _nn = len(self.replay_layers[l+1])

                                        # Create/reset the lists
                                        dn = np.zeros(_nn)

                                        # nn = node of next layer
                                        for nn in range(_nn):
                                            n = d * _n1 * _n2 + n1 * _n2 + n2
                                            dn[nn] = self.replay_layers[l+1][nn].dn[n]

                                        self.replay_layers[l][d][n1][n2].back(dn, "FC")

                        elif self.lInfo[l+1][0] == "MaxPool":

                            dn = np.zeros((len(self.replay_layers[l]), \
                                            len(self.replay_layers[l][0]), \
                                            len(self.replay_layers[l][0][0])))

                            _nd  = len(self.replay_layers[l+1])
                            _nn1 = len(self.replay_layers[l+1][0])
                            _nn2 = len(self.replay_layers[l+1][0][0])

                            for nd in range(_nd):
                                for nn1 in range(_nn1):
                                    for nn2 in range(_nn2):

                                        (_, f, s)   = self.lInfo[l+1][1]
                                        (x1, x2) = (nn1*s[0], nn1*s[0]+f[0])
                                        (y1, y2) = (nn2*s[1], nn2*s[1]+f[1])

                                        temp = self.replay_layers[l+1][nd][nn1][nn2].dn
                                        dn[d, x1:x2, y1:y2] = temp

                            _d  = len(self.replay_layers[l])
                            _n1 = len(self.replay_layers[l][0])
                            _n2 = len(self.replay_layers[l][0][0])

                            for d in range(_d):
                                for n1 in range(_n1):
                                    for n2 in range(_n2):
                                        self.replay_layers[l][d][n1][n2].back(dn[d][n1][n2])

                        elif self.lInfo[l+1][0] == "Conv": # TODO padding

                            dn = np.zeros((len(self.replay_layers[l][0]), \
                                            len(self.replay_layers[l][0][0])))

                            _nd  = len(self.replay_layers[l+1])
                            _nn1 = len(self.replay_layers[l+1][0])
                            _nn2 = len(self.replay_layers[l+1][0][0])

                            for nd in range(_nd):
                                for nn1 in range(_nn1):
                                    for nn2 in range(_nn2):

                                        (_, _, f, p, s, _) = self.lInfo[l+1][1]
                                        (x1, x2) = (nn1*s[0], nn1*s[0]+f[0])
                                        (y1, y2) = (nn2*s[1], nn2*s[1]+f[1])

                                        temp = self.replay_layers[l+1][nd][nn1][nn2].dn
                                        dn[x1:x2, y1:y2] = np.add(dn[x1:x2, y1:y2],\
                                                                  temp)

                            _d  = len(self.replay_layers[l])
                            _n1 = len(self.replay_layers[l][0])
                            _n2 = len(self.replay_layers[l][0][0])

                            for d in range(_d):
                                for n1 in range(_n1):
                                    for n2 in range(_n2):
                                        self.replay_layers[l][d][n1][n2].back(dn[n1][n2])

                    elif self.lInfo[l][0]  == "MaxPool":

                        if self.lInfo[l+1][0]  == "FC" or \
                           self.lInfo[l+1][0]  == "DQN":

                            _d  = len(self.replay_layers[l])
                            _n1 = len(self.replay_layers[l][0])
                            _n2 = len(self.replay_layers[l][0][0])

                            for d in range(_d):
                                for n1 in range(_n1):
                                    for n2 in range(_n2):

                                        # Number of neurones in the next layer
                                        _nn = len(self.replay_layers[l+1])

                                        # Create/reset the lists
                                        dn = np.zeros(_nn)

                                        # nn = node of next layer
                                        for nn in range(_nn):
                                            n = d * _n1 * _n2 + n1 * _n2 + n2
                                            dn[nn] = self.replay_layers[l+1][nn].dn[n]

                                        self.replay_layers[l][d][n1][n2].back(dn)

                        elif self.lInfo[l+1][0] == "MaxPool": # TODO
                            pass

                        elif self.lInfo[l+1][0] == "Conv":  # TODO
                            pass

                    elif self.lInfo[l][0]  == "FC":

                        if self.lInfo[l+1][0]  == "FC" or \
                           self.lInfo[l+1][0]  == "DQN":

                            # Number of neurones in the next layer
                            nLen = len(self.replay_layers[l+1])

                            for n in range(len(self.replay_layers[l])):

                                # Create/reset the lists
                                dn = np.zeros(nLen)

                                # nn = node of next layer
                                for nn in range(nLen):
                                    dn[nn] = self.replay_layers[l+1][nn].dn[n]

                                self.replay_layers[l][n].back(dn, "FC")

            # Set an update during the next training session
            if self.replay_cnt >= self.replay_limit:

                print "NEW WEIGHT LOAD"
                self.replay_cnt  = 0

                self.layers_mutex.acquire()
                self.layers  = copy(self.replay_layers)
                self.filters = copy(self.replay_filters)
                self.layers_mutex.release()
            else:
                self.replay_cnt += 1

    def train(self, x, r, first):

        # Save experiance et = (st,at,rt,st+1)
        if not first:

            tuple_x     = self.x.tolist()
            tuple_xPlus = x.tolist()

            # Convert the numpy array to tuple to save them in the dict
            for i in range(len(tuple_x)):
                tuple_x[i] = tuple(map(tuple, tuple_x[i]))
                tuple_xPlus[i] = tuple(map(tuple, tuple_xPlus[i]))
            tuple_x = tuple(map(tuple, tuple_x))
            tuple_xPlus = tuple(map(tuple, tuple_xPlus))

            if (tuple_x, self.a, r, tuple_xPlus) not in self.replay:
                self.replay_mutex.acquire()
                self.replay[tuple_x, self.a, r, tuple_xPlus] = 0
                self.replay_mutex.release()

        # Select the next action
        self.x = x

        # Avoid the replay thread to upload the new weight during a pass
        self.layers_mutex.acquire()
        self.a, _ = self.run(x, self.layers)
        self.layers_mutex.release()

        return self.a

    def run(self, x, layers):

        nbrLayers = len(layers)

        # Parse each layer
        for l in range(nbrLayers):

            # Layer's input value
            if l == 0:
                lIn = x
            else:
                lIn = lOut

            # Layer's output value
            lOut = np.zeros(np.asarray(layers[l]).shape)

            if self.lInfo[l][0] == "DQN":
                for n in range(len(layers[l])):
                    lOut[n] = layers[l][n].run(lIn.reshape(-1))

            elif self.lInfo[l][0] == "FC":
                for n in range(len(layers[l])):
                    lOut[n] = layers[l][n].run(lIn.reshape(-1))

            else:   # Conv, ReLu, MaxPool

                _d  = len(layers[l])
                _n1 = len(layers[l][0])
                _n2 = len(layers[l][0][0])

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

                            lOut[d, n1, n2] = layers[l][d][n1][n2].run(nIn)

        # greedy selection
        # TODO:
        greedy = np.full(len(lOut), self.explore/(len(lOut)-1))
        greedy[np.argmax(lOut)] = 1.0 - self.explore

        return np.random.choice(len(lOut), 1, p=greedy.flatten())[0], lOut

# ------------------------------------------------------------------------------

VERBOSE = True

# General Parameters
SQUARE_SIZE    = 10
WINDOW_WIDTH   = 10
WINDOW_LENGTH  = 10
WINDOW_COLOR   = (0, 0, 0)#(50, 50, 50)

#dql1 = 1e-8, 1e-8, 5e-2, 5e-1

DQL_EXPLORE = 9e-1
CONV_ALPHA  = 1e-6
FC_ALPHA    = 1e-6
DQN_ALPHA   = 5e-1
DQN_GAMMA   = 5e-1
TRAINING    = True

MOVE_REWARD_POS  =  0.1 # Reward for good move
MOVE_REWARD_NEG  = -0.1 # Reward for bad move
LOSE_REWARD      = -1
FRUIT_REWARD     = 10

ACTION_TIME      = 1 #ms

FRUIT_COLOR  = (255, 0, 0)#(225, 80, 50)

SNAKE_INIT_LENGTH = 5 # SQUARES
SNAKE_GROWING     = 1 # SQUARES
SNAKE_INIT_POSX   = WINDOW_WIDTH  / 2 # SQUARES
SNAKE_INIT_POSY   = WINDOW_LENGTH / 2 # SQUARES
SNAKE_COLOR       = (100, 0, 0) #(226, 226, 226)
SNAKE_COLOR_HEAD  = (150, 0, 0)

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

# MOVE = ["TURN_RIGHT", "TURN_LEFT", "STAY_STRAIGHT"]

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
            print "Snake left wall"
            return False

        elif self.body[0][0] >= WINDOW_WIDTH  * SQUARE_SIZE:
            print "Snake right wall"
            return False

        elif self.body[0][1] < 0:
            print "Snake bottom walls"
            return False

        elif self.body[0][1] >= WINDOW_LENGTH * SQUARE_SIZE:
            print "Snake top wall"
            return False

        # Check if snake eats itself
        for i in range(len(self.body)-1):
            if self.body[0][0] == self.body[i+1][0] and\
               self.body[0][1] == self.body[i+1][1]:
               print "Snake ate itself"
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
                    WINDOW_LENGTH * SQUARE_SIZE:
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
        self.posy = SQUARE_SIZE * randint(0, WINDOW_LENGTH-1)

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
    d = sqrt(abs(snake_pos[0] - fruit_pos[0])**2 +\
             abs(snake_pos[1] - fruit_pos[1])**2)

    # If new distance smaller than previous, positif reward
    if (pd - d) >= 0:
        r = MOVE_REWARD_POS
    else:
        r = MOVE_REWARD_NEG

    # Save distance
    pd = d

    return r

def getState(snake, fruit):

    surface  = pygame.display.get_surface()
    shape    = pygame.surfarray.pixels_red(surface).shape
    state    = np.zeros((1, shape[0], shape[0]))
    state[0] = pygame.surfarray.pixels_red(surface)
    # state[0] = np.divide(pygame.surfarray.pixels_red(surface, \
    #                      np.max((np.max(state), 1))))

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
            agent(state, rewards, False, TRAINING)

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
    # Conv    -> (depth, inShape, fShape, padding, stride, alpha)
    # ReLu    -> ()
    # MaxPool -> (inShape, fShape, stride)
    # DQN     -> (nbrN, alpha, gamma)

    # (w - f + 2 * p) / s + 1 = number of neurones along each row

    w1 = (WINDOW_WIDTH * SQUARE_SIZE, WINDOW_WIDTH * SQUARE_SIZE)
    f1 = (4, 4)
    s1 = (2, 2)
    p1 = (0, 0)
    d1 = 4

    w2 = (49, 49)
    f2 = (7, 7)
    s2 = (6, 6)
    p2 = (0, 0)
    d2 = 4

    w3 = (8, 8)
    f3 = (2, 2)
    s3 = (2, 2)
    p3 = (0, 0)
    d3 = 4

    # w4 = (12, 12)
    # f4 = (4, 4)
    # s4 = (2, 2)
    # p4 = (0, 0)
    # d4 = 64

    n5 = 16
    # n6 = 512
    # n7 = 256
    n8 = 3


    l = [
            ("Conv",     (d1, w1, f1, p1, s1, CONV_ALPHA)),
            ("ReLu",     ()),
            ("Conv",     (d2, w2, f2, p2, s2, CONV_ALPHA)),
            ("ReLu",     ()),
            ("Conv",     (d3, w3, f3, p3, s3, CONV_ALPHA)),
            ("ReLu",     ()),
            # ("Conv",     (d4, w4, f4, p4, s4, CONV_ALPHA)),
            # ("ReLu",     ()),
            ("FC",       (n5, FC_ALPHA)),
            # ("FC",       (n6, FC_ALPHA)),
            # ("FC",       (n7, FC_ALPHA)),
            ("DQN",      (n8, DQN_ALPHA, DQN_GAMMA)),
        ]

    agent = DQL(l, (1, WINDOW_WIDTH  * SQUARE_SIZE,
                       WINDOW_LENGTH * SQUARE_SIZE), DQL_EXPLORE)
    print "DQL created"
    print("AI initialized")

    # init Pygame
    pygame.init()

    # initialize window
    window = pygame.display.set_mode((WINDOW_WIDTH  * SQUARE_SIZE,
                                      WINDOW_LENGTH * SQUARE_SIZE))
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

    print("Game initialized")

    game_cnt = 0

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
