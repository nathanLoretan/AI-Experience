# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

import sys
import gym
import time
import math
import numpy as np
from math import sqrt, exp
from random import randint
from collections import defaultdict

def relu(x):
    x[x < 0] = 0
    return x

def sigmoid(x):
      return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
     e = np.exp(x, dtype=np.float)
     return e / np.sum(e)

class ActorCritic():

    def __init__(self, out, inp, alpha=0.9, beta=0.9, gamma=0.9):

        self.gamma    = gamma
        self.alpha    = alpha
        self.beta     = beta
        self.s        = None
        self.a        = None
        self.l        = None
        self.t        = np.random.uniform(-1.0, 1.0, (out, inp))  # policy weights
        self.w        = np.random.uniform(-1.0, 1.0, (out, inp))  # Q-Value weights
        self.b        = np.zeros(out)

    def reset(self):

        self.s  = None
        self.a  = None
        self.r  = None
        self.l  = None

        self.epi = []

    def __call__(self, s2, r):

        # Calculate the policy and select the next action
        l2 = np.dot(self.t, s2) + self.b
        pi2 = softmax(l2)
        a2  = np.random.choice(len(pi2), 1, p=pi2)[0]

        if self.s is not None:

            a     = self.a
            s     = self.s
            l, pi = self.l

            q  = np.dot(self.w, s)
            q2 = np.dot(self.w, s2)

            grad = np.zeros(len(pi))
            grad[a] = (l[a] - np.dot(l, pi)) * q[a]

            self.t += self.alpha * grad[np.newaxis, :].T * np.asarray(s)
            self.b += self.alpha * grad

            grad = np.zeros(len(pi))
            grad[a] = (r + self.gamma * q2[a2] - q[a])
            self.w += self.beta * grad[np.newaxis, :].T * np.asarray(s)

        self.a = a2
        self.s = s2
        self.l = (l2, pi2)

        return self.a

# STATE:
# ------
# [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

# ACTIONS:
# --------
# +1, 0 or -1

EPSIODES = 10000
ALPHA = 0.000001
BETA  = 0.000001
GAMMA = 0.95

if __name__ == "__main__":

    env = gym.make('MountainCar-v0')
    agent = ActorCritic(3, 2, ALPHA, BETA, GAMMA)

    score = 0.0

    for e in range(EPSIODES):

        t = 0
        r = 0
        R = 0
        s = env.reset()

        prev = s[0]
        pos = s[0]
        vel = s[1]

        while True:

            env.render()

            if vel == 0:
                vel = s[1]

            a = agent((s[0], s[1]*1000), abs(s[1])-0.07)

            prev= s[0]

            s, r, done, info = env.step(a)

            R += r

            if done:
                if t+1 >= 200:
                    agent((s[0], s[1]*1000), abs(s[1])-0.07)
                else:
                    agent((s[0], s[1]*1000), 10)

                score += R
                print "Episode {} finished after {} timesteps, score {}"\
                                                .format(e+1, t+1, score/(e+1))
                break

            t += 1

        agent.reset()
