# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

# TODO: Use cross entropy as loss function, with softmax

import sys
import gym
import time
import math
import pickle
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
     # e = np.exp(x-np.max(x), dtype=np.float)
     e = np.exp(x-np.max(x), dtype=np.float)
     return e / np.sum(e)

rply = []

class A2C():

    def __init__(self, out, inp, batch, alpha=0.9, beta=0.9, gamma=0.9, exp_factor=0):

        self.gamma      = gamma
        self.alpha      = alpha
        self.beta       = beta
        self.s          = None
        self.a          = None
        self.l          = None
        # self.w          =  np.random.normal(0, 0.1, (out, 256))  # Policy
        # self.t          = [np.random.normal(0, 0.1, (256, inp)),
        #                    np.random.normal(0, 0.1, (out, 256))]  # Q-Value
        self.w          =  np.random.randn(out, 256) * np.sqrt(2.0 / out)  # Policy
        self.t          = [np.random.randn(256, inp) * np.sqrt(2.0 / 256),
                           np.random.randn(out, 256) * np.sqrt(2.0 / out)]  # Q-Value
        self.dw         =  np.zeros(self.w.shape)
        self.dt         = [np.zeros(self.t[0].shape),
                           np.zeros(self.t[1].shape)]
        self.batch      = batch
        self.batch_cnt  = 0
        self.exp_cnt    = 1
        self.exp_factor = exp_factor

    def replay(self):

        # Get an experiance randomly
        exp = np.random.randint(low=0, high=len(rply))
        r, s, a, s2, a2 = rply[exp]

        l0 = np.dot(self.t[0], s)
        l0[l0 < 0] = 0
        l1 = np.dot(self.t[1], l0)
        pi = softmax(l1)

        l02 = np.dot(self.t[0], s2)
        l02[l02 < 0] = 0
        l12 = np.dot(self.t[1], l02)
        pi2 = softmax(l12)

        v  = np.dot(self.w, l0)
        v2 = np.dot(self.w, l02)

        # A(s,a) = Q(s,a) - V(s,a) with Q(s,a) = r + gamma * V(s,a)
        adv = (r + self.gamma * v2[a2] - v[a])

        grad = np.zeros(len(pi))
        grad[a] = (l1[a] - np.dot(l1, pi)) * adv
        dn = grad[np.newaxis, :].T * self.t[1]
        self.dt[1] += self.alpha * grad[np.newaxis, :].T * l0

        grad = np.zeros(len(pi))
        grad[a] = (r + self.gamma * v2[a2] - v[a])
        self.dw += self.beta * grad[np.newaxis, :].T * l0

        # grad = np.zeros(len(self.t[0]))
        # for i in range(len(self.t[0])):
        #     grad[i] = np.sum(dn[:, i])
        grad = np.sum(dn, axis=0)
        grad[l0 <= 0] = 0
        self.dt[0] += self.alpha * grad[np.newaxis, :].T * np.asarray(s)

    def reset(self):

        self.s  = None
        self.a  = None
        self.r  = None
        self.l  = None

    def __call__(self, s2, r):

        global rply

        # Calculate the policy and select the next action
        l02 = np.dot(self.t[0], s2)
        l02[l02 < 0] = 0
        l12 = np.dot(self.t[1], l02)
        pi2 = softmax(l12)
        a2  = np.random.choice(len(pi2), 1, p=pi2)[0]

        # Exploration
        e = self.exp_factor / (self.exp_factor + self.exp_cnt)
        greedy = np.full(len(pi2), e / (len(pi2)-1))
        greedy[a2] = 1.0 - e
        a2 = np.random.choice(len(pi2), 1, p=greedy)[0]
        self.exp_cnt += 1

        if self.s is not None:

            a     = self.a
            s     = self.s
            l0, l1, pi = self.l

            # print l1, pi, r

            # exp = (r, s, a, s2, a2)
            #
            # if exp not in enumerate(rply):
            #      rply.append(exp)
            #
            # self.replay()

            v  = np.dot(self.w, l0)
            v2 = np.dot(self.w, l02)

            # A(s,a) = Q(s,a) - V(s,a) with Q(s,a) = r + gamma * V(s,a)
            adv = (r + self.gamma * v2[a2] - v[a])

            grad = np.zeros(len(pi))
            grad[a] = (l1[a] - np.dot(l1, pi)) * adv
            dn = grad[np.newaxis, :].T * self.t[1]
            self.dt[1] += self.alpha * grad[np.newaxis, :].T * l0

            grad = np.zeros(len(pi))
            grad[a] = (r + self.gamma * v2[a2] - v[a])
            self.dw += self.beta * grad[np.newaxis, :].T * l0

            # grad = np.zeros(len(self.t[0]))
            # for i in range(len(self.t[0])):
            #     grad[i] = np.sum(dn[:, i])
            grad = np.sum(dn, axis=0)
            grad[l0 <= 0] = 0
            self.dt[0] += self.alpha * grad[np.newaxis, :].T * np.asarray(s)

            self.batch_cnt += 1

            if self.batch_cnt >= self.batch:
                self.batch_cnt = 0
                self.w    = np.add(self.w, self.dw)
                self.t[0] = np.add(self.t[0], self.dt[0])
                self.t[1] = np.add(self.t[1], self.dt[1])

                # print np.min(self.dt[0]), np.max(self.dt[0])
                # print np.min(self.dt[1]), np.max(self.dt[1])
                # print np.min(self.t[0]), np.max(self.t[0])
                # print np.min(self.t[1]), np.max(self.t[1])
                # print "--------------------------------------------------------"

                # while 1:
                #     pass

                self.dt = [np.zeros(self.dt[0].shape),
                           np.zeros(self.dt[1].shape)]
                self.dw =  np.zeros(self.dw.shape)

        self.a = a2
        self.s = s2
        self.l = (l02, l12, pi2)

        return self.a

# STATE:
# ------
# [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

# ACTIONS:
# --------
# +1, 0 or -1

EPSIODES = 10000
BATCH = 32
ALPHA = 1e-5
BETA  = 1e-5
GAMMA = 0.99
EXPLORE = 5000.0

scale = lambda x: (x[0], x[1]*100)

if __name__ == "__main__":

    env = gym.make('MountainCar-v0')

    agent = A2C(3, 2, BATCH, ALPHA, BETA, GAMMA, EXPLORE)

    score = 0.0

    for e in range(EPSIODES):

        t = 0
        R = 0
        r = 0
        s = env.reset()

        pos  = s[0]
        vel  = s[1]
        prev = s[0]

        while True:

            env.render()

            # a = agent(scale(s), abs(s[1]))
            # a = agent(s, abs(s[1]))
            a = agent(s, r)

            s, r, done, info = env.step(a)
            R += r

            if done:

                a = agent(np.zeros(s.shape), r)

                score += R
                print "Episode {} finished after {} timesteps, score {}"\
                                                .format(e+1, t+1, score/(e+1))
                break

            t += 1

        agent.reset()

# a = agent(scale(s), abs(s[1]))
# a = agent(s, r)
# a = agent(scale(s), r)
# a = agent(s, 10*r)
# a = agent(scale(s), abs(s[1])*100)
# a = agent(s, abs(s[1]))

# if vel * s[1] < 0 and \
#    abs(abs(pos) - abs(s[0])) > abs(abs(pos) - 0.5):# and \
#    #abs(s[0]+0.5) > abs(pos+0.5):
#     # a = agent((s[0], s[1]), abs(pos+0.5)*100)
#     # a = agent(s, abs(pos+0.5)*100)#100)
#     a = agent(scale(s), abs(pos+0.5))
#     pos = s[0]
#     vel = s[1]
# else:
#     a = agent(scale(s), 0)

# if abs(s[0]) > abs(pos):
#    #abs(s[0]+0.5) > abs(pos+0.5):
#     # a = agent((s[0], s[1]), abs(pos+0.5)*100)
#     # a = agent(s, abs(pos+0.5)*100)#100)
#     a = agent(scale(s), abs(pos+0.5))
#     pos = s[0]
#     vel = s[1]
# else:
#     a = agent(scale(s), 0)


# prev = s[0]
#
# if vel == 0:
#     vel = s[1]


# if t+1 >= 200:
#     # agent((s[0], s[1]*100), abs(pos+0.5)*100)
#     agent(scale(s), 0)
#     # agent(scale(s), abs(s[1])*100)
#     # agent(s, abs(s[1]))
# else:
#     agent(scale(s), 1)

# # agent(scale(s), abs(s[1]))

# agent(s, r)
# agent(scale(s), r)
