# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

import sys
import gym
import time
import numpy as np
from math import sqrt, exp
from random import randint
from collections import defaultdict

class QApprox():

    def __init__(self, out, inp, alpha=0.9, beta=0.9, gamma=0.9):

        # alpha     learning factor
        # gamma     reward factor
        # A         list of possible actions
        # s         state
        # a         action
        # r         reward
        # pi         policy

        self.gamma    = gamma
        self.alpha    = alpha

        self.s        = None
        self.a        = None
        self.q        = None

        self.w        = np.random.uniform(-1.0, 1.0, (out, inp))
        self.b        = np.zeros(out)

    def reset(self):

        self.s  = None
        self.a  = None
        self.q  = None

    def __call__(self, s2, r):

        # Calculate the policy and select the next action
        q2 = np.dot(self.w, s2) + self.b
        a2  = np.argmax(q2)

        if self.s is not None:

            a = self.a
            s = self.s
            q = self.q

            grad = np.zeros(len(q))
            grad[a] = (r + self.gamma * q2[a2] - q[a])
            self.w -= self.alpha * grad[np.newaxis, :].T * np.asarray(s)
            self.b -= self.alpha * grad

        q2 = np.dot(self.w, s2) + self.b
        a2  = np.argmax(q2)


        self.a = a2
        self.s = s2
        self.q = q2

        return self.a

# Observation
# -----------
# Num 	Observation 	Min 	Max
# 0 	Cart Position 	-2.4 	2.4
# 1 	Cart Velocity 	-Inf 	Inf
# 2 	Pole Angle 	~ -41.8° 	~ 41.8°
# 3 	Pole Velocity At Tip 	-Inf 	Inf
#
# Actions
# -------
# Num 	Action
# 0 	Push cart to the left
# 1 	Push cart to the right

state = lambda x: x/abs(x) if x != 0 else 0

EPSIODES = 10000
ALPHA = 0.1
GAMMA = 0.95

if __name__ == "__main__":

    agent = QApprox(2, 4, ALPHA, GAMMA)
    env = gym.make('CartPole-v0')

    score = 0.0

    for e in range(EPSIODES):

        t = 0
        r = 0
        R = 0
        s = env.reset()

        while True:

            env.render()
            # s = (state(obs[0]), state(obs[1]), state(obs[2]), state(obs[3]))

            if abs(s[1]) <= 0.1 and abs(s[3]) <= 0.1:
                a = agent(s, 0.1)
            else:
                a = agent(s, -0.1)

            s, r, done, info = env.step(a)

            R += r

            if done:

                # s = (state(obs[0]), state(obs[1]), state(obs[2]), state(obs[3]))

                if t+1 >= 200:
                    a = agent(s, 1)
                else:
                    a = agent(s, -1)

                score += R
                print "Episode {} finished after {} timesteps, score {}/{}"\
                        .format(e+1, t+1, score/((e%100)+1), (e%100)+1)

                if (e+1) % 100 == 0:
                    score = 0.0

                break

            t += 1

        agent.reset()
