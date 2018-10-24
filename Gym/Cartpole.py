# -*- coding: utf-8 -*-
import sys
import gym
import time
import numpy as np
from copy import deepcopy
from math import sqrt, exp
from random import randint
from collections import defaultdict

# DEBUG
debug       = False

# RPLY_SIZE   = between 1,000 and 10,000,000
# BATCH       = start at 32 and increase
# ALPHA       = 0.1 and 0.000001
# GAMMA       = 0.99
# LAMBD       = start at 0.00001 and increase
# TAU_UPDATE  = between every 100 updates and every 40000 updates

EPSIODES    = 5000

# Hyperparameters
LAYERS      = 4
HIDDEN      = 64
RPLY_SIZE   = 1000
BATCH       = 32
ALPHA       = 1e-6
GAMMA       = 0.99
LAMBD       = 1e-3
TAU         = 1e-1
TAU_UPDATE  = 1000
EPS_START   = 0.5
EPS_STOP    = 0.05
EPS_STEP    = (EPS_START - EPS_STOP) / 1000.0
EPS         = [EPS_START, EPS_STOP, EPS_STEP]

class DDQN():
    """ Double DQN """

    def __init__(self,
                 out,
                 inp,
                 h=32,
                 layers=2,
                 rply_size=1000,
                 batch=32,
                 alpha=0.9,
                 gamma=0.9,
                 lambd=0.1,
                 tau=0.9,
                 tau_update=100,
                 epsilon=0.0):

        self.alpha      = alpha         # Learning Factor
        self.gamma      = gamma         # Discount factor
        self.lambd      = lambd         # L2 Regularization parameter
        self.tau        = tau           # Soft update of target network
        self.tau_update = tau_update    # When updating target network
        self.eps        = epsilon[0]    # Exploration
        self.eps_stop   = epsilon[1]
        self.eps_step   = epsilon[2]

        self.s = None
        self.a = None
        self.q = None

        self.w        = []
        for i in range(layers):

            if i == 0:
                self.w.append(np.random.randn(h, inp) * np.sqrt(2.0 / inp))
            elif i == layers-1:
                self.w.append(np.random.randn(out, h) * np.sqrt(2.0 / h))
            else:
                self.w.append(np.random.randn(h, h) * np.sqrt(2.0 / h))

        # target network
        self.w_targ = deepcopy(self.w)

        self.dw = []
        for i in range(len(self.w)):
            self.dw.append(np.zeros(self.w[i].shape))

        self.batch      = batch
        self.batch_cnt  = 0
        self.update_cnt = 0

        self.rply_size  = rply_size
        self.rply       = [] # Experiance replay
        self.rply_cnt   = [] # Number of times experiance replayed
        self.rply_prio  = [] # Priority of the experiance based TD-error

        # Prioritized Replay
        self.beta      = 0
        self.beta_step = 0.000001

        # Momentum update
        # self.m = 0.5
        # self.m_step = 0.000001
        # self.v = deepcopy(self.dw)

        # Full Adam
        self.m     = deepcopy(self.dw)
        self.mt    = deepcopy(self.dw)
        self.v     = deepcopy(self.dw)
        self.vt    = deepcopy(self.dw)
        self.beta1 = 0.9
        self.beta2 = 0.999

    def reset(self):

        self.s = None
        self.a = None
        self.q = None

    def forward(self, s, w):

        l = []
        q = 0
        a = 0

        for i in range(len(w)):

            l.append(np.zeros(w[i].shape))

            if i == 0:
                l[i] = np.dot(w[i], s)
                l[i][l[i] < 0] = 0

            elif i == len(w)-1:
                q = np.dot(w[i], l[i-1])
                a = np.argmax(q)

            else:
                l[i] = np.dot(w[i], l[i-1])
                l[i][l[i] < 0] = 0

        l = tuple(l)

        return a, q, l

    def replay(self):

        if len(self.rply) < self.batch:

            # DEBUG
            # print "Not learning"
            return

        for i in range(self.batch):

            # Uniform Replay
            # exp = np.random.randint(low=0, high=len(self.rply))
            # r, s, a, s2 = self.rply[exp]

            # a2, _, _ = self.forward(s2, self.w)
            # _, q2, _ = self.forward(s2, self.w_targ)
            # _, q, l  = self.forward(s, self.w)

            # is_w = 1

            # Prioritized Replay
            p = np.asarray(self.rply_prio) / np.sum(np.asarray(self.rply_prio))
            exp = np.random.choice(len(p), 1, p=p)[0]
            r, s, a, s2 = self.rply[exp]

            a2, _, _ = self.forward(s2, self.w)
            _, q2, _ = self.forward(s2, self.w_targ)
            _, q, l  = self.forward(s, self.w)

            self.rply_cnt[exp] += 1
            self.rply_prio[exp] = abs((r + self.gamma * q2[a2] - q[a])) + 1e-6

            is_w = 1 / (self.rply_cnt[exp] * p[exp]) ** self.beta

            if self.beta < 1:
                self.beta += self.beta_step

                if self.beta > 1:
                    self.beta == 1

            # DEBUG
            # print (r + self.gamma * q2[a2] - q[a]), q[a], q2[a2]

            # DEBUG
            if np.isnan((r + self.gamma * q2[a2] - q[a]) * is_w):
                print is_w, self.rply_cnt[exp], p[exp]
                print "ERROR is_w is Nan"
                exit()

            grad = 0

            for i in range(len(self.w)-1, -1, -1):

                if i == len(self.w)-1:
                    grad = np.zeros(len(q))
                    grad[a] = (r + self.gamma * q2[a2] - q[a]) * is_w
                    dn = grad[np.newaxis, :].T * self.w[i]
                    self.dw[i] += grad[np.newaxis, :].T * l[i-1]

                elif i == 0:
                    grad = np.sum(dn, axis=0)
                    grad[l[i] <= 0] = 0
                    dn = grad[np.newaxis, :].T * self.w[i]
                    self.dw[i] += grad[np.newaxis, :].T * np.asarray(s)

                else:
                    grad = np.sum(dn, axis=0)
                    grad[l[i] <= 0] = 0
                    dn = grad[np.newaxis, :].T * self.w[i]
                    self.dw[i] += grad[np.newaxis, :].T * l[i-1]

        # DEBUG
        if np.isnan(np.max(self.dw[0])):
            print "ERROR dw is Nan"
            exit()

        # DEBUG
        global debug
        if debug:
            print np.min(self.alpha * self.dw[0] + self.lambd * self.w[0]), \
                  np.max(self.alpha * self.dw[0] + self.lambd * self.w[0]), \
                  np.min(self.w[0]), np.max(self.w[0])
            print "----------------"

        self.update_cnt += 1

        # Vanilla Update
        # self.w[0] -= self.alpha * self.dw[0] + self.lambd * self.w[0]
        # self.w[1] -= self.alpha * self.dw[1] + self.lambd * self.w[1]

        # Momentum Update
        # self.v[0] = self.v[0] * self.mu - self.alpha * self.dw[0] - self.lambd * self.w[0]
        # self.v[1] = self.v[1] * self.mu - self.alpha * self.dw[1] - self.lambd * self.w[1]
        # self.w[0] += self.v[0]
        # self.w[1] += self.v[1]

        # if self.mu < 1:
        #     self.mu += self.mu_step
        #
        #     if self.mu >= 1:
        #         self.mu == 0.99

        # Full Adam
        for i in range(len(self.w)):
            self.m[i]   = self.beta1 * self.m[i] + (1 - self.beta1) * self.dw[i]
            self.mt[i]  = self.m[i] / (1 - self.beta1**self.update_cnt)
            self.v[i]   = self.beta2 * self.v[i] + (1 - self.beta2) * (self.dw[i]**2)
            self.vt[i]  = self.v[i] / (1 - self.beta2**self.update_cnt)
            self.w[i]  += -self.alpha * self.mt[i] / (np.sqrt(self.vt[i]) + 1e-8)

        # DEBUG
        if np.isnan(np.max(self.w[0])):
            print "ERROR w is Nan"
            exit()

        self.dw = []
        for i in range(len(self.w)):
            self.dw.append(np.zeros(self.w[i].shape))

        if self.update_cnt % self.tau_update == 0:

            for i in range(len(self.w)):
                self.w_targ[i] = \
                        (1 - self.tau) * self.w_targ[i] + self.tau * self.w[i]

    def __call__(self, s2, r):

        # Calculate the policy and select the next action
        a2, q2, _ = self.forward(s2, self.w)

        # Exploration
        greedy = np.full(len(q2), self.eps / (len(q2)-1))
        greedy[a2] = 1.0 - self.eps
        a2 = np.random.choice(len(q2), 1, p=greedy)[0]

        # Decrease exploration factor
        if self.eps > self.eps_stop:
            self.eps -= self.eps_step

            if self.eps < 0:
                self.eps = self.eps_stop

        if self.s is not None:

            s = self.s
            a = self.a
            q = self.q

            exp = (r, s, a, s2)
            if exp not in enumerate(self.rply):
                self.rply.append(exp)
                self.rply_cnt.append(0)
                self.rply_prio.append(abs((r + self.gamma * q2[a2] - q[a])) + 1e-6)

                # DEBUG
                # print (r + self.gamma * q2[a2] - q[a]), q[a], q2[a2]

                if len(self.rply) > self.rply_size:
                    exp = np.random.randint(low=0, high=len(self.rply))
                    self.rply.pop(exp)
                    self.rply_cnt.pop(exp)
                    self.rply_prio.pop(exp)

        self.replay()

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

if __name__ == "__main__":

    agent = DDQN(2,
                 4,
                 HIDDEN,
                 LAYERS,
                 RPLY_SIZE,
                 BATCH,
                 ALPHA,
                 GAMMA,
                 LAMBD,
                 TAU,
                 TAU_UPDATE,
                 EPS)

    env = gym.make('CartPole-v0')

    results = []

    for e in range(EPSIODES):

        t = 0
        r = 0
        R = 0
        s = env.reset()

        while True:
            env.render()

            a = agent(s, r)
            s, r, done, info = env.step(a)
            R += r

            if done:

                a = agent(np.zeros(s.shape), r)

                results.append(R)
                if len(results) >= 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                print "Episode {} finished after {} timesteps, score {}"\
                        .format(e+1, t+1, score)

                print agent.eps

                break

            t += 1

        agent.reset()
