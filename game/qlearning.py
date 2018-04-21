# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

from search import *
from collections import defaultdict

class QLearning():

    def __init__(self, actions, Ne, Rplus, alpha=(lambda n: 1./(1+n)), gamma=0.9):

        # Ne        Number of time needed before considering a state as known
        # Rplus     explorating reward estimated of unkown states
        # alpha     learning factor
        # gamma     reward factor
        # Rplus     Reward used to initiate exploration
        # Q         action values of state/action
        # Nsa       frequencies of state/action pairs
        # s         state
        # a         action
        # r         reward
        # u         utility
        # action    list of possible action

        self.gamma    = gamma
        self.alpha    = alpha
        self.Rplus    = Rplus
        self.Ne       = Ne

        self.Q        = defaultdict(float)
        self.Nsa      = defaultdict(float)
        self.s        = None
        self.a        = None

        self.actions   = actions

    def f(self, u, n):
        """ Exploration function. Returns fixed Rplus untill
            agent has visited state, action a Ne number of times.
            Same as ADP agent in book.

            [in] u      : utility of the state
            [in] n      : Number of time the state has been visited """

        if n < self.Ne:
            return self.Rplus
        else:
            return u

    def reset(self):
        """ Reset the different elements needed to allow the agent to run
        correctly the next trial. """

        self.s = None
        self.a = None
        self.r = None

    def __call__(self, state, reward):
        """ Learn the Q(s,a) of the current state and return the next action
            the agent must perform.

            [in] percept        : current state and reward earned
            [in] terminalFound  : position of the terminal state"""

        ns = state # next state

        Q   = self.Q   # Get reference for fancier notation
        Nsa = self.Nsa # Get reference for fancier notation

        r = reward
        s = self.s # Get reference for fancier notation
        a = self.a # Get reference for fancier notation

        alpha = self.alpha # Get reference for fancier notation
        gamma = self.gamma # Get reference for fancier notation

        actions = self.actions # Get reference for fancier notation

        # Check if the agent has already visited the previous state s
        # and update Q matrix
        if s is not None:
            Nsa[s, a] += 1
            Q[s, a]   += alpha(Nsa[s, a]) * \
                           (r + gamma * max(Q[ns, na] for na in actions) - \
                            Q[s, a])

        # Get the next action to perform
        na = argmax(actions, key=lambda na: self.f(Q[ns, na], Nsa[ns, na]))

        # Save the next state and next action
        self.s, self.a = ns, na

        # Return new action
        return na
