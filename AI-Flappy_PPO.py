import gym
import pygame
import logging
import threading
import click
from collections import namedtuple
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from copy import deepcopy, copy
from ple import PLE
from time import sleep
from pygame.locals import *
from math import sqrt, exp
from random import randint
from ple.games.flappybird import FlappyBird

# References:
# https://arxiv.org/abs/1707.06347

SAVE_FILE_PATH = "AI-Flappy-PPO.torch"

class Actor(nn.Module):

    HIDDEN_LAYER_SIZE = 128

    def __init__(self, inputs, outputs):
        super(Actor, self).__init__()

        self.h1 = nn.Linear(inputs, self.HIDDEN_LAYER_SIZE)
        self.pi = nn.Linear(self.HIDDEN_LAYER_SIZE, outputs)

    def forward(self, x):

        x = F.relu(self.h1(x))
        return F.softmax(self.pi(x), dim=0)

class Critic(nn.Module):

    HIDDEN_LAYER_SIZE = 128

    def __init__(self, inputs):
        super(Critic, self).__init__()

        self.h1 = nn.Linear(inputs, self.HIDDEN_LAYER_SIZE)
        self.v = nn.Linear(self.HIDDEN_LAYER_SIZE, 1)

    def forward(self, x):

        x = F.relu(self.h1(x))
        return self.v(x)

class PPO:

    ALPHA = 5e-5
    GAMMA = 0.9
    EPSILON = 2e-1
    ENTROPY = 1e-6
    VALUE = 1e-6
    EPOCH = 10
    MEMORY_SIZE = 1000
    BATCH_SIZE = 32
    UPDATE = 32

    experience = namedtuple('Experience', ('s', 's2', 'r', 'a', 'done', 'old_log_prob'))

    memory = []
    update = 0

    def __init__(self, inputs, outputs):

        self.actor = Actor(inputs, outputs)
        self.critic = Critic(inputs)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.ALPHA)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.ALPHA)

    def action(self, s):

        with torch.no_grad():
            pi = self.actor(torch.as_tensor(np.float32(s)))

        pi_dist = Categorical(pi)
        a = pi_dist.sample()

        return a, pi_dist.log_prob(a)

    def store(self, *args):

        self.memory.append(self.experience(*args))

        # If no more memory for memory, removes the first one
        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.pop(0)

    def train(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        self.update += 1

        # Train the network each UPDATE episode
        if self.update % self.UPDATE != 0:
            return

        samples = random.sample(self.memory, self.BATCH_SIZE)
        batch = self.experience(*zip(*samples))

        states = torch.as_tensor(batch.s).float()
        next_states = torch.as_tensor(batch.s2).float()
        actions = torch.as_tensor(batch.a)
        rewards = torch.as_tensor(batch.r)
        done = torch.as_tensor(batch.done)
        q = torch.zeros([self.BATCH_SIZE, 1])
        old_log_prob = torch.as_tensor(batch.old_log_prob)

        with torch.no_grad():

            v2 = self.critic(next_states).detach()
            v = self.critic(states).detach()

            for i in range(self.BATCH_SIZE):
                if done[i]:
                    q[i] = rewards[i]
                else:
                    q[i] = rewards[i] + self.GAMMA * v2[i]

            adv = (q - v).detach()

        for i in range(self.EPOCH):

            surr = torch.zeros(self.BATCH_SIZE)
            entropy = torch.zeros(self.BATCH_SIZE)

            v = self.critic(states)
            pi = self.actor(states)

            for y in range(self.BATCH_SIZE):

                pi_dist = Categorical(pi[y])
                entropy[y] = pi_dist.entropy()

                ratio = torch.exp(pi_dist.log_prob(actions[y]) - (old_log_prob[y] + 1e-10))
                clip = ratio.clamp(1.0 - self.EPSILON, 1.0 + self.EPSILON)
                surr[y] = torch.min(ratio * adv[y], clip * adv[y])

            loss_critic = torch.nn.MSELoss()(v, q)
            loss_critic_detached = loss_critic.clone().detach()

            loss_actor = \
                - surr.mean() + \
                self. VALUE * loss_critic_detached - \
                self.ENTROPY * entropy.mean()

            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

def normalize_state(obs):

    s = np.zeros(8)

    # Normalize
    s[0] = obs["player_y"] / 390.0
    s[1] = obs["player_vel"] / 10.0
    s[2] = obs["next_pipe_dist_to_player"] / 309.0
    s[3] = obs["next_pipe_top_y"] / 192.0
    s[4] = obs["next_pipe_bottom_y"] / 292.0
    s[5] = obs["next_next_pipe_dist_to_player"] / 453.0
    s[6] = obs["next_next_pipe_top_y"] / 192.0
    s[7] = obs["next_next_pipe_bottom_y"] / 292.0

    return s

def play_agent(flappy, env, agent):

    results = []
    episodes = 0

    while 1:

        env.reset_game()

        # Number of pipes passed as rewards
        passed = 0

        # Get initial state and normalize
        s = normalize_state(flappy.getGameState())

        while 1:

            a, log_prob = agent.action(s)
            r = env.act(p.getActionSet()[a])

            s = normalize_state(flappy.getGameState())

            r += -0.1

            if r > 0:
                r = 5
                passed += 1

            # check if the game is over
            if env.game_over():
                done = True
            else:
                done = False

            steps += 1

            if done:

                episodes += 1

                print("Episode", episodes,
                      "finished after", passed)

                break

def train_agent(flappy, env, agent):

    episodes = 0
    results = []

    while 1:

        env.reset_game()

        # Number of pipes passed as rewards
        passed = 0

        # Get initial state and normalize
        s = normalize_state(flappy.getGameState())

        while 1:

            a, log_prob = agent.action(s)
            r = env.act(env.getActionSet()[a])

            # r += -1
            #
            # The bird passed a pipe
            if r > 0:
                r = 5
                passed += 1

            s2 = normalize_state(flappy.getGameState())

            # check if the game is over
            if env.game_over():
                done = True
            else:
                done = False

            agent.store(s, s2, r, a, done, log_prob)
            agent.train()

            s = s2

            if done:

                # Calcul the score total over 100 episodes
                results.append(passed)
                if len(results) > 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                if score >= 100:
                    print("Finished!!!")
                    torch.save((agent.critic.state_dict(), agent.actor.state_dict()), SAVE_FILE_PATH)
                    exit()

                episodes += 1

                print("Episode", episodes,
                      "Number of pipes passed", passed,
                      ", total score", score)

                # Save the state of the agent
                if episodes % 100 == 0:
                    torch.save((agent.critic.state_dict(), agent.actor.state_dict()), SAVE_FILE_PATH)

                break

def clean_agent():
    os.remove(SAVE_FILE_PATH)

@click.command()
@click.option('--play', flag_value='play', default=False)
@click.option('--train', flag_value='train', default=True)
@click.option('--clean', flag_value='clean', default=False)
def run(play, train, clean):

    if clean:
        clean_agent()
        exit()

    # Start environment
    flappy = FlappyBird()

    if play:
        env = PLE(flappy, fps=30, display_screen=True, force_fps=False)
    else:
        env = PLE(flappy, fps=30, display_screen=False, force_fps=False)
    env.init()

    # Create an agent
    agent = PPO(len(flappy.getGameState()), len(env.getActionSet()))

    try:
        critic, actor = torch.load(SAVE_FILE_PATH)

        agent.critic.load_state_dict(critic)
        agent.actor.load_state_dict(actor)
        agent.critic.eval()
        agent.actor.eval()
        print("Agent loaded!!!")
    except:
        print("Agent created!!!")

    if play:
        play_agent(flappy, env, agent)
    elif train:
        train_agent(flappy, env, agent)

if __name__ == '__main__':
    run()
