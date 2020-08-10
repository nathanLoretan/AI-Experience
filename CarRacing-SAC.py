import click
from collections import namedtuple
import gym
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

SAVE_FILE_PATH = "CarRacing-SAC.torch"

# if gpu is used
device = ("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):

    HIDDEN_LAYER_SIZE_1 = 512
    HIDDEN_LAYER_SIZE_2 = 256

    def __init__(self, inputs2D, outputs):
        super(Policy, self).__init__()

        self.inputs1D = inputs2D[0] * inputs2D[1]

        self.h1 = nn.Linear(self.inputs1D, self.HIDDEN_LAYER_SIZE_1)
        self.h2 = nn.Linear(self.HIDDEN_LAYER_SIZE_1, self.HIDDEN_LAYER_SIZE_2)
        self.pi = nn.Linear(self.HIDDEN_LAYER_SIZE_2, outputs)

    def forward(self, x):

        x = x.view(-1, self.inputs1D)

        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return torch.tanh(self.pi(x))

class Q(nn.Module):

    HIDDEN_LAYER_SIZE_1 = 512
    HIDDEN_LAYER_SIZE_2 = 256

    def __init__(self, inputs2D, actions):
        super(Q, self).__init__()

        self.inputs1D = inputs2D[0] * inputs2D[1]
        self.actions = actions

        self.h1 = nn.Linear(self.inputs1D + actions, self.HIDDEN_LAYER_SIZE_1)
        self.h2 = nn.Linear(self.HIDDEN_LAYER_SIZE_1, self.HIDDEN_LAYER_SIZE_2)
        self.q = nn.Linear(self.HIDDEN_LAYER_SIZE_2, 1)

    def forward(self, x, a):

        x = x.view(-1, self.inputs1D)

        for i in range(self.actions):
            x = torch.cat((x, a[i]), 1)

        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return self.q(x)

class SAC:

    LR = 1e-3
    GAMMA = 0.99
    POLYAK = 5e-3
    ALPHA = 2e-1
    NOISE_DECAY = 1e-6
    NOISE_MIN = 0.2
    NOISE_START = 1.0
    BATCH_SIZE = 64
    UPDATE_INTERVAL = 50
    MEMORY_SIZE = 10000.0

    experience = namedtuple('Experience', ('s', 's2', 'r', 'a', 'done'))

    memory = []
    update = 0
    noise = NOISE_START

    def __init__(self, inputs, outputs):

        self.outputs = outputs
        self.inputs = inputs

        self.q1 = Q(inputs, outputs).to(device)
        self.q2 = Q(inputs, outputs).to(device)

        self.q1_target = Q(inputs, outputs).to(device)
        self.q2_target = Q(inputs, outputs).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.optimizer_q1 = optim.Adam(self.q1.parameters(), lr=self.LR)
        self.optimizer_q2 = optim.Adam(self.q2.parameters(), lr=self.LR)

        self.pi = Policy(inputs, outputs).to(device)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.LR)

    def action(self, s, use_noise=False):

        with torch.no_grad():
            pi = self.pi(torch.as_tensor(s.copy()).float().to(device))
            pi = pi.cpu()[0]

        if use_noise:

            pi = Normal(pi, self.noise).sample()

            if self.noise > self.NOISE_MIN:
                self.noise -= self.NOISE_DECAY

        a = []

        for i in range(self.outputs):
            a.append(pi[i].clamp(-1.0, 1.0).item())

        return a

    def store(self, *args):

        self.memory.append(self.experience(*args))

        # If no more memory for memory, removes the first one
        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.pop(0)

    def train(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        if self.update < self.UPDATE_INTERVAL:
            self.update += 1
            return

        self.update = 0

        samples = random.sample(self.memory, self.BATCH_SIZE)
        batch = self.experience(*zip(*samples))

        states = torch.as_tensor(np.float32(batch.s), device=device)
        next_states = torch.as_tensor(np.float32(batch.s2), device=device)
        rewards = torch.as_tensor(batch.r, device=device)
        done = torch.as_tensor(batch.done, device=device)
        y = torch.zeros([self.BATCH_SIZE, 1], device=device)
        actions = torch.zeros([self.outputs, self.BATCH_SIZE, 1], device=device)

        for i in range(self.BATCH_SIZE):
            for j in range(self.outputs):
                actions[j][i] = batch.a[i][j]

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)

        with torch.no_grad():

            next_pi = self.pi(next_states)
            next_pi_dist = Normal(next_pi, 1e-8)
            next_pi_logprob = next_pi_dist.log_prob(next_pi).sum(-1, keepdim=True)
            next_pi_actions = torch.zeros([self.outputs, self.BATCH_SIZE, 1], device=device)

            for i in range(self.BATCH_SIZE):
                for j in range(self.outputs):
                    next_pi_actions[j][i] = next_pi[i][j].clamp(-1.0, 1.0)

            next_q1 = self.q1_target(next_states, next_pi_actions)
            next_q2 = self.q2_target(next_states, next_pi_actions)

            next_q = torch.min(next_q1, next_q2)

            for i in range(self.BATCH_SIZE):

                if done[i]:
                    y[i] = rewards[i]
                else:
                    y[i] = rewards[i] + self.GAMMA * (next_q[i] - self.ALPHA * next_pi_logprob[i])

        q1_loss = torch.nn.MSELoss()(q1, y)
        q2_loss = torch.nn.MSELoss()(q2, y)

        self.optimizer_q1.zero_grad()
        q1_loss.backward()
        self.optimizer_q1.step()

        self.optimizer_q2.zero_grad()
        q2_loss.backward()
        self.optimizer_q2.step()

        pi = self.pi(states)
        pi_dist = Normal(pi, 1e-8)
        pi_actions = torch.zeros([self.outputs, self.BATCH_SIZE, 1], device=device)

        for i in range(self.BATCH_SIZE):
            for j in range(self.outputs):
                pi_actions[j][i] = pi[i][j]

        q1 = self.q1(states, pi_actions)
        q2 = self.q2(states, pi_actions)
        q = torch.min(q1, q2)

        pi_loss = - (q - self.ALPHA * pi_dist.log_prob(pi).sum(axis=1)).mean()

        self.optimizer_pi.zero_grad()
        pi_loss.backward()
        self.optimizer_pi.step()

        # Update target network
        for target_param, source_param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.POLYAK) + \
                source_param.data * self.POLYAK)

        for target_param, source_param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.POLYAK) + \
                source_param.data * self.POLYAK)

def play_agent(env, agent):

    results = []
    episode = 0

    while 1:

        rewards = 0
        s = env.reset()

        # Convert image to greyscale
        s = np.dot(s[...,:3], [0.299, 0.587, 0.144])

        while 1:

            env.render()

            a = agent.action(np.expand_dims(s, 0), True)
            s, r, done, _ = env.step(a)

            # Convert image to greyscale
            s = np.dot(s[...,:3], [0.299, 0.587, 0.144])

            rewards += r

            if done:

                episode += 1

                print("Episode", episode,
                      "finished after", rewards)

                break

def train_agent(env, agent):

    episode = 0
    results = []

    while 1:

        rewards = 0
        s = env.reset()

        # Convert image to greyscale
        s = np.dot(s[...,:3], [0.299, 0.587, 0.144])

        while 1:

            a = agent.action(np.expand_dims(s, 0), True)
            s2, r, done, _ = env.step(a)

            # Convert image to greyscale
            s2 = np.dot(s2[...,:3], [0.299, 0.587, 0.144])

            rewards += r

            agent.store(s, s2, r, a, done)
            agent.train()

            s = s2

            if done:

                # Calcul the score total over 100 episodes
                results.append(rewards)
                if len(results) > 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                if score >= 200:
                    torch.save((
                        agent.q1.state_dict(), \
                        agent.q1_target.state_dict(), \
                        agent.q2.state_dict(), \
                        agent.q2_target.state_dict(), \
                        agent.pi.state_dict()), SAVE_FILE_PATH)
                    print("Finished!!!")
                    exit()

                episode += 1

                print("Episode", episode,
                      "rewards", rewards,
                      "score", score)

                # Save the state of the agent
                if episode % 20 == 0:
                    torch.save((
                        agent.q1.state_dict(), \
                        agent.q1_target.state_dict(), \
                        agent.q2.state_dict(), \
                        agent.q2_target.state_dict(), \
                        agent.pi.state_dict()), SAVE_FILE_PATH)

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

    # Start OpenAI environment
    env = gym.make('CarRacing-v0')

    # Create an agent
    agent = SAC([env.observation_space.shape[0],
        env.observation_space.shape[1]], env.action_space.shape[0])

    try:
        q1, q1_target, q2, q2_target, pi, pi_target = torch.load(SAVE_FILE_PATH)
        agent.q1.load_state_dict(q1)
        agent.q1_target.load_state_dict(q1_target)
        agent.q2.load_state_dict(q2)
        agent.q2_target.load_state_dict(q2_target)
        agent.pi.load_state_dict(pi)
        print("Agent loaded!!!")
    except:
        print("Agent created!!!")

    if play:
        agent.q1.eval()
        agent.q2.eval()
        agent.pi.eval()
        play_agent(env, agent)
    elif train:
        train_agent(env, agent)

if __name__ == '__main__':
    run()
