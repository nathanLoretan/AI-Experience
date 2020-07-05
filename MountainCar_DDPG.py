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

SAVE_FILE_PATH = "MountainCar-DDPG.torch"

# if gpu is used
device = ("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):

    HIDDEN_LAYER_SIZE_1 = 512
    HIDDEN_LAYER_SIZE_2 = 256

    def __init__(self, inputs, outputs):
        super(Policy, self).__init__()

        self.h1 = nn.Linear(inputs, self.HIDDEN_LAYER_SIZE_1)
        self.norm1 = nn.LayerNorm(self.HIDDEN_LAYER_SIZE_1)
        self.relu1 = nn.ReLU()

        self.h2 = nn.Linear(self.HIDDEN_LAYER_SIZE_1, self.HIDDEN_LAYER_SIZE_2)
        self.norm2 = nn.LayerNorm(self.HIDDEN_LAYER_SIZE_2)
        self.relu2 = nn.ReLU()

        self.pi = nn.Linear(self.HIDDEN_LAYER_SIZE_2, outputs)
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.h1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.h2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        return self.tanh(self.pi(x))

class Q(nn.Module):

    HIDDEN_LAYER_SIZE_1 = 512
    HIDDEN_LAYER_SIZE_2 = 256

    def __init__(self, inputs, actions):
        super(Q, self).__init__()

        self.h1 = nn.Linear(inputs, self.HIDDEN_LAYER_SIZE_1)
        self.norm1 = nn.LayerNorm(self.HIDDEN_LAYER_SIZE_1)
        self.relu1 = nn.ReLU()

        self.h2 = nn.Linear(self.HIDDEN_LAYER_SIZE_1 + actions, self.HIDDEN_LAYER_SIZE_2)
        self.norm2 = nn.LayerNorm(self.HIDDEN_LAYER_SIZE_2)
        self.relu2 = nn.ReLU()

        self.q = nn.Linear(self.HIDDEN_LAYER_SIZE_2, 1)

    def forward(self, x, a):

        x = self.h1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = torch.cat((x, a), 1)

        x = self.h2(x)
        x = self.norm2(x)
        x = self.relu1(x)

        return self.q(x)

class DDPG:

    ALPHA = 1e-4
    GAMMA = 0.99
    POLYAK = 1e-3
    NOISE_DECAY = 1e-5
    NOISE_MIN = 0.2
    BATCH_SIZE = 64
    MEMORY_SIZE = 10000.0
    UPDATE = 1

    experience = namedtuple('Experience', ('s', 's2', 'r', 'a', 'done'))

    memory = []
    update = 0
    noise = 1.0

    def __init__(self, inputs, outputs):

        self.q = Q(inputs, outputs).to(device)
        self.pi = Policy(inputs, outputs).to(device)

        self.q_target = Q(inputs, outputs).to(device)
        self.pi_target = Policy(inputs, outputs).to(device)

        self.q_target.load_state_dict(self.q.state_dict())
        self.pi_target.load_state_dict(self.pi.state_dict())

        self.optimizer_q = optim.Adam(self.q.parameters(), lr=self.ALPHA)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.ALPHA)

    def action(self, s, use_noise=False):

        with torch.no_grad():
            pi = self.pi(torch.as_tensor(s).float().to(device)).cpu()

        if use_noise:

            pi = Normal(pi, self.noise).sample()

            if self.noise > self.NOISE_MIN:
                self.noise -= self.NOISE_DECAY

        a = pi.cpu()

        return a.clamp(-1.0, 1.0)

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

        states = torch.as_tensor(np.float32(batch.s), device=device)
        next_states = torch.as_tensor(np.float32(batch.s2), device=device)
        actions = torch.zeros([self.BATCH_SIZE, 1], device=device)
        rewards = torch.as_tensor(batch.r, device=device)
        done = torch.as_tensor(batch.done, device=device)

        y = torch.zeros([self.BATCH_SIZE, 1], device=device)

        q = self.q(states, actions)

        with torch.no_grad():

            pi2 = self.pi_target(torch.as_tensor(next_states).float())

            q2 = self.q_target(next_states, pi2)

            for i in range(self.BATCH_SIZE):

                if done[i]:
                    y[i] = rewards[i]
                else:
                    y[i] = rewards[i] + self.GAMMA * q2[i]

        self.optimizer_q.zero_grad()
        q_loss = torch.nn.MSELoss()(q, y)
        q_loss.backward()
        self.optimizer_q.step()

        pi = self.pi(torch.as_tensor(states).float())

        self.optimizer_pi.zero_grad()
        pi_loss = - self.q(states, pi).mean()
        pi_loss.backward()
        self.optimizer_pi.step()

        # Update target network
        for target_param, source_param in zip(self.q_target.parameters(), self.q.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.POLYAK) + \
                source_param.data * self.POLYAK)

        for target_param, source_param in zip(self.pi_target.parameters(), self.pi.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.POLYAK) + \
                source_param.data * self.POLYAK)

def play_agent(env, agent):

    episode = 0
    steps = 0

    while 1:

        steps = 0
        s = env.reset()

        while 1:

            env.render()

            a = agent.action(s)
            s, r, done, _ = env.step(a)

            steps += 1

            if done:

                episode += 1

                print("Episode", episode,
                      "finished after", steps)

                break

def train_agent(env, agent):

    episode = 0
    results = np.full(100, 200).tolist()

    while 1:

        rewards = 0
        steps = 0
        s = env.reset()

        while 1:

            a = agent.action(s, True)
            s2, r, done, _ = env.step(a)

            # if s2[0] >= 0.5:
            #     r += 100
            # elif s2[0] >= 0.25:
            #     r += 20
            # elif s2[0] >= 0.1:
            #     r += 10
            # elif s2[0] >= -0.1:
            #     r += 2
            # elif s2[0] >= -0.25:
            #     r += 1

            rewards += r
            steps += 1

            agent.store(s, s2, r , a, done)
            agent.train()

            s = s2

            if done:
                # Calcul the score total over 100 episodes
                results.append(steps)
                if len(results) > 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                if score < 170:
                    torch.save((
                        agent.q.state_dict(), \
                        agent.q_target.state_dict(), \
                        agent.pi.state_dict(), \
                        agent.pi_target.state_dict()), SAVE_FILE_PATH)
                    print("Finished!!!")
                    exit()

                episode += 1

                print("Episode", episode,
                      "finished after", rewards, steps,
                      "score", score)

                # Save the state of the agent
                if episode % 20 == 0:
                    torch.save((
                        agent.q.state_dict(), \
                        agent.q_target.state_dict(), \
                        agent.pi.state_dict(), \
                        agent.pi_target.state_dict()), SAVE_FILE_PATH)

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
    env = gym.make('MountainCarContinuous-v0')

    # Create an agent
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0])

    try:
        # q, q_target, pi, pi_target = torch.load(SAVE_FILE_PATH)
        # agent.q.load_state_dict(q)
        # agent.q_target.load_state_dict(q_target)
        # agent.pi.load_state_dict(pi)
        # agent.pi_ta<rget.load_state_dict(pi_target)
        print("Agent loaded!!!")
    except:
        print("Agent created!!!")

    if play:
        agent.q.eval()
        agent.pi.eval()
        play_agent(env, agent)
    elif train:
        train_agent(env, agent)

if __name__ == '__main__':
    run()
