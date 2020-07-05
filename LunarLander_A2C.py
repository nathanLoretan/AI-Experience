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
from torch.distributions import Categorical

SAVE_FILE_PATH = "LunarLander-A2C.torch"

class Actor(nn.Module):

    HIDDEN_LAYER_SIZE = 128

    def __init__(self, inputs, outputs):
        super(Actor, self).__init__()

        self.h1 =  nn.Linear(inputs, self.HIDDEN_LAYER_SIZE)
        self.h2 =  nn.Linear(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE)
        self.pi = nn.Linear(self.HIDDEN_LAYER_SIZE, outputs)

    def forward(self, x):

        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return F.softmax(self.pi(x), dim=0)

class Critic(nn.Module):

    HIDDEN_LAYER_SIZE = 128

    def __init__(self, inputs):
        super(Critic, self).__init__()

        self.h1 =  nn.Linear(inputs, self.HIDDEN_LAYER_SIZE)
        self.h2 =  nn.Linear(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE)
        self.v = nn.Linear(self.HIDDEN_LAYER_SIZE, 1)

    def forward(self, x):

        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return self.v(x)

class A2C:

    ALPHA = 0.0001
    GAMMA = 0.99
    BATCH_SIZE = 64
    MEMORY_SIZE = 10000.0
    UPDATE = 1

    experience = namedtuple('Experience', ('s', 's2', 'r', 'a', 'done'))

    memory = []
    update = 0

    def __init__(self, inputs, outputs):

        # Create the model that will run on GPU
        self.actor = Actor(inputs, outputs)
        self.critic = Critic(inputs)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.ALPHA)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.ALPHA)

    def action(self, s):

        with torch.no_grad():
            pi = self.actor(torch.as_tensor(np.float32(s))).cpu()

        # Get a random action
        return int(np.random.choice(len(pi), 1, p=pi.numpy())[0])

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

        states = torch.as_tensor(np.float32(batch.s))
        next_states = torch.as_tensor(np.float32(batch.s2))
        actions = torch.as_tensor(batch.a)
        rewards = torch.as_tensor(batch.r)
        done = torch.as_tensor(batch.done)

        pi = self.actor(states)
        v = self.critic(states)
        v2 = self.critic(next_states).detach()

        adv = torch.zeros(self.BATCH_SIZE)
        q = torch.zeros([self.BATCH_SIZE, 1])
        log_probs = torch.zeros(self.BATCH_SIZE)

        entropy = 0

        for i in range(self.BATCH_SIZE):

            dist = Categorical(pi[i])
            log_probs[i] = dist.log_prob(actions[i])

            if done[i]:
                q[i][0] = rewards[i]
            else:
                q[i][0] = rewards[i] + self.GAMMA * v2[i]

            adv[i] = q[i][0] - v[i]

        loss_actor = - (log_probs * adv.detach()).mean()
        loss_critic = torch.nn.MSELoss()(v, q)

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

def play_agent(env, agent):

    results = []
    episodes = 0

    while 1:

        rewards = 0
        s = env.reset()

        while 1:

            env.render()

            a = agent.action(s)
            s, r, done, _ = env.step(a)

            rewards += r

            if done:

                episodes += 1

                print("Episode", episodes,
                      "finished after", rewards)

                break

def train_agent(env, agent):

    episode = 0
    results = []

    while 1:

        rewards = 0
        s = env.reset()

        while 1:

            a = agent.action(s)
            s2, r, done, _ = env.step(a)

            rewards += r

            agent.store(s, s2, r, a, done)
            agent.train()

            s = s2

            if done:

                # Calcul the score total over 100 episodes
                results.append(rewards)
                if len(results) > 150:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                if score >= 200:
                    print("Finished!!!")
                    exit()

                episode += 1

                print("Episode", episode,
                      "rewards", rewards,
                      "score", score)

                # Save the state of the agent
                if episode % 20 == 0:
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

    # Start OpenAI environment
    env = gym.make('LunarLander-v2')

    # Create an agent
    agent = A2C(env.observation_space.shape[0], env.action_space.n)

    try:
        critic, actor = torch.load(SAVE_FILE_PATH)
        agent.critic.load_state_dict(critic)
        agent.actor.load_state_dict(actor)
        print("Agent loaded!!!")
    except:
        print("Agent created!!!")

    if play:
        agent.critic.eval()
        agent.actor.eval()
        play_agent(env, agent)
    elif train:
        train_agent(env, agent)

if __name__ == '__main__':
    run()
