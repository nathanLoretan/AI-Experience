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

SAVE_FILE_PATH = "Carpole-DQN.torch"

# if gpu is used
device = ("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):

    HIDDEN_LAYER_SIZE = 128

    def __init__(self, inputs, outputs):
        super(Model, self).__init__()

        self.h1 = nn.Linear(inputs, self.HIDDEN_LAYER_SIZE)
        self.q = nn.Linear(self.HIDDEN_LAYER_SIZE, outputs)

    def forward(self, x):

        x = F.relu(self.h1(x))
        return self.q(x)

class DQN:

    ALPHA = 0.001
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    BATCH_SIZE = 64
    MEMORY_SIZE = 1000

    experience = namedtuple('Experience', ('s', 's2', 'r', 'a', 'done'))

    def __init__(self, inputs, outputs):

        self.memory = []
        self.epsilon = self.EPSILON

        # Create the model that will run on GPU
        self.model = Model(inputs, outputs).to(device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.ALPHA)

    def action(self, s):

        with torch.no_grad():
            q = self.model(torch.as_tensor(np.float32(s)).to(device)).cpu()

        # e-Greedy explore
        greedy = np.full(len(q), self.epsilon / (len(q)-1))
        greedy[np.argmax(q)] = 1.0 - self.epsilon

        # Get a random action
        a = int(np.random.choice(len(q), 1, p=greedy.flatten())[0])

        # Decrease the exploration rate
        if self.epsilon > self.EPSILON_MIN:
            self.epsilon *= self.EPSILON_DECAY
        else:
            self.epsilon = self.EPSILON_MIN

        return a

    def store(self, *args):

        self.memory.append(self.experience(*args))

        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.pop(0)

    def train(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        samples = random.sample(self.memory, self.BATCH_SIZE)
        batch = self.experience(*zip(*samples))

        states = torch.as_tensor(np.float32(batch.s), device=device)
        next_states = torch.as_tensor(np.float32(batch.s2), device=device)
        actions = torch.as_tensor(batch.a, device=device)
        rewards = torch.as_tensor(batch.r, device=device)
        done = torch.as_tensor(batch.done, device=device)

        q = self.model(states)
        q2 = self.model(next_states).detach()
        qtarget = q.clone()

        for i in range(self.BATCH_SIZE):

            if done[i]:
                qtarget[i, actions[i]] = rewards[i]
            else:
                qtarget[i, actions[i]] = rewards[i] + self.GAMMA * torch.max(q2[i])

        loss = torch.nn.MSELoss()(q, qtarget)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def play_agent(env, agent):

    results = []
    episodes = 0
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

                episodes += 1

                print("Episode", episodes,
                      "finished after", steps)

                break

def train_agent(env, agent):

    episode = 0
    results = []

    while 1:

        steps = 0
        s = env.reset()

        while 1:

            a = agent.action(s)
            s2, r, done, _ = env.step(a)

            agent.store(s, s2, r, a, done)
            agent.train()

            s = s2

            steps += 1

            if done:

                # Calcul the score total over 100 episodes.
                # The problem is considered sovled when a score
                # of 195 is reached.
                results.append(steps)
                if len(results) > 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                if score >= 195:
                    print("Finished!!!")
                    exit()

                episode += 1

                print("Episode", episode,
                      "finished after", steps,
                      "timesteps, score", score)

                # Save the state of the agent
                if episode % 20 == 0:
                    torch.save(agent.model.state_dict(), SAVE_FILE_PATH)

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
    env = gym.make('CartPole-v0')

    # Create an agent
    agent = DQN(env.observation_space.shape[0], env.action_space.n)

    try:
        agent.model.load_state_dict(torch.load(SAVE_FILE_PATH))
        agent.model.eval()
        print("Agent loaded!!!")
    except:
        print("Agent created!!!")

    if play:
        play_agent(env, agent)
    elif train:
        train_agent(env, agent)

if __name__ == '__main__':
    run()
