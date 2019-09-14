import gym
import pygame
import logging
import threading
import numpy as np
import tensorflow as tf
import tensorflow.losses as tfl
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from ple import PLE
from time import sleep
from pygame.locals import *
from math import sqrt, exp
from random import randint
from ple.games.flappybird import FlappyBird

class Model(tf.keras.Model):
    def __init__(self, nbr_actions):
        super().__init__('mlp_policy')

        self.hidden1_q  = kl.Dense(128, activation='relu')
        self.q          = kl.Dense(nbr_actions)

    def call(self, inputs):

        x = tf.convert_to_tensor(inputs)

        hidden1_q = self.hidden1_q(x)
        return self.q(hidden1_q)

    def q_value(self, state):

        q = self.predict(state)
        return np.squeeze(q)

model_mutex = threading.Lock()
experience_mutex = threading.Lock()

class DDQN:

    def __init__(self, model):

        self.experiences        = []
        self.experiences_size   = 5000.0

        self.alpha          = 0.01
        self.gamma          = 0.99
        self.epsilon        = 1.0
        self.epsilon_min    = 0.01
        self.epsilon_decay  = 0.99995
        self.batch_size     = 32
        self.target_update  = 0.01

        self.model = model
        self.model.compile(optimizer=ko.Adam(self.alpha), loss="MSE")

        self.target_model = model
        self.target_model.compile(optimizer=ko.Adam(self.alpha), loss="MSE")

        self.train_model = model
        self.train_model.compile(optimizer=ko.Adam(self.alpha), loss="MSE")

        # Copy the weights of the online model to the target model
        self.target_model.set_weights(self.model.get_weights())
        self.train_model.set_weights(self.model.get_weights())

    def get_action(self, state):

        model_mutex.acquire()
        q = self.model.q_value(state)
        model_mutex.release()

        # e-Greedy explore
        greedy = np.full(len(q), self.epsilon / (len(q)-1))
        greedy[np.argmax(q)] = 1.0 - self.epsilon

        # Get a random action
        a = np.random.choice(len(q), 1, p=greedy.flatten())[0]

        # Decrease the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return a

    def store(self, experience):

        experience_mutex.acquire()
        self.experiences.append(experience)

        if len(self.experiences) > self.experiences_size:
            self.experiences.pop(0)

        experience_mutex.release()

    def train(self):

        while 1:

            experience_mutex.acquire()
            if len(self.experiences) < self.batch_size:
                experience_mutex.release()
                return
            experience_mutex.release()

            batch  = []
            states = []

            for i in range(self.batch_size):

                # Get experiances randomly
                experience_mutex.acquire()
                exp = int(np.random.uniform(0, len(self.experiences)))
                s, s2, r, a, done = self.experiences[exp]
                experience_mutex.release()

                states.append(np.squeeze(s, axis=0))

                q = self.train_model.q_value(s)

                if done == True:
                    q[a] = r
                else:
                    a2 = np.argmax(self.train_model.q_value(s2))
                    q2 = self.target_model.q_value(s2)

                    q[a] = r + self.gamma * q2[a2]

                batch.append(q)

            self.train_model.train_on_batch(np.array(states), np.array(batch))

            # Update the target network
            weights_target = self.target_model.get_weights()
            weights_model  = self.train_model.get_weights()

            for i in range(len(weights_target)):

                weights_target[i] *= (1 - self.target_update)
                weights_model[i]  *= self.target_update

                weights_target[i] += weights_model[i]

            self.target_model.set_weights(weights_target)

            model_mutex.acquire()
            self.model.set_weights(self.train_model.get_weights())
            model_mutex.release()

def get_state(obs):

    state = np.zeros((1, 8))

    # Normalize
    state[0, 0] = obs["player_y"] / 390
    state[0, 1] = obs["player_vel"] / 10
    state[0, 2] = obs["next_pipe_dist_to_player"] / 309
    state[0, 3] = obs["next_pipe_top_y"] / 192
    state[0, 4] = obs["next_pipe_bottom_y"] / 292
    state[0, 5] = obs["next_next_pipe_dist_to_player"] / 453
    state[0, 6] = obs["next_next_pipe_top_y"] / 192
    state[0, 7] = obs["next_next_pipe_bottom_y"] / 292

    # state = state / np.sum(state)

    return state

if __name__ == '__main__':

    # Start environment
    flappy = FlappyBird()
    p = PLE(flappy, fps=30, display_screen=True, force_fps=False)
    p.init()

    # Create an agent
    model = Model(len(p.getActionSet()))
    agent = DDQN(model)

    episodes = 0
    results  = []

    # Structure to store the state
    s2 = np.zeros((1, 8))
    s  = np.zeros((1, 8))

    replay_thread = threading.Thread(target=agent.train)
    replay_thread.daemon = True
    replay_thread.start()

    while 1:

        # Number of pipes passed
        passed = 0

        # Get initial state
        s = get_state(flappy.getGameState())

        while 1:

            # Get next state
            a = agent.get_action(s)
            r = p.act(p.getActionSet()[a])

            if r > 0:
                passed += 1

            s2 = get_state(flappy.getGameState())

            # check if the game is over
            if p.game_over():
                done = True
            else:
                done = False

            agent.store((s.copy(), s2.copy(), r, a, done))
            # agent.train()

            s = s2.copy()

            if done:

                p.reset_game()

                # Calcul the score total over 100 episodes
                results.append(passed)
                if len(results) > 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                if score >= 100:
                    print("Finished!!!")
                    exit()

                episodes += 1

                print("Episode", episodes,
                      "Number of pipes passed", passed,
                      ", total score", score)

                break
