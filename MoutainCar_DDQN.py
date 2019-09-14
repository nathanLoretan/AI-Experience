import gym
import logging
import numpy as np
import tensorflow as tf
import tensorflow.losses as tfl
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from copy import deepcopy

class Model(tf.keras.Model):
    def __init__(self, nbr_actions, nbr_inputs):
        super().__init__('mlp_policy')

        self.hidden1_q  = kl.Dense(128, activation='relu', input_dim=(nbr_inputs))
        self.q          = kl.Dense(nbr_actions)

    def call(self, inputs):

        x = tf.convert_to_tensor(inputs)

        hidden1_q = self.hidden1_q(x)
        return self.q(hidden1_q)

    def q_value(self, state):

        q = self.predict(state)
        return np.squeeze(q)

class DDQN:

    def __init__(self, model):

        self.experiences        = []
        self.experiences_size   = 5000.0

        self.alpha          = 0.01
        self.gamma          = 0.99
        self.epsilon        = 1.0
        self.epsilon_min    = 0.01
        self.epsilon_decay  = 0.995
        self.batch_size     = 32
        self.target_update  = 0.01

        self.model = model
        self.model.compile(optimizer=ko.Adam(self.alpha), loss="MSE")

        self.target_model = model
        self.target_model.compile(optimizer=ko.Adam(self.alpha), loss="MSE")

        # Copy the weights of the online model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):

        q = self.model.q_value(state)

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

        self.experiences.append(experience)

        if len(self.experiences) > self.experiences_size:
            self.experiences.pop(0)

    def train(self):

        if len(self.experiences) < self.batch_size:
            return

        batch  = []
        states = []

        for i in range(self.batch_size):

            # Get experiances randomly
            exp = int(np.random.uniform(0, len(self.experiences)))
            s, s2, r, a, done = self.experiences[exp]

            states.append(np.squeeze(s, axis=0))

            q = self.model.q_value(s)

            if done == True:
                q[a] = r
            else:
                a2 = np.argmax(self.model.q_value(s2))
                q2 = self.target_model.q_value(s2)

                q[a] = r + self.gamma * q2[a2]

            batch.append(q)

        self.model.train_on_batch(np.array(states), np.array(batch))

        # Update the target network
        weights_target = self.target_model.get_weights()
        weights_model  = self.model.get_weights()

        for i in range(len(weights_target)):

            weights_target[i] *= (1 - self.target_update)
            weights_model[i]  *= self.target_update

            weights_target[i] += weights_model[i]

        self.target_model.set_weights(weights_target)

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)

    env = gym.make('MountainCar-v0')

    # Create an agent
    model = Model(env.action_space.n, env.observation_space.shape)
    agent = DDQN(model)

    episode = 0
    results = np.full(100, 200).tolist()

    while 1:

        s  = np.zeros((1, env.observation_space.shape[0]))
        s2 = np.zeros((1, env.observation_space.shape[0]))

        steps = 0

        s[0] = env.reset()

        while 1:

            env.render()

            a = agent.get_action(s)
            s2[0], r, done, _ = env.step(a)

            steps += 1

            if s2[0][0] >= 0.5:
                r += 100
            elif s2[0][0] >= 0.25:
                r += 20
            elif s2[0][0] >= 0.1:
                r += 10
            elif s2[0][0] >= -0.1:
                r += 2
            elif s2[0][0] >= -0.25:
                r += 1

            agent.store([s.copy(), s2.copy(), r, a, done])
            agent.train()

            s = s2.copy()

            if done:

                results.append(steps)
                if len(results) > 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                if score < 170:
                    print("Finished!!!")
                    exit()

                episode += 1

                print("Episode", episode,
                      "finished after", steps,
                      "timesteps, score", score)

                break
