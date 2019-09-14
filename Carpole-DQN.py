import gym
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

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

class DQN:

    def __init__(self, model):

        self.experiences        = []
        self.experiences_size   = 1000.0

        self.alpha          = 0.001
        self.gamma          = 0.99
        self.epsilon        = 1.0
        self.epsilon_min    = 0.01
        self.epsilon_decay  = 0.995
        self.batch_size     = 32

        self.model = model
        self.model.compile(optimizer=ko.Adam(self.alpha), loss="MSE")

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
                q2 = self.model.q_value(s2)
                q[a] = r + self.gamma * np.amax(q2)

            batch.append(q)

        self.model.train_on_batch(np.array(states), np.array(batch))

if __name__ == '__main__':

    # Start OpenAI environment
    env = gym.make('CartPole-v0')

    # Create an agent
    model = Model(env.action_space.n, env.observation_space.shape)
    agent = DQN(model)

    episode = 0
    results = []

    while 1:

        s  = np.zeros((1, env.observation_space.shape[0]))
        s2 = np.zeros((1, env.observation_space.shape[0]))

        steps = 0

        s[0] = env.reset()

        while 1:

            # env.render()

            a = agent.get_action(s)
            s2[0], r, done, _ = env.step(a)

            steps += 1

            agent.store((s.copy(), s2.copy(), r, a, done))
            agent.train()

            s = s2.copy()

            if done:

                # Calcul the score total over 100 episodes
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

                break
