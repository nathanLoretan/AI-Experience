import gym
import numpy as np
import tensorflow as tf
import tensorflow.losses as tfl
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

class Model(tf.keras.Model):
    def __init__(self, nbr_actions, nbr_inputs):
        super().__init__('mlp_policy')

        self.hidden1_v  = kl.Dense(128, activation='relu', input_dim=(nbr_inputs))
        self.hidden1_pi = kl.Dense(128, activation='relu', input_dim=(nbr_inputs))
        self.v          = kl.Dense(1)
        self.pi         = kl.Dense(nbr_actions, activation="softmax")

    def call(self, inputs):

        x = tf.convert_to_tensor(inputs)

        hidden1_v = self.hidden1_v(x)
        v = self.v(hidden1_v)

        hidden1_pi = self.hidden1_pi(x)
        pi = self.pi(hidden1_pi)

        # pi == actor, v == critic
        return pi, v

    def pi_v_value(self, state):

        # Return the policy and the value
        pi, v = self.predict(state)
        return np.squeeze(pi), v

class A2C:

    def __init__(self, model):

        self.experiences        = []
        self.experiences_size   = 1000.0

        self.alpha          = 0.001
        self.gamma          = 0.99
        self.entropy_coef   = 0.0001
        self.epsilon        = 1.0
        self.epsilon_min    = 0.01
        self.epsilon_decay  = 0.995
        self.batch_size     = 32

        self.model = model
        self.model.compile(optimizer=ko.Adam(self.alpha),
                           loss=[self.pi_loss, self.v_loss])

    def get_action(self, state):

        pi, v = self.model.pi_v_value(state)

        # e-Greedy explore
        greedy = np.full(len(pi), self.epsilon / (len(pi)-1))
        greedy[np.argmax(pi)] = 1.0 - self.epsilon

        # Get a random action
        a = np.random.choice(len(pi), 1, p=greedy.flatten())[0]

        # Decrease the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return a

    def store(self, experience):

        self.experiences.append(experience)

        # If no more memory for experiences, removes the first one
        if len(self.experiences) > self.experiences_size:
            self.experiences.pop(0)

    def train(self):

        if len(self.experiences) < self.batch_size:
            return

        actions = []
        states  = []

        policy     = []
        values     = []
        q_values   = []
        entropies  = []
        advantages = []

        for i in range(self.batch_size):

            # Get experiances randomly
            exp = int(np.random.uniform(0, len(self.experiences)))
            s, s2, r, a, done = self.experiences[exp]

            states.append(np.squeeze(s, axis=0))
            actions.append(np.array(a))

            _, v = self.model.pi_v_value(s)

            # Q = r + gamma * V
            if done == True:
                q = r
            else:
                _, v2 = self.model.pi_v_value(s2)
                q = r + self.gamma * v2[0, 0]

            # Advantage = Q - V
            adv = q - v[0, 0]

            values.append(q)
            advantages.append(adv)

        pi_loss_info = np.concatenate(
            [np.array(actions)[:, None], np.array(advantages)[:, None]], axis=-1)

        self.model.train_on_batch(
            np.array(states), [np.array(pi_loss_info), np.array(values)])

    def v_loss(self, true, value):

        # 1/2 * (R + gamma * V' - V)^2 where:
        # V == predicition
        # R + gamma * V' == true
        return 0.5 * tfl.mean_squared_error(true, value)

    def pi_loss(self, true, pi):

        a, adv = tf.split(true, 2, axis=-1)

        # Calculate the cross entropy
        pi_loss = tfl.sparse_softmax_cross_entropy(tf.cast(a, tf.int32), pi, adv)

        # Calculate the entropy of the policy, no need to multiply by the
        # probability as softmax is used
        entropy = tf.reduce_sum(tf.log(pi))

        return pi_loss - self.entropy_coef * entropy

if __name__ == '__main__':

    # Start OpenAI environment
    env = gym.make('CartPole-v0')

    # Create an agent
    model = Model(env.action_space.n, env.observation_space.shape)
    agent = A2C(model)

    episodes = 0
    results  = []

    # Structure to store the state
    s  = np.zeros((1, env.observation_space.shape[0]))
    s2 = np.zeros((1, env.observation_space.shape[0]))

    while 1:

        steps = 0

        s[0] = env.reset()

        while 1:

            # env.render()

            a = agent.get_action(s)
            s2[0], r, done, _ = env.step(a)

            agent.store((s.copy(), s2.copy(), r, a, done))
            agent.train()

            s  = s2.copy()

            steps += 1

            if done:

                # Calcul the score total over 100 episodes
                results.append(steps)
                if len(results) > 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                if score >= 195:
                    print("Finished!!!")
                    exit()

                episodes += 1

                print("Episode", episodes,
                      "finished after", steps,
                      "timesteps, score", score)

                break
