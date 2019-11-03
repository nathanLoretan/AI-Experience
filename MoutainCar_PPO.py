import gym
import logging
import numpy as np
import tensorflow as tf
import tensorflow.losses as tfl
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from copy import deepcopy

# References:
# https://medium.com/@jonathan_hui/rl-the-math-behind-trpo-ppo-d12f6c745f33
# https://arxiv.org/abs/1707.06347

class Model():

    def __init__(self, action_space, state_space):

        self.alpha             = 1e-3
        self.gamma             = 0.99
        self.eps               = 2e-1
        self.ent_coef          = 1e-2
        self.value_coef        = 1e-1

        self.state_space  = state_space
        self.action_space = action_space

        self.start = True

        with tf.variable_scope("policy"):

            self.states_pi = \
                tf.placeholder(tf.float32, shape=(None, self.state_space), name='states_pi')
            self.actions = \
                tf.placeholder(tf.int32, shape=(None, 2), name='actions')
            self.advantages = \
                tf.placeholder(tf.float32, shape=(None,), name='advantages')
            self.pi_old = \
                tf.placeholder(tf.float32, shape=(None, self.action_space), name='old_pi')

            # Create neural network
            self.h1_pi   = tf.layers.dense(self.states_pi, 32, tf.tanh, name="h1_pi")
            self.pi      = tf.layers.dense(self.h1_pi, self.action_space, tf.nn.softmax, name="pi")

            # Weights used by the model and dimension of the network
            self.theta_policy     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy')
            self.theta_policy_dim = [t.shape.as_list() for t in self.theta_policy]
            self.theta_policy_nbr = np.sum([np.prod(d) for d in self.theta_policy_dim])

            # NOTE: The weights of the network are updating by reshaping the
            # dimension as a vector.
            self.flat_theta_policy = tf.concat([tf.reshape(t, [-1]) for t in self.theta_policy], axis=0)
            self.flat_theta_policy_updater = tf.placeholder(tf.float32, self.theta_policy_nbr)

            # Change the weights of the network, when changing the weight
            # of the network, theta must be given as a 1-D vector
            self.policy_updater = []

            i = 0
            for layer, shape in enumerate(self.theta_policy_dim):
                s = np.prod(shape)
                theta = tf.reshape(self.flat_theta_policy_updater[i:i + s], shape)
                self.policy_updater.append(self.theta_policy[layer].assign(theta))
                i += s

        with tf.variable_scope("value"):

            self.states_v = tf.placeholder(tf.float32, (None, self.state_space), 'states_v')
            self.true_v   = tf.placeholder(tf.float32, (None,), 'true_v')

            # Create neural network
            self.h1_v = tf.layers.dense(self.states_v, 128, tf.nn.relu, name="h1_v")
            self.v    = tf.squeeze(tf.layers.dense(self.h1_v, 1, tf.tanh, name="v"))

            self.theta_value     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'value')
            self.theta_value_dim = [t.shape.as_list() for t in self.theta_value]
            self.theta_value_nbr = np.sum([np.prod(d) for d in self.theta_value_dim])

        # Square error loss
        self.loss_value = tf.reduce_mean(tf.square(self.true_v - self.v))

        # Adam optimization for grdient descent
        optimizer = tf.train.AdamOptimizer(self.alpha)
        self.adam_value = optimizer.minimize(self.loss_value, var_list=self.theta_value)

        self.entropy = tf.reduce_mean(tf.reduce_sum(self.pi * tf.log(self.pi), axis=1), axis=0)

        self.ratio = tf.gather_nd(tf.exp(tf.log(self.pi) - tf.log(self.pi_old)), self.actions)

        self.clip = tf.clip_by_value(self.ratio, 1 - self.eps, 1 + self.eps)

        self.loss_policy = \
            - tf.reduce_mean(tf.minimum(tf.multiply(self.advantages, self.ratio),
                                        tf.multiply(self.advantages, self.clip)))

        self.loss = self.loss_policy + \
                    self.ent_coef * self.entropy + \
                    self.value_coef * self.loss_value

        # Adam optimization for grdient descent
        optimizer = tf.train.AdamOptimizer(self.alpha)
        self.adam_policy = optimizer.minimize(self.loss, var_list=self.theta_policy)

    def get_flat_theta(self):
        return tf.get_default_session().run(self.flat_theta_policy)

    def assign_policy_theta(self, theta):

        feed_dict = {
            self.flat_theta_policy_updater: theta
        }

        tf.get_default_session().run(self.policy_updater, feed_dict=feed_dict)

    def update_policy(self, states, actions, advantages, true_v):

        if self.start:
            self.theta_policy_old = self.get_flat_theta()
            self.start = False
            return

        feed_dict = {
            self.states_pi: states,
            self.actions: actions,
            self.states_v: states,
            self.true_v: true_v,
            self.advantages: advantages
        }

        # Save the current weights
        theta_policy_curr = self.get_flat_theta()

        # Get the previous policy used during linear search
        self.assign_policy_theta(self.theta_policy_old)
        pi_old = tf.get_default_session().run([self.pi], feed_dict)
        feed_dict[self.pi_old] = np.squeeze(pi_old)

        # Restore the current weights
        self.assign_policy_theta(theta_policy_curr)

        tf.get_default_session().run(self.adam_policy, feed_dict=feed_dict)

        self.theta_policy_old = theta_policy_curr


    def update_value(self, states, true_v):

        feed_dict = {
            self.states_v: states,
            self.true_v: true_v
        }

        tf.get_default_session().run(self.adam_value, feed_dict=feed_dict)

    def policy(self, state):

        feed_dict = {
            self.states_pi: state
        }

        return tf.get_default_session().run(self.pi, feed_dict=feed_dict)

    def value(self, state):

        feed_dict = {
            self.states_v: state
        }

        return tf.get_default_session().run(self.v, feed_dict=feed_dict)

    def run(self, state):

        pi = np.squeeze(self.policy(state))
        v  = np.squeeze(self.value(state))

        return pi, v

class PPO:

    def __init__(self, action_space, state_space):

        self.experiences        = []
        self.experiences_size   = 1000.0

        self.batch          = [[], [], [], [], [], []]
        self.batch_size     = 32

        self.sess = tf.Session()
        self.sess.__enter__()

        self.model = Model(action_space, state_space)
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):

        pi, _ = self.model.run(state)

        # Get a random action
        a = np.random.choice(len(pi), 1, p=pi.flatten())[0]

        return a

    def store(self, s, s2, r, a, done):

        self.experiences.append((s, s2, r, a, done))

        # If no more memory for experiences, removes the first one
        if len(self.experiences) > self.experiences_size:
            self.experiences.pop(0)

        agent.train()

    def train(self):

        if len(self.experiences) < self.batch_size:
            return

        actions    = []
        states     = []
        discounts  = []
        advantages = []

        for i in range(self.batch_size):

            # Get experiances randomly
            exp = int(np.random.uniform(0, len(self.experiences)))
            s, s2, r, a, done = self.experiences[exp]

            states.append(np.squeeze(s, axis=0))
            actions.append([i, a])

            _, v = self.model.run(s)

            # Discount = r + gamma * V
            if done == True:
                q = r
            else:
                _, v2 = self.model.run(s2)
                q = r + self.model.gamma * v2

            # Advantage = Q - V
            adv = q - v

            discounts.append(q)
            advantages.append(adv)

        actions = np.reshape(np.asarray(actions), [self.batch_size, 2])

        self.model.update_policy(states, actions, advantages, discounts)
        self.model.update_value(states, discounts)

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)

    env = gym.make('MountainCar-v0')

    # Create an agent
    agent = PPO(env.action_space.n, env.observation_space.shape[0])

    episode = 0
    results = np.full(100, 200).tolist()

    while 1:

        s  = np.zeros((1, env.observation_space.shape[0]))
        s2 = np.zeros((1, env.observation_space.shape[0]))

        steps = 0

        s = env.reset()

        # Reshape for tensorflow
        s = s.astype(np.float32).reshape((1, -1))

        while 1:

            env.render()

            a = agent.get_action(s)

            s2, _, done, _ = env.step(a)

            # Reshape for tensorflow
            s2 = s2.astype(np.float32).reshape((1, -1))

            # Add a rewards for speed
            r = abs(s2[0][1]) / 0.07

            # Add reward for position
            if s2[0][0] >= 0.5:
                r += 1
            elif s2[0][0] >= 0.25:
                r += 0.2
            elif s2[0][0] >= 0.1:
                r += 0.1

            agent.store(s, s2, r, a, done)

            s = s2.copy()

            steps += 1

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
