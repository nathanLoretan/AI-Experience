import gym
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

# References:
# https://arxiv.org/abs/1707.06347
# https://github.com/snowyukischnee/Pendulum_PPO/blob/master/pendulum.py

class Model():

    def __init__(self, state_space, action_space, action_range):

        self.alpha      = 1e-4
        self.gamma      = 0.9
        self.eps        = 2e-1
        self.ent_coef   = 1e-3
        self.value_coef = 1e-3
        self.epoch      = 10

        self.kl_target  = 0.01
        self.kl_coef    = 0.5

        self.state_space  = state_space
        self.action_space = action_space
        self.action_range = action_range
        self.action_scale = ((self.action_range[1] - self.action_range[0]) / 2)[0]

        self.states = \
            tf.placeholder(tf.float32, (None, self.state_space), 'state')
        self.actions = \
            tf.placeholder(tf.float32, (None, self.action_space), 'action')
        self.rewards = \
            tf.placeholder(tf.float32, (None, 1), 'reward')
        self.v2 = \
            tf.placeholder(tf.float32, (None, 1), 'v2')
        self.advantages = \
            tf.placeholder(tf.float32, (None, 1), 'advantage')
        self.v_loss = \
            tf.placeholder(tf.float32, (None), 'v_loss')

        with tf.variable_scope('policy'):

            self.beta = \
                tf.placeholder(tf.float32, None, name="beta")
            self.pi_mean_old = \
                tf.placeholder(tf.float32, (None, self.action_space), name="pi_mean_old")
            self.pi_var_old = \
                tf.placeholder(tf.float32, (None, self.action_space), name="pi_var_old")

            self.h1_pi = \
                tf.layers.dense(self.states, 128, tf.nn.relu)
            self.pi_mean = \
                tf.layers.dense(self.h1_pi, self.action_space, tf.nn.tanh) * self.action_scale
            self.pi_var = \
                tf.layers.dense(self.h1_pi, self.action_space, tf.nn.softplus,
                    kernel_initializer=tf.initializers.random_uniform(0.1, 1),
                    bias_initializer=tf.constant_initializer(0.1))

            self.pi_dist = \
                tf.distributions.Normal(loc=self.pi_mean, scale=self.pi_var)
            self.pi_dist_old = \
                tf.distributions.Normal(loc=self.pi_mean_old, scale=self.pi_var_old)

            self.pi = tf.squeeze(self.pi_dist.sample(1), axis=0)

            ratio = self.pi_dist.prob(self.actions) / \
                           (self.pi_dist_old.prob(self.actions) + 1e-8)

            clip = tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps)

            entropy = tf.reduce_mean(self.ent_coef * self.pi_dist.entropy())
            v_loss = self.value_coef * self.v_loss

            self.kl = tf.reduce_mean(tf.distributions.kl_divergence(self.pi_dist, self.pi_dist_old))
            self.surr = tf.reduce_mean(tf.multiply(self.advantages, ratio))
            self.loss_policy =  - self.surr + self.beta * self.kl #+ entropy - v_loss

            self.adam_policy = tf.train.AdamOptimizer(self.alpha).minimize(self.loss_policy)

        with tf.variable_scope('value'):

            self.h1_v = \
                tf.layers.dense(self.states, 128, tf.nn.relu)
            self.v = \
                tf.layers.dense(self.h1_v, 1, None)

            self.adv = self.rewards + self.gamma * self.v2 - self.v

            self.loss_value = tf.reduce_mean(tf.square(self.adv))
            self.adam_value = tf.train.AdamOptimizer(self.alpha).minimize(self.loss_value)


    def update(self, batch):

        states, next_states, actions, rewards, _ = zip(*batch)

        states      = np.vstack(states)
        actions     = np.vstack(actions)
        next_states = np.vstack(next_states)
        rewards     = np.vstack(rewards)

        feed_dict = {
            self.states: next_states,
        }

        v2 = tf.get_default_session().run(self.v, feed_dict=feed_dict)

        feed_dict[self.states] = states
        feed_dict[self.rewards] = rewards
        feed_dict[self.v2] = v2

        advantages = tf.get_default_session().run(self.adv, feed_dict=feed_dict)

        feed_dict[self.actions] = actions
        feed_dict[self.advantages] = advantages

        pi_mean_old, pi_var_old = \
            tf.get_default_session().run([self.pi_mean, self.pi_var], feed_dict=feed_dict)

        v_loss = tf.get_default_session().run(self.loss_value, feed_dict=feed_dict)

        feed_dict[self.pi_mean_old] = pi_mean_old
        feed_dict[self.pi_var_old] = pi_var_old
        feed_dict[self.beta] = self.kl_coef
        feed_dict[self.v_loss] = v_loss

        for _ in range(self.epoch):
            tf.get_default_session().run(self.adam_policy, feed_dict=feed_dict)

        for _ in range(self.epoch):
            tf.get_default_session().run(self.adam_value, feed_dict=feed_dict)

        # Calculate the new kl coefficient
        kl = tf.get_default_session().run(self.kl, feed_dict=feed_dict)

        if kl < self.kl_target / 1.5:
            self.kl_coef /= 2
        elif kl > self.kl_target * 1.5:
            self.kl_coef *= 2

        self.kl_coef = np.clip(self.kl_coef, 1e-4, 10)

    def run(self, state):

        feed_dict = {
            self.states: state
        }

        pi = tf.get_default_session().run(self.pi, feed_dict=feed_dict)

        return np.clip(np.squeeze(pi), self.action_range[0], self.action_range[1])

class PPO:

    def __init__(self, action_space, action_range, state_space):

        self.experiences        = []
        self.experiences_size   = 200

        self.update_cnt = 0

        self.batch_size      = 32
        self.update_interval = 32

        self.sess = tf.Session()
        self.sess.__enter__()

        self.model = Model(state_space, action_space, action_range)
        tf.get_default_session().run(tf.global_variables_initializer())

    def get_action(self, state):
        return self.model.run(state)

    def store(self, s, s2, r, a, done):

        self.experiences.append((s, s2, a, r, done))

        # If no more memory for experiences, removes the first one
        if len(self.experiences) > self.experiences_size:
            self.experiences.pop(0)

        if self.update_cnt >= self.update_interval:
            agent.train()
            self.update_cnt = 0

        self.update_cnt += 1

    def train(self):

        if len(self.experiences) < self.batch_size:
            return

        actions    = []
        states     = []
        discounts  = []
        advantages = []
        experience = []

        for _ in range(self.batch_size):
            exp = int(np.random.uniform(0, len(self.experiences)))
            experience.append(self.experiences[exp])

        self.model.update(experience)

if __name__ == '__main__':

    # Start OpenAI environment
    env = gym.make('Pendulum-v0')

    # Create an agent
    agent = PPO(env.action_space.shape[0],
               [env.action_space.low, env.action_space.high],
                env.observation_space.shape[0])

    episode = 0
    results = np.full(100, -2000).tolist()

    while 1:

        rewards = 0

        s = env.reset()

        # Reshape for tensorflow
        s = s.astype(np.float32).reshape((1, -1))

        while 1:

            # env.render()

            a = agent.get_action(s)

            s2, r, done, _ = env.step(a)

            # Reshape for tensorflow
            s2 = s2.astype(np.float32).reshape((1, -1))

            rewards += r

            # Normalize the reward
            r = (r / 16.2736044) + 0.5

            agent.store(s, s2, r, a, done)

            s = s2.copy()

            if done:

                # Calcul the score total over 100 episodes
                results.append(rewards)
                if len(results) > 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100

                if score >= -300:
                    print("Finished!!!")
                    exit()

                episode += 1

                print("Episode", episode,
                      "rewards", rewards,
                      "score", score)

                break
