import gym
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.losses as tfl

# References:
# https://arxiv.org/abs/1502.05477
# https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a
# https://en.wikipedia.org/wiki/Backtracking_line_search
# https://github.com/MahanFathi/TRPO-TensorFlow

class Model():

    def __init__(self, action_space, state_space):

        self.alpha = 5e-3
        self.gamma = 0.99
        self.delta = 1e-3
        self.epoch = 10

        self.state_space  = state_space
        self.action_space = action_space

        self.start = True;

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
            self.h1_pi = tf.layers.dense(self.states_pi, 128, tf.nn.relu, name="h1_pi")
            self.pi = tf.layers.dense(self.h1_pi, self.action_space, tf.nn.softmax, name="pi")

            # Weights used by the model and dimension of the network
            self.theta_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy')
            self.theta_policy_dim = [t.shape.as_list() for t in self.theta_policy]
            self.theta_policy_nbr = np.sum([np.prod(d) for d in self.theta_policy_dim])

            # NOTE: Surrate loss and Dkl are calculated with batches.

            # surrogate loss, consider the policy of the action taken
            self.surr = - tf.reduce_mean(self.advantages * \
                tf.gather_nd(self.pi, self.actions) / \
                (tf.gather_nd(self.pi_old, self.actions) + 1e-8))

            # Kullback–Leibler divergence
            self.kl = tf.reduce_mean(tf.reduce_sum(self.pi * \
                tf.log(tf.div(self.pi, self.pi_old + 1e-8) + 1e-8), axis=1))

            # NOTE: The weights of the network are updating by reshaping the
            # dimension as a vector. It is why pg, hvp and theta are shaped
            # in a 1-D vector
            self.flat_theta_policy = tf.concat([tf.reshape(t, [-1]) for t in self.theta_policy], axis=0)
            self.flat_theta_policy_updater = tf.placeholder(tf.float32, self.theta_policy_nbr)

            self.vector = \
                tf.placeholder(tf.float32, shape=(self.theta_policy_nbr,), name='vector')

            # Policy gradient, Reshape as vector
            grad = tf.gradients(self.surr, self.theta_policy)
            self.pg = tf.concat([tf.reshape(g, [-1]) for g in grad], axis=0)

            # First gradient for Hessian Vector Product
            self.grad1 = tf.gradients(self.kl, self.theta_policy)
            flatgrad = tf.concat([tf.reshape(g, [-1]) for g in self.grad1], axis=0)

            # Gradient vector product
            self.gvp = tf.reduce_sum(flatgrad * self.vector)

            # Hessian Vector Product, flatten to 1-D
            grad = tf.gradients(self.gvp, self.theta_policy)
            self.hvp = tf.concat([tf.reshape(g, [-1]) for g in grad], axis=0)

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
            self.true_v = tf.placeholder(tf.float32, (None,), 'true_v')

            # Create neural network
            self.h1_v = tf.layers.dense(self.states_v, 128, tf.nn.relu, name="h1_v")
            self.v  = tf.squeeze(tf.layers.dense(self.h1_v, 1, None, name="v"))

            self.theta_value = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'value')
            self.theta_value_dim = [t.shape.as_list() for t in self.theta_value]
            self.theta_value_nbr = np.sum([np.prod(d) for d in self.theta_value_dim])

            # MSE loss for value
            self.loss = tf.reduce_mean(tf.square(self.true_v - self.v))

            # Adam optimization for grdient descent
            self.adam_value = tf.train.AdamOptimizer(self.alpha).minimize(self.loss, var_list=self.theta_value)

    def get_flat_theta(self):
        return tf.get_default_session().run(self.flat_theta_policy)

    def cg(self, A, b, iterations=10, tolerance=1e-10):
        """
        Conjugate Gradient
        https://medium.com/@jonathan_hui/rl-conjugate-gradient-5a644459137a

        Solving Ax = b considering
        A = Hessian vector product
        b = surrogate gradient
        return x

        d is the direction (the conjugate vector) to go for the next move.
        alpha is how far should we move in direction d
        beta allows to have a direction A-orthogonal

        """

        r = b.copy()
        d = b.copy()
        x = np.zeros_like(b)

        rTr = r.dot(r)
        for i in range(iterations):

            Ad = A(d)

            alpha = rTr / d.dot(Ad)

            x = x + alpha * d
            r = r - alpha * Ad

            new_rTr = r.dot(r)

            beta = new_rTr / rTr

            d = r + beta * d

            rTr = new_rTr

            # Use rTr because the gradient given is all the weights of the model
            if rTr < tolerance:
                break

        return x

    def linesearch(self, get_policy_loss, x, step, grad, delta=1e-2, c=0.5, tau=0.5, iterations=10):

        """
        Backtracking line search
        https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a

        tau -> shrink parameter between 0 to 1
        step -> direction of local decrease
        i -> nbr steps

        surr -> f(x)
        surr_new -> f(x + i * step)
        """

        m = - grad.dot(step)

        surr, _ = get_policy_loss(x)

        for i in np.arange(iterations):

            x_new = x + (tau ** i) * step

            surr_new, kl_new = get_policy_loss(x_new)

            # if (surr - surr_new) > 0 and kl_new <= self.delta:
            #     return x_new

            actual_improve = (surr - surr_new)
            expected_improve = m * (tau ** i)
            ratio = actual_improve /expected_improve

            if ratio > 0.1 and actual_improve > 0:
                return x_new

        return x

    def assign_policy_theta(self, theta):

        feed_dict = {
            self.flat_theta_policy_updater: theta
        }

        tf.get_default_session().run(self.policy_updater, feed_dict=feed_dict)

    def update_policy(self, states, actions, advantages):

        feed_dict = {
            self.states_pi: states,
            self.actions: actions,
            self.advantages: advantages
        }

        theta_policy_curr = self.get_flat_theta()

        # Get the previous policy used during linear search
        pi_old = tf.get_default_session().run(self.pi, feed_dict)
        feed_dict[self.pi_old] = pi_old

        def get_pg():
            # Calculate policy gradient
            return tf.get_default_session().run(self.pg, feed_dict)

        def get_hvp(vector, damping=1e-3):
            # Calculate hessian vector product
            feed_dict[self.vector] = vector
            return tf.get_default_session().run(self.hvp, feed_dict) + damping * vector

        def get_policy_loss(theta):
            self.assign_policy_theta(theta)
            return tf.get_default_session().run([self.surr, self.kl], feed_dict)

        pg = get_pg()

        # Do nothing if the gradient is 0
        if np.allclose(pg, 0):
            return

        # Calculate conjugate gradient and gradient step
        x = self.cg(get_hvp, -pg)
        dx = np.sqrt(2 * self.delta / (x.dot(get_hvp(x)) + 1e-8)) * x

        # Get the new weights using linesearch
        theta_policy_new = self.linesearch(get_policy_loss, theta_policy_curr, dx, pg, self.delta)

        # Update the model with the new weights
        self.assign_policy_theta(theta_policy_new)

    def update_value(self, states, true_v):

        feed_dict = {
            self.states_v: states,
            self.true_v: true_v
        }

        for _ in range(self.epoch):
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

class TRPO:

    def __init__(self, action_space, state_space):

        self.experiences        = []
        self.experiences_size   = 200.0

        self.batch_size     = 32

        self.update_counter  = 0
        self.update_interval = 32

        self.sess = tf.Session()
        self.sess.__enter__()

        self.model = Model(action_space, state_space)
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):

        pi, v = self.model.run(state)

        # Get a random action
        a = np.random.choice(len(pi), 1, p=pi.flatten())[0]

        return a

    def store(self, s, s2, r, a, done):

        self.experiences.append((s, s2, r, a, done))

        # If no more memory for experiences, removes the first one
        if len(self.experiences) > self.experiences_size:
            self.experiences.pop(0)

        if self.update_counter > self.update_interval:
            agent.train()
            self.update_counter = 0;
            
        self.update_counter += 1

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

        self.model.update_policy(states, actions, advantages)
        self.model.update_value(states, discounts)

if __name__ == '__main__':

    # Start OpenAI environment
    env = gym.make('CartPole-v0')

    # Create an agent
    agent = TRPO(env.action_space.n, env.observation_space.shape[0])

    episode = 0
    results = []

    max_score = 0

    while 1:

        steps = 0

        s = env.reset()
        v = None

        # Reshape for tensorflow
        s = s.astype(np.float32).reshape((1, -1))

        while 1:

            # Get an action
            a = agent.get_action(s)

            # Perfom an action
            s2, r, done, _ = env.step(a)

            # Reshape for tensorflow
            s2 = s2.astype(np.float32).reshape((1, -1))

            # Store the experiance
            agent.store(s, s2, r, a, done)

            s = s2.copy()

            steps += 1

            if done:

                # Calcul the score total over 100 episodes
                results.append(steps)
                if len(results) > 100:
                    results.pop(0)

                score = np.sum(np.asarray(results)) / 100
                max_score= max(score, max_score)

                if score >= 195:
                    print("Finished!!!")
                    exit()

                episode += 1

                print("Episode", episode,
                      "finished after", steps,
                      "timesteps, score", score, max_score)

                break
