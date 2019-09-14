import gym
import pygame
import logging
import threading
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from keras import regularizers as rl

from pygame.locals import *
from math import sqrt, exp
from random import randint
from collections import defaultdict
from tensorflow.keras.models import load_model

class Model(tf.keras.Model):
    def __init__(self, nbr_actions):
        super().__init__('mlp_policy')

        # 100x100x1

        self.hidden1 = kl.Conv2D(4, (3, 3), activation='relu', input_shape=(10, 10, 1))
        self.hidden2 = kl.MaxPooling2D((2, 2), strides=(2, 2))

        # 50x50x16

        self.hidden3 = kl.Conv2D(8, (3, 3), activation='relu')
        self.hidden4 = kl.MaxPooling2D((2, 2), strides=(2, 2))

        # 25x25x32

        self.hidden5 = kl.Conv2D(16, (3, 3), activation='relu')

        # 25x25x64

        self.hidden6 = kl.Conv2D(16, (3, 3), activation='relu')

        # 25x25x16

        self.hidden7 = kl.Flatten()
        self.hidden8 = kl.Dense(128, activation='relu')
        self.q        = kl.Dense(nbr_actions, name='q')

        # # 100x100x1
        #
        # self.hidden1 = kl.Conv2D(16, (3, 3), activation='relu', input_shape=(10, 10, 1))
        # self.hidden2 = kl.BatchNormalization()
        # self.hidden3 = kl.MaxPooling2D((2, 2), strides=(2, 2))
        #
        # # 50x50x16
        #
        # self.hidden4 = kl.Conv2D(32, (3, 3), activation='relu')
        # self.hidden5 = kl.BatchNormalization()
        # self.hidden6 = kl.MaxPooling2D((2, 2), strides=(2, 2))
        #
        # # 25x25x32
        #
        # self.hidden7 = kl.Conv2D(64, (3, 3), activation='relu')
        # self.hidden8 = kl.BatchNormalization()
        #
        # # 25x25x64
        #
        # self.hidden9 = kl.Conv2D(16, (3, 3), activation='relu')
        # self.hidden10 = kl.BatchNormalization()
        #
        # # 25x25x16
        #
        # self.hidden11 = kl.Flatten()
        # self.hidden12 = kl.Dense(128, activation='relu')
        # self.q        = kl.Dense(nbr_actions, name='q')

    def call(self, inputs):

        x = tf.convert_to_tensor(inputs)

        hidden1 = self.hidden1(x)
        hidden2 = self.hidden2(hidden1)
        hidden3 = self.hidden3(hidden2)
        hidden4 = self.hidden4(hidden3)
        hidden5 = self.hidden5(hidden4)
        hidden6 = self.hidden6(hidden5)
        hidden7 = self.hidden7(hidden6)
        hidden8 = self.hidden8(hidden7)
        q = self.q(hidden8)

        # hidden1 = self.hidden1(x)
        # hidden2 = self.hidden2(hidden1)
        # hidden3 = self.hidden3(hidden2)
        # hidden4 = self.hidden4(hidden3)
        # hidden5 = self.hidden5(hidden4)
        # hidden6 = self.hidden6(hidden5)
        # hidden7 = self.hidden7(hidden6)
        # hidden8 = self.hidden8(hidden7)
        # hidden9 = self.hidden9(hidden8)
        # hidden10 = self.hidden10(hidden9)
        # hidden11 = self.hidden11(hidden10)
        # hidden12 = self.hidden12(hidden11)
        # q = self.q(hidden12)

        return q

    def q_value(self, state):

        q = self.predict(state)
        return np.squeeze(q)

class DQN:

    def __init__(self, model):

        self.experiences        = []
        self.experiences_size   = 5000.0

        self.alpha          = 0.001
        self.gamma          = 0.99
        self.entropy_coef   = 0.001
        self.epsilon        = 1.0
        self.epsilon_min    = 0.01
        self.epsilon_decay  = 0.9995
        self.batch_size     = 32

        self.model = model

        try:
            self.model.load_weights("AI-Snake.h5")
        except:
            self.model.compile(optimizer=ko.Adam(self.alpha),
                               loss="MSE")

    def get_action(self, state, explore):

        q = self.model.q_value(state)

        # e-Greedy explore
        greedy = np.full(len(q), self.epsilon / (len(q)-1))
        greedy[np.argmax(q)] = 1.0 - self.epsilon

        print(np.argmax(q), q, greedy)

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

        # If no more memory for experiences, removes the first one
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

        # self.model.save_weights("AI-Snake.h5")

    def loss(self, true, prediction):

        # 1/2 * (R + gamma * Qplus - Q)^2
        return  0.5 * kls.mean_squared_error(true, prediction)

# General Parameters
SQUARE_SIZE    = 10
WINDOW_WIDTH   = 10
WINDOW_HEIGHT  = 10
WINDOW_COLOR   = (255, 255, 255)

MOVE_REWARD_POS  = 0.1e0 # Reward for good move
MOVE_REWARD_NEG  = -0.1e0 # Reward for bad move
LOSE_REWARD      = -1e0
FRUIT_REWARD     =  1e0

FRUIT_COLOR  = (0, 0, 0)

SNAKE_INIT_LENGTH = 5 # SQUARES
SNAKE_GROWING     = 1 # SQUARES
SNAKE_INIT_POSX   = WINDOW_WIDTH  / 2 # SQUARES
SNAKE_INIT_POSY   = WINDOW_HEIGHT / 2 # SQUARES
SNAKE_COLOR       = (120, 120, 120)
SNAKE_COLOR_HEAD  = (200, 200, 200)

ACTION_TIME      = 1 #ms

INIT_DIRECTION = "UP"

DIRECTIONS = {
    "UP":    ( 0, -1),
    "DOWN":  ( 0,  1),
    "LEFT":  (-1,  0),
    "RIGHT": ( 1,  0),
}
                   # TURN: RIGHT=0  |   LEFT = 1   |  STRAIGHT = 2
DIRS_ACTION = {"UP":    [ "RIGHT",      "LEFT",         "UP"],
               "DOWN":  [ "LEFT",       "RIGHT",        "DOWN"],
               "LEFT":  [ "UP",         "DOWN",         "LEFT"],
               "RIGHT": [ "DOWN",       "UP",           "RIGHT"]}

EVENT = {
    "TIMER":    USEREVENT + 0,
    "START":    USEREVENT + 1,
}

# previous distance snake and fruit
prev_distance = 0

class Snake:

    dir        = ""
    body       = []      # index 0 is the head of the snake
    snake_body = None
    snake_head = None

    def __init__(self):

        self.dir = INIT_DIRECTION

        for i in range(SNAKE_INIT_LENGTH):
            x = SQUARE_SIZE * (SNAKE_INIT_POSX - DIRECTIONS[self.dir][0]*i)
            y = SQUARE_SIZE * (SNAKE_INIT_POSY - DIRECTIONS[self.dir][1]*i)

            self.body.append([x, y])

        self.snake_body = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
        self.snake_body.fill(SNAKE_COLOR)

        self.snake_head = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
        self.snake_head.fill(SNAKE_COLOR_HEAD)

    def reset(self):

        self.body = []

        self.dir = INIT_DIRECTION

        for i in range(SNAKE_INIT_LENGTH):
            x = SQUARE_SIZE * (SNAKE_INIT_POSX - DIRECTIONS[self.dir][0]*i)
            y = SQUARE_SIZE * (SNAKE_INIT_POSY - DIRECTIONS[self.dir][1]*i)

            self.body.append([x, y])

    def movement(self):

        # Move the snake
        for i in range(len(self.body)-1, 0, -1):
            self.body[i][0] = self.body[i-1][0]
            self.body[i][1] = self.body[i-1][1]

        x = DIRECTIONS[self.dir][0] * SQUARE_SIZE
        y = DIRECTIONS[self.dir][1] * SQUARE_SIZE

        self.body[0][0] += x
        self.body[0][1] += y

        # Check if the snake went out of the window
        if self.body[0][0] < 0:
            # left wall
            return False

        elif self.body[0][0] >= WINDOW_WIDTH  * SQUARE_SIZE:
            # Right wall
            return False

        elif self.body[0][1] < 0:
            # Bottom walls
            return False

        elif self.body[0][1] >= WINDOW_HEIGHT * SQUARE_SIZE:
            # Top wall
            return False

        # Check if snake eats itself
        for i in range(len(self.body)-1):
            if self.body[0][0] == self.body[i+1][0] and\
               self.body[0][1] == self.body[i+1][1]:
               # Snake ate itself
               return False

        return True

    def check_wall(self):

        wall = [0,0,0]
        actions = DIRS_ACTION[self.dir]

        # Check for each possible next direction if a collision may occured
        for a in range(len(actions)):

            dir = DIRECTIONS[actions[a]]

            if self.body[0][0] + (dir[0] * SQUARE_SIZE) < 0\
            or self.body[0][0] + (dir[0] * SQUARE_SIZE) >= \
                    WINDOW_WIDTH * SQUARE_SIZE\
            or self.body[0][1] + (dir[1] * SQUARE_SIZE) < 0\
            or self.body[0][1] + (dir[1] * SQUARE_SIZE) >= \
                    WINDOW_HEIGHT * SQUARE_SIZE:
                wall[a] = 1

        return wall

    def check_body(self):

        body_part = [0,0,0]
        actions = DIRS_ACTION[self.dir]

        # Check if snake eats itself for each next direction
        for a in range(len(actions)):

            dir = DIRECTIONS[actions[a]]

            for i in range(len(self.body)-1):
                    if  self.body[0][0] + (dir[0] * SQUARE_SIZE) == \
                            self.body[i+1][0]\
                    and self.body[0][1] + (dir[1] * SQUARE_SIZE) == \
                            self.body[i+1][1]:
                        body_part[a] = 1;

        return body_part

    def new_dir(self, dir):
        self.dir = dir

    def get_position(self):
        return (self.body[0][0], self.body[0][1])

    def get_dir(self):
        return self.dir

    def growing(self):

        last   = self.body[len(self.body)-1] # last part of body
        b_last = self.body[len(self.body)-1] # Before last part of body

        dir = np.asarray(last) - np.asarray(b_last)

        self.body.append([last[0] + dir[0],
                          last[1] + dir[1]])

    def draw(self, w):
        for b in self.body:
            if b == self.body[0]:
                w.blit(self.snake_head, (b[0], b[1]))
            else:
                w.blit(self.snake_body, (b[0], b[1]))

class Fruit:

    posx      = 0
    posy      = 0
    snake     = None
    fruit_img = None

    def __init__(self, snake):

        self.snake = snake

        self.new_position()

        self.fruit_img = pygame.Surface((SQUARE_SIZE,SQUARE_SIZE))
        self.fruit_img.fill(FRUIT_COLOR)

    def reset(self):
        self.new_position()

    def new_position(self):
        self.posx = SQUARE_SIZE * randint(0, WINDOW_WIDTH-1)
        self.posy = SQUARE_SIZE * randint(0, WINDOW_HEIGHT-1)

        # Check if the new fruit is on the position than the snake
        for i in range(len(self.snake.body)):
            if  self.snake.body[i][0] == self.posx and \
                self.snake.body[i][1] == self.posy:
                self.new_position()

    def get_position(self):
        return (self.posx, self.posy)

    def draw(self, w):
        w.blit(self.fruit_img, (self.posx, self.posy))

def get_step_reward(snake, fruit):

    global prev_distance

    # The rewards is positif if the snake is nearer from the fruit and negatif
    # if it is further from the fruit
    snake_pos = snake.get_position()
    fruit_pos = fruit.get_position()

    # Calcul the distance between fruit and snake
    distance = sqrt((snake_pos[0] - fruit_pos[0])**2 +\
                    (snake_pos[1] - fruit_pos[1])**2)

    # If new distance smaller than previous, positif reward
    if (prev_distance - distance) > 0:
        r = MOVE_REWARD_POS
    else:
        r = MOVE_REWARD_NEG

    # Save distance
    prev_distance = distance

    return r

def get_state(snake, fruit):

    surface  = pygame.display.get_surface()
    color    = pygame.surfarray.pixels_red(surface)

    state    = np.zeros((1, WINDOW_WIDTH * SQUARE_SIZE,
                            WINDOW_HEIGHT * SQUARE_SIZE, 1))

    for x in range(WINDOW_WIDTH * SQUARE_SIZE):
        for y in range(WINDOW_HEIGHT * SQUARE_SIZE):
            state[0, x, y, 0] = color[x][y] / np.sum(color)

    return state

def run(agent, snake, fruit):

    action = "UP"
    first = True

    # Redraw snake and screen
    window.fill(WINDOW_COLOR)
    snake.draw(window)
    fruit.draw(window)

    # Update display on screen
    pygame.display.flip()

    # Get the initial state
    state = get_state(snake, fruit)

    score = 0

    # Loop til doomsday
    while True:

        next = agent.get_action(state, True)
        action = DIRS_ACTION[action][next]

        reward = 0

        # Wait an event from pygame
        event = pygame.event.wait()

        # Close windows
        if event.type == QUIT:
            print("Quit")
            pygame.quit()

        # Timer to move
        elif event.type is not EVENT["TIMER"]:
            continue

        # Perform action recommended by the agent
        snake.new_dir(action)

        # Move the snake and check if it touch itself or the wall
        if not snake.movement():

            # Update state rewards for the agent
            reward += LOSE_REWARD

            # Redraw snake and screen
            window.fill(WINDOW_COLOR)
            snake.draw(window)
            fruit.draw(window)

            # Update display on screen
            pygame.display.flip()

            state2 = get_state(snake, fruit)

            # Add the final losing state
            agent.store((state, state2, reward, next, True))
            agent.train()

            return score

        # Check if the snake has eaten a fruit
        if snake.get_position() == fruit.get_position():

            snake.growing()
            fruit.new_position()

            reward += FRUIT_REWARD

            score += 1

            print("The snake ate a FRUIT!!!")

        # Rewards regarding the distance of the snake
        reward += get_step_reward(snake, fruit)

        # Redraw snake and screen
        window.fill(WINDOW_COLOR)
        snake.draw(window)
        fruit.draw(window)

        # Update display on screen
        pygame.display.flip()

        # Get new state
        state2 = get_state(snake, fruit)

        agent.store((state, state2, reward, next, False))
        agent.train()

        state = state2.copy()

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)

    # init Pygame
    pygame.init()

    # initialize window
    window = pygame.display.set_mode((WINDOW_WIDTH  * SQUARE_SIZE,
                                      WINDOW_HEIGHT * SQUARE_SIZE))
    window.fill(WINDOW_COLOR)

    # Title of the window
    pygame.display.set_caption('Snake')

    # create snake
    snake = Snake()

    # place one fruit
    fruit = Fruit(snake)

    # Create an agent
    model = Model(3)
    agent = DQN(model)

    # Init timer
    clk = pygame.time.Clock()
    pygame.time.set_timer(EVENT["TIMER"], ACTION_TIME)

    game_cnt = 0

    print("Game initialized")

    episode = 0
    results = []

    # Loop til doomsday
    while True:

        # Run snake
        score = run(agent, snake, fruit)

        results.append(score)
        if len(results) > 100:
            results.pop(0)

        total_score = np.sum(np.asarray(results)) / 100

        if total_score >= 20:
            print("Finished!!!")
            exit()

        episode += 1

        print("Episode", episode,
              "fruits", score,
              "timesteps, score", total_score)

        # create snake
        snake.reset()

        # place one fruit
        fruit.reset()

        # Redraw snake and screen
        window.fill(WINDOW_COLOR)
        snake.draw(window)
        fruit.draw(window)

        # Update display on screen
        pygame.display.flip()
