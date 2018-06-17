# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

import sys
import pygame
import numpy as np
from math import sqrt, exp
from random import randint
from pygame.locals import *
from collections import defaultdict

class QLearning():

    def __init__(self, A, Ne, Rplus, alpha=(lambda n: 1./(1+n)), gamma=0.9):
        """Q-Learning for reinforcement learning"""

        # Ne        Number of time needed before considering a state as known
        # Rplus     explorating reward estimated of unkown states
        # alpha     learning factor
        # gamma     reward factor
        # Rplus     Reward used to initiate exploration
        # Q         action values of state/action
        # Nsa       frequencies of state/action pairs
        # s         state
        # a         action
        # r         reward
        # u         utility
        # A         list of possible actions

        self.gamma    = gamma
        self.alpha    = alpha
        self.Rplus    = Rplus
        self.Ne       = Ne
        self.A        = A

        self.Q        = defaultdict(float)
        self.Nsa      = defaultdict(float)
        self.s        = None
        self.a        = None


    def f(self, u, n):
        """ Exploration function. Returns fixed Rplus untill
            agent has visited state, action a Ne number of times.
            Same as ADP agent in book.

            [in] u      : utility of the state = Q(s, a)
            [in] n      : Number of time the state has been visited """

        if n < self.Ne:
            return self.Rplus
        else:
            return u

    def reset(self):
        """ Reset the different elements needed to allow the agent to run
        correctly the next trial. """

        self.s = None
        self.a = None
        self.r = None

    def __call__(self, s, r):
        """ Learn the Q(s,a) of the current state and return the next action
            the agent must perform.

            [in] s : state
            [in] r : reward """

        ns = s # next state

        # Get reference for fancier notation
        s = self.s
        a = self.a
        A = self.A

        Q   = self.Q
        Nsa = self.Nsa
        alpha = self.alpha
        gamma = self.gamma

        # Check if the agent has already visited the previous state s
        # and update Q matrix
        if s is not None:
            Nsa[s, a] += 1
            Q[s, a]   += alpha(Nsa[s, a]) * \
                           (r + gamma * np.max([Q[ns, na] for na in A]) - \
                            Q[s, a])

        # Get the next action to perform
        na = np.argmax([self.f(Q[ns, na], Nsa[ns, na]) for na in A])

        # Save the next state and next action
        self.s, self.a = ns, na

        # Return new action
        return na

# ------------------------------------------------------------------------------

VERBOSE = True

# General Parameters
SQUARE_SIZE   = 10
WINDOW_WIDTH  = 30
WINDOW_LENGTH = 30
WINDOW_COLOR  = (50, 50, 50)

MOVE_REWARD_POS  =  0 # Reward for good move
MOVE_REWARD_NEG  = -1 # Reward for bad move
FRUIT_REWARD     =  sys.maxint
LOSE_REWARD      = -sys.maxint - 1

TIMEOUT          = 20000 # 20s, avoid the snake to be blocked
ACTION_TIME      = 10 #ms

FRUIT_COLOR  = (225, 80, 50)

SNAKE_INIT_LENGTH = 5 # SQUARES
SNAKE_GROWING     = 1 # SQUARES
SNAKE_INIT_POSX   = WINDOW_WIDTH  / 2 # SQUARES
SNAKE_INIT_POSY   = WINDOW_LENGTH / 2 # SQUARES
SNAKE_COLOR       = (226, 226, 226)
SNAKE_COLOR_HEAD  = (150, 150, 150)

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

MOVE = ["TURN_RIGHT", "TURN_LEFT", "STAY_STRAIGHT"]

EVENT = {
    "TIMER":    USEREVENT + 0,
}

pd = 0 # previous distance snake and fruit

tr = 0 # time reward, the reward decrease with time to reach a fruit, it avoids
       # the snake to get lock and move in a circle

class Snake:

    dir        = ""
    body       = []      # index 0 is the head of the snake
    snake_body = None
    snake_head   = None

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

        for i in range(len(self.body)-1, 0, -1):
            self.body[i][0] = self.body[i-1][0]
            self.body[i][1] = self.body[i-1][1]

        x = DIRECTIONS[self.dir][0] * SQUARE_SIZE
        y = DIRECTIONS[self.dir][1] * SQUARE_SIZE

        self.body[0][0] += x
        self.body[0][1] += y

        # Check if the snake went out of the window
        if self.body[0][0] < 0:
            print "Snake left wall"
            return False

        elif self.body[0][0] >= WINDOW_WIDTH  * SQUARE_SIZE:
            print "Snake right wall"
            return False

        elif self.body[0][1] < 0:
            print "Snake bottom walls"
            return False

        elif self.body[0][1] >= WINDOW_LENGTH * SQUARE_SIZE:
            print "Snake top wall"
            return False

        # Check if snake eats itself
        for i in range(len(self.body)-1):
            if self.body[0][0] == self.body[i+1][0] and\
               self.body[0][1] == self.body[i+1][1]:
               print "Snake ate itself"
               return False

        return True

    def checkWall(self):

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
                    WINDOW_LENGTH * SQUARE_SIZE:
                wall[a] = 1

        return wall

    def checkBody(self):

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

    def newDir(self, _dir):
        self.dir = _dir

    def getPosition(self):
        return (self.body[0][0], self.body[0][1])

    def getDir(self):
        return self.dir

    def growing(self):

        last  = self.body[len(self.body)-1] # last part of body
        bLast = self.body[len(self.body)-1] # Before last part of body

        dir = np.asarray(last) - np.asarray(bLast)

        self.body.append([last[0] + dir[0],
                          last[1] + dir[1]])

    def draw(self, _w):
        for b in self.body:
            if b == self.body[0]:
                _w.blit(self.snake_head, (b[0], b[1]))
            else:
                _w.blit(self.snake_body, (b[0], b[1]))

class Fruit:

    posx      = 0
    posy      = 0
    fruit_img = None

    def __init__(self, snake):

        self.snake = snake

        self.newPosition()

        self.fruit_img = pygame.Surface((SQUARE_SIZE,SQUARE_SIZE))
        self.fruit_img.fill(FRUIT_COLOR)

    def reset(self):
        self.newPosition()

    def newPosition(self):
        self.posx = SQUARE_SIZE * randint(0, WINDOW_WIDTH-1)
        self.posy = SQUARE_SIZE * randint(0, WINDOW_LENGTH-1)

        # Check if the new fruit is on the position than the snake
        for i in range(len(self.snake.body)):
            if  self.snake.body[i][0] == self.posx and \
                self.snake.body[i][1] == self.posy:
                self.newPosition()

    def getPosition(self):
        return (self.posx, self.posy)

    def draw(self, w):
        w.blit(self.fruit_img, (self.posx, self.posy))

def getReward(snake, fruit):

    global pd
    global tr

    # The rewards is positif if the snake is nearer from the fruit and negatif
    # if it is further from the fruit
    snake_pos = snake.getPosition()
    fruit_pos = fruit.getPosition()

    # Calcul the distance between fruit and snake
    d = sqrt(abs(snake_pos[0] - fruit_pos[0])**2 +\
             abs(snake_pos[1] - fruit_pos[1])**2)

    # If new distance smaller than previous, positif reward
    if (pd - d) >= 0:
        r = MOVE_REWARD_POS
    else:
        r = MOVE_REWARD_NEG + tr

    # Save distance
    pd = d

    return r

def getState(snake, fruit):

    # state define by object near the head of the snake (body or wall) and the
    # position of the fruit with respect to the head
    # Possible number of state = 2^3 * 3^2

    wall = snake.checkWall()
    body = snake.checkBody()

    snake_pos = snake.getPosition()
    fruit_pos = fruit.getPosition()

    # Obstacle is a binary value, from DIRS_ACTION r=0 l=1 s=2
    obsr = wall[0] ^ body[0]    # obstacle on the current right
    obsl = wall[1] ^ body[1]    # obstacle on the current left
    obsf = wall[2] ^ body[2]    # obstacle on the current front

    # The position x/y can be either 0,1,-1
    posx = 0
    posy = 0

    if snake_pos[0] - fruit_pos[0] < 0:
        posx = 1
    elif snake_pos[0] - fruit_pos[0] > 0:
        posx = -1
    elif snake_pos[0] - fruit_pos[0] == 0:
        posx = 0

    if snake_pos[1] - fruit_pos[1] < 0:
        posy = 1
    elif snake_pos[1] - fruit_pos[1] > 0:
        posy = -1
    elif snake_pos[1] - fruit_pos[1] == 0:
        posy = 0

    return (obsr, obsl, obsf, posx, posy)

def run(agent, snake, fruit):

    global tr   # time reward
    global pd   # previous snake-fruit distance

    pd = 0
    tr = 0
    cnt = 0 # Check for timeout

    action = "UP"

    # Loop til doomsday
    while True:

        rewards = 0
        state = 0

        # Wait an event from pygame
        event = pygame.event.wait()

        # Close windows
        if event.type == QUIT:
            print("Quit")
            pygame.quit()

        # Change direction
        elif event.type == pygame.KEYDOWN and waitMove == False:

            if event.key == pygame.K_UP and snake.getDir() != "DOWN":
                snake.newDir("UP")
            elif event.key == pygame.K_DOWN and snake.getDir() != "UP":
                snake.newDir("DOWN")
            elif event.key == pygame.K_LEFT and snake.getDir() != "RIGHT":
                snake.newDir("LEFT")
            elif event.key == pygame.K_RIGHT and snake.getDir() != "LEFT":
                snake.newDir("RIGHT")

            # Wait to move before allowing to rechange the direction
            waitMove = True

        # Timer to move
        elif event.type == EVENT["TIMER"]:

            # Perform action recommended by the agent
            snake.newDir(action)

            if not snake.movement():
                # Update state rewards for the agent
                rewards += LOSE_REWARD
                agent(getState(snake, fruit), rewards)

                return False

            rewards += getReward(snake, fruit)

            # Check if the snake has eaten a fruit
            if snake.getPosition() == fruit.getPosition():
                snake.growing()
                fruit.newPosition()

                tr = MOVE_REWARD_NEG  # Reset time reward
                rewards += FRUIT_REWARD

                cnt = 0;    # Check for timeout

            else:
                tr += MOVE_REWARD_NEG # Decrease time reward

                cnt += 1
                if cnt >= float(TIMEOUT)/ACTION_TIME:
                    print "Timeout"
                    return False

            waitMove = False

            # Update state rewards for the agent
            action = DIRS_ACTION[action][agent(getState(snake, fruit), rewards)]

        # Redraw snake and screen
        window.fill(WINDOW_COLOR)
        snake.draw(window)
        fruit.draw(window)

        # Update display on screen
        pygame.display.flip()

if __name__ == "__main__":

    alpha_cst = (lambda n: 0.1)
    alpha_decreasing = (lambda n: 100./(99.+n))

    # Initate AI Agent, actions: "RIGHT" = 0 "LEFT" = 1 "STRAIGHT" = 2
    q_learner = QLearning(A     = [0,1,2],
                          Ne    = 100,
                          Rplus = 1000,
                          alpha = alpha_cst,
                          gamma = 0.9)

    # init Pygame
    pygame.init()

    # initialize window
    window = pygame.display.set_mode((WINDOW_WIDTH  * SQUARE_SIZE,
                                      WINDOW_LENGTH * SQUARE_SIZE))
    window.fill(WINDOW_COLOR)

    # Title of the window
    pygame.display.set_caption('Snake')

    # create snake
    snake = Snake()

    # place one fruit
    fruit = Fruit(snake)

    # Init timer
    clk = pygame.time.Clock()
    pygame.time.set_timer(EVENT["TIMER"], ACTION_TIME)

    print("Game initialized")

    game_cnt = 0

    # Loop til doomsday
    while True:

        game_cnt += 1
        print "NEW GAME: " + str(game_cnt)

        # Run snake
        run(q_learner, snake, fruit)

        # Restart agent
        q_learner.reset()

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
