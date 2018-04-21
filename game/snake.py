# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

import pygame
from random import randint
import numpy as np
from pygame.locals import *
import sys

# General Parameters
SQUARE_SIZE   = 10
WINDOW_WIDTH  = 50
WINDOW_LENGTH = 50
WINDOW_COLOR  = (50, 50, 50)

MOVE_REWARD  = -1
FRUIT_REWARD = 10
LOSE_REWARD  = -100

FRUIT_COLOR  = (225, 80, 50)

SNAKE_INIT_LENGTH = 5 # SQUARES
SNAKE_GROWING     = 1 # SQUARES
SNAKE_INIT_POSX   = WINDOW_WIDTH  / 2 # SQUARES
SNAKE_INIT_POSY   = WINDOW_LENGTH / 2 # SQUARES
SNAKE_COLOR       = (226, 226, 226)

INIT_DIRECTION = "UP"

DIRECTIONS = {
    "UP":    ( 0, -1),
    "DOWN":  ( 0,  1),
    "LEFT":  (-1,  0),
    "RIGHT": ( 1,  0),
}

EVENT = {
    "TIMER":    USEREVENT + 0,
}

class Snake:

    dir       = ""
    body      = []      # index 0 is the head of the snake
    snake_img = None

    def __init__(self):

        self.dir = INIT_DIRECTION

        for i in range(SNAKE_INIT_LENGTH):
            x = SQUARE_SIZE * (SNAKE_INIT_POSX - DIRECTIONS[self.dir][0]*i)
            y = SQUARE_SIZE * (SNAKE_INIT_POSY - DIRECTIONS[self.dir][1]*i)

            self.body.append([x, y])

        self.snake_img = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
        self.snake_img.fill(SNAKE_COLOR)


        print "Snake initialized"

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
        if self.body[0][0] < 0 or self.body[0][0] >= WINDOW_WIDTH  * SQUARE_SIZE\
        or self.body[0][1] < 0 or self.body[0][1] >= WINDOW_LENGTH * SQUARE_SIZE:
            print "Snake out of boundaries"
            return False

        # Check if snake eats itself
        for i in range(len(self.body)-1):
            if self.body[0][0] == self.body[i+1][0] and\
               self.body[0][1] == self.body[i+1][1]:
               print "Snake ate itself"
               return False

        return True

    def newDir(self, dir):
        self.dir = dir

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

    def draw(self, w):
        for b in self.body:
            w.blit(self.snake_img, (b[0], b[1]))

class Fruit:

    posx      = 0
    posy      = 0
    fruit_img = None

    def __init__(self):
        self.newPosition()

        self.fruit_img = pygame.Surface((SQUARE_SIZE,SQUARE_SIZE))
        self.fruit_img.fill(FRUIT_COLOR)

        print "Fruit initialized"

    def reset(self):
        self.newPosition()

    def newPosition(self):
        self.posx = SQUARE_SIZE * randint(0, WINDOW_WIDTH-1)
        self.posy = SQUARE_SIZE * randint(0, WINDOW_LENGTH-1)

    def getPosition(self):
        return (self.posx, self.posy)

    def draw(self, w):
        w.blit(self.fruit_img, (self.posx, self.posy))

def run():

    print "Run"

    waitMove = False

    # Loop til doomsday
    while True:

        # Wait an event from pygame
        event = pygame.event.wait()

        # Close windows
        if event.type == QUIT:
            print "Quit"
            pygame.quit()

        # Change direction
        elif event.type == pygame.KEYDOWN and waitMove == False:
            print "Keystroke"

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
            print "Timer"

            if not snake.movement():
                pygame.quit()
                sys.exit(1)

            # Check if the snake has eaten a fruit
            if snake.getPosition() == fruit.getPosition():
                print "Fruit eaten"
                snake.growing()
                fruit.newPosition()

            waitMove = False

        # Redraw snake and screen
        window.fill(WINDOW_COLOR)
        snake.draw(window)
        fruit.draw(window)

        # Update display on screen
        print "Refresh Screen"
        pygame.display.flip()


if __name__ == "__main__":

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
    fruit = Fruit()

    # Init timer
    clk = pygame.time.Clock()
    pygame.time.set_timer(EVENT["TIMER"], 80)

    print "Game initialized"

    # Loop til doomsday
    while True:

        # Run snake
        run()
