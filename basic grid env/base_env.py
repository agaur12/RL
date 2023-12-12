import numpy as np
import random
import math
import gym
from gym import spaces
import math
import pygame as pg
import sys


class GridEnvironment(gym.Env):

    def __init__(self, size_x: int, size_y: int, rand_goal: bool = False, rand_start: bool = False):
        super(GridEnvironment, self).__init__()
        self.actions = ["up", "right", "down", "left"]
        self.rand_start = rand_start
        self.rand_goal = rand_goal
        self.max_x = size_x - 1
        self.max_y = size_y - 1
        self.done = False
        self.episode_length = 0
        self.distances = []
        self.max_episode_length = int(np.log(np.power(self.max_x * self.max_y, 5)) * int(np.power(20 * self.max_x * self.max_y, 0.25))) + 100.0
        print(self.max_episode_length)
        if rand_start:
            self.init_x, self.init_y = random.randint(1, self.max_x), random.randint(1, self.max_x)
            self.x, self.y = self.init_x, self.init_y
        else:
            self.x, self.y = 0, 0
        if rand_goal:
            self.goal = [random.randint(1, self.max_x), random.randint(1, self.max_x)]
            while self.goal == [self.x, self.y]:
                self.goal = [random.randint(1, self.max_x), random.randint(1, self.max_x)]
        else:
            self.goal = [self.max_x, self.max_y]
        self.state = [self.x, self.y]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=max(self.max_x, self.max_y), shape=(2,), dtype=np.float32)

    def reset(self):
        if self.rand_start:
            self.x, self.y = self.init_x, self.init_y
        else:
            self.x, self.y = 0, 0
        if self.rand_goal:
            self.goal = [random.randint(1, self.max_x), random.randint(1, self.max_x)]
        # self.x, self.y = random.randint(1, self.max_x), random.randint(1, self.max_x)
        """
        self.goal = [random.randint(1, self.max_x), random.randint(1, self.max_x)]
        while self.goal == [self.x, self.y]:
            self.goal = [random.randint(1, self.max_x), random.randint(1, self.max_x)]
        """
        self.done = False
        self.episode_length = 0
        self.state = [self.x, self.y]
        return np.array(self.state)

    def step(self, action):
        if self.state == self.goal or self.episode_length > 200:
            self.done = True

        self.action = self.actions[action]
        self.distances.append(math.hypot(self.state[0] - self.goal[0], self.state[1] - self.goal[1]))
        self.state = self.take_action()
        self.reward = self.get_reward()
        self.episode_length += 1

        if self.state == self.goal or self.episode_length > 200:
            self.done = True
        return self.state, self.reward, self.done

    def get_reward(self):
        if self.episode_length > (self.max_x * self.max_y):
            reward = -1
        elif self.state == self.goal:
            reward = 10
        else:
            reward = -0.5
            reward += (self.distances[-1] - math.hypot(self.state[0] - self.goal[0],
                                                       self.state[1] - self.goal[1])) / 100
        return reward

    def take_action(self):
        if (self.action == "left" and self.x == 0) or (self.action == "right" and self.x == self.max_x):
            self.x = self.x
            self.episode_length -= 1
        elif self.action == "left":
            self.x -= 1
        elif self.action == "right":
            self.x += 1
        else:
            self.x = self.x
        if (self.action == "down" and self.y == 0) or (self.action == "up" and self.y == self.max_x):
            self.y = self.y
            self.episode_length -= 1
        elif self.action == "down":
            self.y -= 1
        elif self.action == "up":
            self.y += 1
        else:
            self.y = self.y

        return [self.x, self.y]
    
    def render():
        pg.init()

        WIDTH, HEIGHT = 1000, 1000
        GRID_SIZE = int(1000 / (self.max_x + 1))
        AGENT_COLOR = (255, 0, 0)
        OBJECTIVE_COLOR = (0, 255, 0)
        GRID_COLOR = (255, 255, 255)

        screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Grid Game")

        agent_pos = self.state
        objective_pos = self.goal


        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            keys = pg.key.get_pressed()
            if keys[pg.K_a]:
                self.step(3)
            if keys[pg.K_d]:
                self.step(1)
            if keys[pg.K_s]:
                self.step(2)
            if keys[pg.K_w]:
                self.step(0)
            agent_pos = self.state

            screen.fill(GRID_COLOR)
            for x in range(0, WIDTH, GRID_SIZE):
                pg.draw.line(screen, (0, 0, 0), (x, HEIGHT), (x, 0), 1)
            for y in range(0, HEIGHT, GRID_SIZE):
                pg.draw.line(screen, (0, 0, 0), (0, HEIGHT - y), (WIDTH, HEIGHT - y), 1)

            pg.draw.rect(screen, AGENT_COLOR, ((agent_pos[0] * GRID_SIZE), (HEIGHT - (agent_pos[1] * GRID_SIZE) - GRID_SIZE), GRID_SIZE, GRID_SIZE))
            pg.draw.rect(screen, OBJECTIVE_COLOR, (objective_pos[0] * GRID_SIZE, (HEIGHT - (objective_pos[1] * GRID_SIZE) - GRID_SIZE), GRID_SIZE, GRID_SIZE))
            pg.display.flip()

            if agent_pos == objective_pos:
                self.reset()

            pg.time.Clock().tick(10)

        pg.quit()
        sys.exit()


