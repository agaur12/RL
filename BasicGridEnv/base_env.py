import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces
import math
import pygame as pg
import sys


class GridEnvironment(gym.Env):
    metadata = {
        'render.modes': ['human']
    }
    render_mode = 'human'
    def __init__(self, size_x: int, size_y: int, rand_goal: bool = False, rand_start: bool = False, agent: bool = True):
        super(GridEnvironment, self).__init__()
        self.actions = ["up", "right", "down", "left"]
        self.rand_start = rand_start
        self.rand_goal = rand_goal
        self.max_x = size_x - 1
        self.max_y = size_y - 1
        self.done = False
        self.episode_length = 0

        self.goals = []
        self.starts = []
        self.agent = agent
        if rand_goal:
            self.goal = [random.randint(1, self.max_x), random.randint(1, self.max_x)]
        else:
            self.goal = [self.max_x, self.max_y]
        if rand_start:
            self.x, self.y = random.randint(1, self.max_x), random.randint(1, self.max_x)
            while math.dist([self.x, self.y], self.goal) < 2:
                self.x, self.y = random.randint(1, self.max_x), random.randint(1, self.max_x)
        else:
            self.x, self.y = 0, 0
        self.goals.append(self.goal)
        self.starts.append(self.starts)
        self.state = [self.x, self.y]
        self.distances = [math.dist(self.state, self.goal)]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=max(self.max_x, self.max_y), shape=(2,), dtype=np.float32)

    def reset(self):
        if self.rand_goal:
            self.goal = [random.randint(1, self.max_x), random.randint(1, self.max_x)]
        if self.rand_start:
            self.x, self.y = random.randint(1, self.max_x), random.randint(1, self.max_x)
            while math.dist([self.x, self.y], self.goal) < 2:
                self.x, self.y = random.randint(1, self.max_x), random.randint(1, self.max_x)
        else:
            self.x, self.y = 0, 0
        self.done = False
        self.episode_length = 0
        self.state = [self.x, self.y]
        return np.array(self.state)

    def step(self, action):
        if self.state == self.goal:
            self.done = True

        self.action = self.actions[action]
        self.distances.append(math.dist(self.state, self.goal))
        self.state = self.take_action()
        self.reward = self.get_reward()
        self.episode_length += 1

        if self.state == self.goal:
            self.done = True
        return self.state, self.reward, self.done

    def get_reward(self):
        TIME_PENALTY = -0.05
        PROGRESS_REWARD = 0.5
        STAGNATION_PENALTY = -0.2
        GOAL_REWARD = 50.0
        OUT_PENALTY = -50.0
        reward = TIME_PENALTY
        if self.distances[-1] < self.distances[-2]:
            reward += PROGRESS_REWARD
        if self.distances[-1] >= self.distances[-2]:
            reward += STAGNATION_PENALTY
        if self.state == self.goal:
            reward = GOAL_REWARD
        if self.x > self.max_x or self.y > self.max_y or self.x < 0 or self.y < 0:
            reward = OUT_PENALTY
        return reward

    def take_action(self):
        if self.action == "left":
            self.x -= 1
        elif self.action == "right":
            self.x += 1
        else:
            self.x = self.x
        if self.action == "down":
            self.y -= 1
        elif self.action == "up":
            self.y += 1
        else:
            self.y = self.y

        return [self.x, self.y]

    def render(self):
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

            if not self.agent:
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

            pg.draw.rect(screen, AGENT_COLOR, (
            (agent_pos[0] * GRID_SIZE), (HEIGHT - (agent_pos[1] * GRID_SIZE) - GRID_SIZE), GRID_SIZE, GRID_SIZE))
            pg.draw.rect(screen, OBJECTIVE_COLOR, (
            objective_pos[0] * GRID_SIZE, (HEIGHT - (objective_pos[1] * GRID_SIZE) - GRID_SIZE), GRID_SIZE, GRID_SIZE))
            pg.display.flip()

            if agent_pos == objective_pos:
                self.reset()

            pg.time.Clock().tick(10)

        pg.quit()
        sys.exit()
