import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces
import math
import pygame as pg
import sys
#import Box2d


class TagEnvironment(gym.Env):
    metadata = {
        'render.modes': ['human']
    }
    render_mode = 'human'

    def __init__(self, size_x, size_y, agents, rand_start=False, agent=True):
        self.max_x = size_x - 1
        self.max_y = size_y - 1

        self.rand_start = rand_start

        self.agent = agent
        self.info = {}

        if rand_start:
            self.x, self.y = random.randint(1, self.max_x), random.randint(1, self.max_x)
        else:
            self.x, self.y = 0.0, 0.0

        self.tagger_pos = [self.x, self.y]

        if rand_start:
            self.x, self.y = random.randint(1, self.max_x), random.randint(1, self.max_x)
            while math.dist([self.x, self.y], self.tagger_pos) < 2:
                self.x, self.y = random.randint(1, self.max_x), random.randint(1, self.max_x)
        else:
            self.x, self.y = 0.0, 0.0
        self.runner_pos = []


