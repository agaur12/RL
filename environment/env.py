import numpy as np
import random
import math

class Grid_Environment:
    def __init__(self, size_x: int, size_y: int, rand_goal: bool=False, rand_start: bool=False):
        self.actions = ["up", "right", "down", "left"]
        self.max_x = size_x - 1
        self.max_y = size_y - 1 
        self.done = False
        self.episode_length = 0
        self.reward = 0
        if (rand_start):
            self.x, self.y = random.randint(1,self.max_x), random.randint(1,self.max_x)
        else:
            self.x, self.y = 0, 0
        if (rand_goal):
            self.goal = [random.randint(1,self.max_x), random.randint(1,self.max_x)]
        else:
            self.goal = [self.max_x, self.max_y]
        self.state = [self.x, self.y]

    def reset(self):
        self.x, self.y = 0, 0
        self.done = False
        self.episode_length = 0
        self.reward = 0
        self.state = [self.x, self.y]
        return [self.x, self.y]

    def action_space(self):
        return self.actions

    def step(self, action):
        if (self.state == self.goal or self.episode_length > 200):
            self.done = True
            return np.array(self.state), self.reward, self.done, self.episode_length

        self.action = action
        self.reward += self.get_reward()
        self.state = self.take_action()
        self.episode_length += 1

        if (self.state == self.goal or self.episode_length > 200):
            self.done = True
            return np.array(self.state), self.reward, self.done, self.episode_length

    def get_reward(self):
        if(self.episode_length > 200):
            reward = -10
        elif(self.state == self.goal):
            reward = 10

    def take_action(self):
        if (self.action == "left" and self.x == 0) or (self.action == "right" and self.x == self.max_x):
            self.x = self.x
        elif (self.action == "left"):
            self.x -= 1
        elif (self.action == "right"):
            self.x += 1
        else:
            self.x = self.x
        if (self.action == "down" and self.y == 0) or (self.action == "up" and self.y == self.max_x):
            self.y = self.y
        elif (self.action == "down"):
            self.y -= 1
        elif (self.action == "up"):
            self.y += 1
        else:
            self.y = self.y

        return [self.x, self.y]



        
        
        
            
