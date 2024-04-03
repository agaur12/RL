import gym
from gym import spaces
import numpy as np
import pygame as pg
import pymunk
import sys

class TagEnv(gym.Env):
    def __init__(self):
        super(TagEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.rand_start = False
        self.rand_goal = False
        self.max_x = 1200
        self.max_y = 800

    def reset(self):
        None

    def step(self, action):
        None

    def render(self):
        pg.init()

        width, height = 1200, 800
        screen = pg.display.set_mode((width, height))
        pg.display.set_caption("Tag Game")

        # Tagger
        red = (255, 0, 0)

        # Runner
        blue = (0, 0, 255)

        white = (255, 255, 255)
        black = (0, 0, 0)
        font = pg.font.Font(None, 36)

        player_radius = 20

        space = pymunk.Space()
        space.gravity = (0, 0)
        tagger_speed_multiplier = 2
        runner_speed_multiplier = 2

        tagger_body = pymunk.Body(1, float('inf'))
        tagger_shape = pymunk.Circle(tagger_body, player_radius)
        tagger_shape.color = red
        tagger_body.position = width // 4, height // 2
        tagger_shape.elasticity = 1.0
        tagger_shape.tag = "tagger"
        space.add(tagger_body, tagger_shape)

        runner_body = pymunk.Body(1, float('inf'))
        runner_shape = pymunk.Circle(runner_body, player_radius)
        runner_shape.color = blue
        runner_body.position = 3 * width // 4, height // 2
        runner_shape.elasticity = 1.0
        runner_shape.tag = "runner"
        space.add(runner_body, runner_shape)

        tagger_points = 0
        runner_points = 0

        clock = pg.time.Clock()

        runner_timer = 0

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

            keys = pg.key.get_pressed()
            tagger_body.velocity = (keys[pg.K_d] - keys[pg.K_a]) * tagger_speed_multiplier * 100, (
                    keys[pg.K_w] - keys[pg.K_s]) * tagger_speed_multiplier * 100
            runner_body.velocity = (keys[pg.K_RIGHT] - keys[pg.K_LEFT]) * runner_speed_multiplier * 100, (
                    keys[pg.K_UP] - keys[pg.K_DOWN]) * runner_speed_multiplier * 100

            for shape in space.shapes:
                if shape.tag == "tagger":
                    for collision in space.shape_query(shape):
                        if collision.shape.tag == "runner":
                            tagger_points += 1
                            tagger_body.position = width // 4, height // 2
                            runner_body.position = 3 * width // 4, height // 2
                            runner_timer = 0

            runner_timer += 1 / 60.0
            if runner_timer >= 10:
                runner_points += 1
                runner_timer = 0

            space.step(1 / 60.0)

            screen.fill(white)
            pg.draw.rect(screen, black, (0, 0, width, height), 2)

            for body, shape in [(tagger_body, tagger_shape), (runner_body, runner_shape)]:
                pos = int(body.position.x), int(height - body.position.y)
                pg.draw.circle(screen, shape.color, pos, player_radius)
            tagger_score_text = font.render("Tagger: {}".format(tagger_points), True, (0, 0, 0))
            runner_score_text = font.render("Runner: {}".format(runner_points), True, (0, 0, 0))
            screen.blit(tagger_score_text, (10, 10))
            screen.blit(runner_score_text, (width - 150, 10))

            pg.display.flip()

            clock.tick(60)

