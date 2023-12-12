import pygame as pg
import sys
from base_env import GridEnvironment


pg.init()

env = GridEnvironment(4, 4, rand_goal=True, rand_start=True )

WIDTH, HEIGHT = 1000, 1000
GRID_SIZE = int(1000 / (env.max_x + 1))
AGENT_COLOR = (255, 0, 0)
OBJECTIVE_COLOR = (0, 255, 0)
GRID_COLOR = (255, 255, 255)

screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Grid Game")

agent_pos = env.state
objective_pos = env.goal


running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    keys = pg.key.get_pressed()
    if keys[pg.K_a]:
        env.step(3)
    if keys[pg.K_d]:
        env.step(1)
    if keys[pg.K_s]:
        env.step(2)
    if keys[pg.K_w]:
        env.step(0)
    agent_pos = env.state

    screen.fill(GRID_COLOR)
    for x in range(0, WIDTH, GRID_SIZE):
        pg.draw.line(screen, (0, 0, 0), (x, HEIGHT), (x, 0), 1)
    for y in range(0, HEIGHT, GRID_SIZE):
        pg.draw.line(screen, (0, 0, 0), (0, HEIGHT - y), (WIDTH, HEIGHT - y), 1)

    pg.draw.rect(screen, AGENT_COLOR, ((agent_pos[0] * GRID_SIZE), (HEIGHT - (agent_pos[1] * GRID_SIZE) - GRID_SIZE), GRID_SIZE, GRID_SIZE))
    pg.draw.rect(screen, OBJECTIVE_COLOR, (objective_pos[0] * GRID_SIZE, (HEIGHT - (objective_pos[1] * GRID_SIZE) - GRID_SIZE), GRID_SIZE, GRID_SIZE))
    pg.display.flip()

    if agent_pos == objective_pos:
        env.reset()

    pg.time.Clock().tick(10)

pg.quit()
sys.exit()
