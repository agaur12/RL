import pygame as pg
import sys

# Initialize pg
pg.init()

WIDTH, HEIGHT = 600, 600
GRID_SIZE = 30
AGENT_COLOR = (255, 0, 0)
OBJECTIVE_COLOR = (0, 255, 0)
GRID_COLOR = (255, 255, 255)

screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Grid Game")

agent_pos = [0, 0]
objective_pos = [10, 10]

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    keys = pg.key.get_pressed()
    if keys[pg.K_LEFT] and agent_pos[0] > 0:
        agent_pos[0] -= 1
    if keys[pg.K_RIGHT] and agent_pos[0] < WIDTH // GRID_SIZE - 1:
        agent_pos[0] += 1
    if keys[pg.K_UP] and agent_pos[1] > 0:
        agent_pos[1] -= 1
    if keys[pg.K_DOWN] and agent_pos[1] < HEIGHT // GRID_SIZE - 1:
        agent_pos[1] += 1

    screen.fill(GRID_COLOR)
    for x in range(0, WIDTH, GRID_SIZE):
        pg.draw.line(screen, (0, 0, 0), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pg.draw.line(screen, (0, 0, 0), (0, y), (WIDTH, y))

    pg.draw.rect(screen, AGENT_COLOR, (agent_pos[0] * GRID_SIZE, agent_pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    pg.draw.rect(screen, OBJECTIVE_COLOR, (objective_pos[0] * GRID_SIZE, objective_pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    pg.display.flip()

    pg.time.Clock().tick(10)

pg.quit()
sys.exit()
