import pygame
import sys
import pymunk
from pymunk.pygame_util import

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Tag Game")

#Tagger
red = (255, 0, 0)

#Runner
blue = (0, 0, 255)

white = (255, 255, 255)
black = (0, 0, 0)
font = pygame.font.Font(None, 36)

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
space.add(tagger_body, tagger_shape)

runner_body = pymunk.Body(1, float('inf'))
runner_shape = pymunk.Circle(runner_body, player_radius)
runner_shape.color = blue
runner_body.position = 3 * width // 4, height // 2
runner_shape.elasticity = 1.0
space.add(runner_body, runner_shape)

tagger_points = 0
runner_points = 0

clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    tagger_body.velocity = (keys[pygame.K_d] - keys[pygame.K_a]) * tagger_speed_multiplier * 100, (keys[pygame.K_w] - keys[pygame.K_s]) * tagger_speed_multiplier * 100
    runner_body.velocity = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * runner_speed_multiplier * 100, (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * runner_speed_multiplier * 100

    space.step(1 / 60.0)

    screen.fill(white)
    pygame.draw.rect(screen, black, (0, 0, width, height), 2)

    for body, shape in [(tagger_body, tagger_shape), (runner_body, runner_shape)]:
        pos = int(body.position.x), int(height - body.position.y)
        pygame.draw.circle(screen, shape.color, pos, player_radius)
    score_text = font.render("Score: {}".format(tagger_points), True, (0, 0, 0))
    screen.blit(score_text, (10, 10))


    pygame.display.flip()

    clock.tick(60)