import pygame
import random

# Inizializza pygame
pygame.init()

# Costanti
WIDTH, HEIGHT = 800, 800
LINE_Y = HEIGHT - 50
NUM_SECTIONS = 25
SECTION_WIDTH = WIDTH // NUM_SECTIONS
SQUARE_SIZE = 10
FPS = 60

WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulazione con 25 Sezioni/nozzles")

clock = pygame.time.Clock()

squares = []

sections = [False] * NUM_SECTIONS


def spawn_square():
    width = random.randint(30, 180)  # Random width
    height = random.randint(30, 150)  # Random height
    x = random.randint(0, WIDTH - width)
    y = -height
    squares.append(pygame.Rect(x, y, width, height))


running = True
while running:
    clock.tick(FPS)
    screen.fill(WHITE)

    # Eventi
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if random.random() < 0.04:
        spawn_square()

    for square in squares:
        square.y += 4

    squares = [square for square in squares if square.top < HEIGHT]

    sections = [False] * NUM_SECTIONS

    for square in squares:
        pygame.draw.rect(screen, BLACK, square, width=2)
        if square.bottom >= LINE_Y:
            index = min(square.centerx // SECTION_WIDTH, NUM_SECTIONS - 1)
            sections[index] = True

    for i in range(NUM_SECTIONS):
        color = RED if sections[i] else GRAY
        rect = pygame.Rect(i * SECTION_WIDTH, LINE_Y, SECTION_WIDTH, 5)
        pygame.draw.rect(screen, color, rect)

    pygame.display.flip()

pygame.quit()
