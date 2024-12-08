import pygame
from genetic_algorithm_functions import *

pygame.init()
screen = pygame.display.set_mode((1080,720))

# tester = Tester(screen, "./save/Generation_7_best_model.txt")
# tester.run()

trainer = Trainer(screen, "./save", 100000)
trainer.run()       