#!/home/env/sokoban/bin/python3

import pygame
from map.tiles import TileMap
from map.spritesheet import Spritesheet
from tkinter import *

############################### LOAD UP BASIC WINDOW ##################################
pygame.init()
win = Tk()
DISPLAY_HEIGHT, DISPLAY_WIDTH = win.winfo_screenheight() - 64 , win.winfo_screenwidth()
canvas = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT))
window = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.RESIZABLE)

pygame.display.set_caption("Sokoban")

running = True

############################### LOAD TILES INFO ################################
spritesheet = Spritesheet('map/images/spritesheet.png')

############################### LOAD LEVEL SET SIZE OF SCREEN ############################################
game_map = TileMap('levels/Level20.csv', spritesheet)
game_map.load_map()
window = pygame.display.set_mode((game_map.map_width, game_map.map_height), pygame.RESIZABLE)

############################### GAME LOOP #############################################
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            pass

    ########################### UPDAE SPRITE #########################################

    ########################### UPDATE WINDOW AND DISPLAY ############################
    canvas.fill((0, 0, 0))
    game_map.draw_map(canvas)
    window.blit(canvas, (0, 0))
    pygame.display.update()