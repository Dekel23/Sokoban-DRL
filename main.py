#!/home/env/sokoban/bin/python3

import pygame
from map.tiles import TileMap
from map.spritesheet import Spritesheet

############################### LOAD UP BASIC WINDOW ##################################
pygame.init()
DISPLAY_HEIGHT, DISPLAY_WIDTH = 1000, 1000
canvas = pygame.Surface((DISPLAY_HEIGHT, DISPLAY_WIDTH))
window = pygame.display.set_mode(((DISPLAY_HEIGHT, DISPLAY_WIDTH)))
running = True

############################### LOAD PLAYER AND BLOCKS ################################
spritesheet = Spritesheet('map/images/spritesheet.png')
player_img = spritesheet.parse_sprite('keeper.png')
player_rect = player_img.get_rect()

############################### LOAD LEVEL ############################################
game_map = TileMap('levels/Level1.csv', spritesheet)
player_rect.x, player_rect.y = game_map.start_x, game_map.start_y

############################### GAME LOOP #############################################
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            pass


    ########################### UPDAE SPRITE #########################################

    ########################### UPDATE WINDOW AND DISPLAY ############################
    canvas.fill((0, 180, 240))
    game_map.draw_map(canvas)
    #canvas.blit(player_img, player_rect)
    window.blit(canvas, (0, 0))
    pygame.display.update()