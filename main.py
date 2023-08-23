#!/home/env/sokoban/bin/python3

import pygame
from map.tiles import TileMap
from map.spritesheet import Spritesheet

############################### LOAD UP BASIC WINDOW ##################################
pygame.init()
DISPLAY_HEIGHT, DISPLAY_WIDTH = 2000, 2000
canvas = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT))
window = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.RESIZABLE)

pygame.display.set_caption("Sokoban")

running = True

############################### LOAD PLAYER AND BLOCKS ################################
spritesheet = Spritesheet('map/images/spritesheet.png')
#player_img = spritesheet.parse_sprite('keeper.png')
# = player_img.get_rect()

############################### LOAD LEVEL ############################################
game_map = TileMap('levels/Level1.csv', spritesheet)
window = pygame.display.set_mode((game_map.map_width, game_map.map_height), pygame.RESIZABLE)

#player_rect.x, player_rect.y = game_map.start_x, game_map.start_y

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
    #canvas.blit(player_img, player_rect)
    window.blit(canvas, (0, 0))
    pygame.display.update()