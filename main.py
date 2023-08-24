#!/home/env/sokoban/bin/python3
# Control the flow of the game
import pygame
from keeper import Keeper
import csv
import os
from map.tiles import TileMap, Tile
from map.spritesheet import Spritesheet
from tkinter import *

############################### LOAD UP BASIC WINDOW ##################################
pygame.init()
win = Tk()
DISPLAY_WIDTH, DISPLAY_HEIGHT = win.winfo_screenwidth() , win.winfo_screenheight() - 64
canvas = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT))
window = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.RESIZABLE)

running = True

############################### LOAD TILES INFO ################################
spritesheet = Spritesheet('map/images/spritesheet.png')

############################### LOAD LEVEL SET SIZE OF SCREEN ############################################
level_index = 1

def load_level(level_index):
    pygame.display.set_caption(f'Sokoban Level {level_index}')
    if level_index < 1 or level_index > 60:
        raise Exception ('Invalid Level')
    level = f'levels/Level{level_index}.csv'
    # Load the map info from file
    map_info = []
    with open(os.path.join(level)) as f:
        data = csv.reader(f, delimiter=',')
        for row in data:
            map_info.append(list(row))
    
    game_map = TileMap(map_info, spritesheet)
    pygame.display.set_mode((game_map.map_width, game_map.map_height), pygame.RESIZABLE)

    player = Keeper(map_info)
    return map_info, game_map, player

map_info, game_map, player = load_level(level_index)

############################### GAME LOOP #############################################
pygame.key.set_repeat(150)
while running:
    info_to_change = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
            match key_name:
                case 'd':
                    info_to_change = player.Move((0,1))
                case 'a':
                    info_to_change = player.Move((0,-1))
                case 'w':
                    info_to_change = player.Move((-1,0))
                case 's':
                    info_to_change = player.Move((1,0))
                case 'q':
                    running = False
                case 'r':
                    map_info, game_map, player = load_level(level_index)
                case 'n':
                    level_index += 1
                    map_info, game_map, player = load_level(level_index)
                case 'p':
                    level_index -= 1
                    map_info, game_map, player = load_level(level_index)

    ########################### UPDAE SPRITE #########################################
    if info_to_change:
        for tile_info in info_to_change:
            (y, x, type) = tile_info
            map_info[y][x] = type
            tile_index = int(y * game_map.map_width/game_map.tile_size + x)
            new_tile = Tile(game_map.tileTypes[int(type)], x * game_map.tile_size, y * game_map.tile_size, game_map.spritesheet)
            game_map.tiles[tile_index] = new_tile

    ########################### CHECK ENDING LEVEL AND UPLOAD NEW #########################################
    end_level = True
    for y, row in enumerate(map_info):
            for x, tile in enumerate(row):
                if tile in ('3', '4'):
                    end_level = False
    if end_level:
        level_index += 1
        map_info, game_map, player = load_level(level_index)

    ########################### UPDATE WINDOW AND DISPLAY ############################
    game_map.load_map()
    canvas.fill((0, 0, 0))
    game_map.draw_map(canvas)
    window.blit(canvas, (0, 0))
    pygame.display.update()
