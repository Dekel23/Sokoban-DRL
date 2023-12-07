import pygame
from map.spritesheet import Spritesheet
from tkinter import *

pygame.init()
win = Tk()
MAX_WIDTH, MAX_HEIGHT = win.winfo_screenwidth() , win.winfo_screenheight() - 64
SURFACE = pygame.Surface((MAX_WIDTH, MAX_HEIGHT))
DISPLAY = pygame.display.set_mode((MAX_WIDTH, MAX_HEIGHT), pygame.RESIZABLE)
TILE_SIZE = 32
SPRITESHEET = Spritesheet('map/images/spritesheet.png')
TILE_TYPES = ['empty.png', 'wall.png', 'floor.png', 'target.png', 'cargo.png', 'cargo_on_target.png',
               'keeper.png', 'keeper_on_target.png']

class Tile(pygame.sprite.Sprite):
    def __init__(self, name, x, y):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = SPRITESHEET.parse_sprite(name) # Tile image
        self.rect = self.image.get_rect() # Tile image proportions
        self.rect.x, self.rect.y = x, y # Position on screen

    # Draw the tile on the surface
    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))

class TileMap():
    def __init__(self, map_info):
        self.load_tiles(map_info)
        self.surface = pygame.Surface((self.map_width, self.map_height)) # New Map Surface
        self.display = pygame.display.set_mode((self.map_width, self.map_height), pygame.RESIZABLE)
        
    # Draw the surface on screen
    def draw_map(self, surface):
        self.surface.set_colorkey((0, 0, 0))
        surface.blit(self.surface, (0, 0))

    # Draw the map to the surface
    def load_map(self):
        self.surface.set_colorkey((0, 0, 0))
        for tile in self.tiles:
            tile.draw(self.surface)
    
    # Load the current map file to tiles list and set the map proportions
    def load_tiles(self, map_info):
        tiles = []

        x, y = 0, 0
        for y, row in enumerate(map_info):
            for x, tile in enumerate(row):
                try:
                    new_tile = Tile(TILE_TYPES[int(tile)], x * TILE_SIZE, y * TILE_SIZE)
                    tiles.append(new_tile)
                except:
                    raise Exception('Invalid tile value (invalid map)')
        
        self.map_width, self.map_height = (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE
        self.tiles = tiles
    
    def change_tile(self, tile_info):
        (y, x, type) = tile_info
        tile_index = int(y * self.map_width/TILE_SIZE + x)
        new_tile = Tile(TILE_TYPES[int(type)], x * TILE_SIZE, y * TILE_SIZE)
        self.tiles[tile_index] = new_tile
    
    def update_ui(self):
        self.load_map()
        SURFACE.fill((0, 0, 0))
        self.draw_map(SURFACE)
        DISPLAY.blit(SURFACE, (0, 0))
        pygame.display.update()