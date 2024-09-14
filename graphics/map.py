from tkinter import Tk
import pygame
from graphics.spritesheet import Spritesheet

class Tile(pygame.sprite.Sprite):
    def __init__(self, name):
        pygame.sprite.Sprite.__init__(self)
        self.spritesheet = Spritesheet('graphics/spritesheet.png') # Create spritesheet handler
        self.image = self.spritesheet.parse_sprite(name) # Get pygame surface
    
    def draw(self, x, y, surface):
        surface.blit(self.image, (x,y)) # Add image to new surface

class TileMap:
    def __init__(self, map_info):

        pygame.init()
        self.tile_size = 32

        # Make adjustable pygame window
        win = Tk()
        self.max_width, self.max_height = win.winfo_screenwidth() , win.winfo_screenheight() - 64
        self.max_surface = pygame.Surface((self.max_width, self.max_height))
        self.max_display = pygame.display.set_mode((self.max_width, self.max_height), pygame.RESIZABLE)
        self.resize_display(map_info)

        # Create all tiles images
        self.tile_types = ['empty.png', 'wall.png', 'floor.png', 'target.png', 'cargo.png', 'cargo_on_target.png', 'keeper.png', 'keeper_on_target.png']
        self.tiles = []
        for var in self.tile_types:
            self.tiles.append(Tile(var))
    
    # Create new surface that corresponding to map size
    def resize_display(self, map_info):
        self.map_height = len(map_info) * self.tile_size
        self.map_width = len(map_info[0]) * self.tile_size
        self.small_surface = pygame.Surface((self.map_width, self.map_height))
        self.small_display = pygame.display.set_mode((self.map_width, self.map_height), pygame.RESIZABLE)
    
    # Load new map and level to pygame data
    def load_level(self, map_info, level):
        if not (self.map_height == len(map_info) * self.tile_size and self.map_width == len(map_info[0]) * self.tile_size):
            self.resize_display(map_info)
        pygame.display.set_caption(f'Sokoban Level {level}')
        self.load_map(map_info)

    # Load new map to pygame data
    def load_map(self, map_info):
        self.small_surface.set_colorkey((0, 0, 0))
        for y, row in enumerate(map_info):
            for x, tile in enumerate(row):
                self.tiles[int(tile)].draw(x * self.tile_size, y * self.tile_size, self.small_surface)
    
    # Copy small surface to the max size surface
    def draw_map(self):
        self.small_surface.set_colorkey((0, 0, 0))
        self.max_surface.blit(self.small_surface, (0, 0))

    # Update the surface to the new map
    def update_ui(self, map_info):
        self.load_map(map_info)
        self.max_surface.fill((0, 0, 0))
        self.draw_map()
        self.max_display.blit(self.max_surface, (0, 0))
        pygame.display.update()