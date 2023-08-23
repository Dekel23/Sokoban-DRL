import pygame
import csv
import os

class Tile(pygame.sprite.Sprite):
    def __init__(self, image, x, y, spritesheet):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = spritesheet.parse_sprite(image)

        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y

    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))

class TileMap():
    def __init__(self, filename, spritesheet):
        self.tile_size = 32
        self.spritesheet = spritesheet
        self.tiles = self.load_tiles(filename)
        self.map_surface = pygame.Surface((self.map_width, self.map_height))
        self.map_surface.set_colorkey((0, 0, 0))
        self.load_map()

    def draw_map(self, surface):
        surface.blit(self.map_surface, (0, 0))

    def load_map(self):
        for tile in self.tiles:
            tile.draw(self.map_surface)

    def read_csv(self, filename):
        game_map = []
        with open(os.path.join(filename)) as f:
            data = csv.reader(f, delimiter=',')
            for row in data:
                game_map.append(list(row))

        return game_map
    
    def load_tiles(self, filename):
        tiles = []
        game_map = self.read_csv(filename)

        x, y = 0, 0
        for y, row in enumerate(game_map):
            for x, tile in enumerate(row):
                match tile:
                    case '0':
                        tiles.append(Tile('empty.png', x * self.tile_size, y * self.tile_size, self.spritesheet))
                    case '1':
                        tiles.append(Tile('wall.png', x * self.tile_size, y * self.tile_size, self.spritesheet))
                    case '2':
                        tiles.append(Tile('floor.png', x * self.tile_size, y * self.tile_size, self.spritesheet))
                    case '3':
                        tiles.append(Tile('target.png', x * self.tile_size, y * self.tile_size, self.spritesheet))
                    case '4':
                        tiles.append(Tile('cargo.png', x * self.tile_size, y * self.tile_size, self.spritesheet))
                    case '5':
                        tiles.append(Tile('cargo_on_target.png', x * self.tile_size, y * self.tile_size, self.spritesheet))
                    case '6':
                        tiles.append(Tile('keeper.png', x * self.tile_size, y * self.tile_size, self.spritesheet))
                    case '7':
                        tiles.append(Tile('keeper_on_target.png', x * self.tile_size, y * self.tile_size, self.spritesheet))
                    case _:
                        raise Exception('Invalid tile value (invalid map)')
        
        self.map_width, self.map_height = (x + 1) * self.tile_size, (y + 1) * self.tile_size
        return tiles