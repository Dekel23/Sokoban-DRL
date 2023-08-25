import pygame


class Tile(pygame.sprite.Sprite):
    def __init__(self, image, x, y, spritesheet):
        pygame.sprite.Sprite.__init__(self)

        self.image = spritesheet.parse_sprite(image)  # Tile image
        self.rect = self.image.get_rect()  # Tile image proportions
        self.rect.x, self.rect.y = x, y  # Position on screen

    # Draw the tile on the surface
    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))


class TileMap():
    def __init__(self, map_info, spritesheet):
        self.tile_size = 32  # Size of any tile image
        self.spritesheet = spritesheet  # Spritesheet to take image from
        self.tileTypes = ['empty.png', 'wall.png', 'floor.png', 'target.png', 'cargo.png', 'cargo_on_target.png',
                          'keeper.png', 'keeper_on_target.png']  # Types of tiles by order in map info
        self.tiles = self.load_tiles(map_info)
        self.map_surface = pygame.Surface(
            (self.map_width, self.map_height))  # New Map Surface

    # Draw the surface on screen
    def draw_map(self, surface):
        self.map_surface.set_colorkey((0, 0, 0))
        surface.blit(self.map_surface, (0, 0))

    # Draw the map to the surface
    def load_map(self):
        self.map_surface.set_colorkey((0, 0, 0))
        for tile in self.tiles:
            tile.draw(self.map_surface)

    # Load the current map file to tiles list and set the map proportions
    def load_tiles(self, map_info):
        tiles = []

        x, y = 0, 0
        for y, row in enumerate(map_info):
            for x, tile in enumerate(row):
                try:
                    tiles.append(Tile(self.tileTypes[int(
                        tile)], x * self.tile_size, y * self.tile_size, self.spritesheet))
                except:
                    raise Exception('Invalid tile value (invalid map)')

        self.map_width, self.map_height = (
            x + 1) * self.tile_size, (y + 1) * self.tile_size
        return tiles
