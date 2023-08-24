import pygame
import json

class Spritesheet:
    def __init__(self, filename):
        self.filename = filename # File to load from
        self.spritesheet = pygame.image.load(filename).convert()
        self.metadata = self.filename.replace('png', 'json')

        with open(self.metadata) as f:
            self.data = json.load(f) # Load the file to data

    # Return new surface from this spritesheet by coordinate 
    def get_sprite(self, x, y, w, h):
        sprite = pygame.Surface((w, h))
        sprite.set_colorkey((0, 0, 0))
        sprite.blit(self.spritesheet, (0, 0), (x, y, w, h))
        return sprite
    
    # Return new surface from this spritesheet by the image name 
    def parse_sprite(self, img_name):
        sprite = self.data['frames'][img_name]['frame']
        x, y, w, h = sprite['x'], sprite['y'], sprite['w'], sprite['h']
        image = self.get_sprite(x, y, w, h)
        return image