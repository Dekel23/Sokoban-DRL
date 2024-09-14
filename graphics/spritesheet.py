import pygame
import json

class Spritesheet:

    # Make sure only one onject is created
    _self = None
    def __new__(cls, *args, **kwargs):
        if cls._self is None:
            cls._self = super(Spritesheet, cls).__new__(cls)
        return cls._self
    
    def __init__(self, filename):
        self.spritesheet = pygame.image.load(filename).convert() # Load file to pygame image

        self.metadata = filename.replace('png', 'json') # Load the json file to data 
        with open(self.metadata) as f:
            self.data = json.load(f)
    
    def get_sprite(self, x, y, w, h):
        sprite = pygame.Surface((w, h)) # Create pygame surface
        sprite.set_colorkey((0, 0, 0))
        sprite.blit(self.spritesheet, (0, 0), (x, y, w, h)) # Append image square to surface
        return sprite

    def parse_sprite(self, img_name):
        sprite = self.data['frames'][img_name]['frame'] # Get image coordinates by name
        x, y, w, h = sprite['x'], sprite['y'], sprite['w'], sprite['h']
        image = self.get_sprite(x, y, w, h) # Create pygame surface with the image
        return image