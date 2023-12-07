#!/home/env/sokoban/bin/python3
# Control the flow of the game
import pygame
import csv
import os
from map.graphics import TileMap
from enum import Enum

pygame.init()

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SokobanGame:
    def __init__(self):
        self.level = 62
        self.reset_level()

    # Reset the game to the current level
    def reset_level(self):
        pygame.display.set_caption(f'Sokoban Level {self.level}')
        if self.level < 1 or self.level > 62:
            raise Exception ('Invalid Level')
        level = f'levels/Level{self.level}.csv'
        # Load the map info from file
        self.map_info = []
        with open(os.path.join(level)) as f:
            data = csv.reader(f, delimiter=',')
            for row in data:
                self.map_info.append(list(row))
        
        self.game_map = TileMap(self.map_info)
        self.search_kepper_pos()
    
    # Find the position of the kepper
    def search_kepper_pos(self):
        x, y = 0, 0
        for y, row in enumerate(self.map_info):
            for x, tile in enumerate(row):
                if tile in ('6','7'):
                    self.x = x
                    self.y = y
    
    # Reset the game to the next level
    def next_level(self):
        self.level += 1
        self.reset_level()
    
    # Reset the game to the previos level
    def prev_level(self):
        self.level -= 1
        self.reset_level()

    # Step to do difined by action
    def step_action(self, action):
        if action == Action.UP:
            self.move((-1,0))
        if action == Action.RIGHT:
            self.move((0,1))
        if action == Action.DOWN:
            self.move((1,0))
        if action == Action.LEFT:
            self.move((0,-1))
        
        if self.check_end():
            self.reset_level()
        self.game_map.update_ui()

    # Step to do difined by the keyboard
    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                match key_name:
                    case 'd':
                        self.move((0,1))
                    case 'a':
                        self.move((0,-1))
                    case 'w':
                        self.move((-1,0))
                    case 's':
                        self.move((1,0))
                    case 'q':
                        pygame.quit()
                        quit()
                    case 'r':
                        self.reset_level()
                    case 'n':
                        self.next_level()
                    case 'p':
                        self.prev_level()
    
        if self.check_end():
            self.reset_level()
        self.game_map.update_ui()

    # Difine what changes need to be done by the step
    def move(self, dist):
        info_to_change = []
        match self.map_info[self.y + dist[0]][self.x + dist[1]]:
            case '1': # If go to wall do nothing
                return
            
            case '2': # If go to empty tile
                if self.map_info[self.y][self.x] == '6': # If he was not on target
                    info_to_change.append((self.y,self.x,'2')) # Change the pos to empty
                if self.map_info[self.y][self.x] == '7': # If he was on target
                    info_to_change.append((self.y,self.x,'3')) # Change the pos to target

                self.y, self.x = self.y + dist[0], self.x + dist[1] # Set x,y to new values
                info_to_change.append((self.y,self.x,'6')) # Change new pos to kepper

            case '3': # If go to target
                if self.map_info[self.y][self.x] == '6': # If he was not on target
                    info_to_change.append((self.y,self.x,'2')) # Change the pos to empty
                if self.map_info[self.y][self.x] == '7': # If he was on target
                    info_to_change.append((self.y,self.x,'3')) # Change the pos to target

                self.y, self.x = self.y + dist[0], self.x + dist[1] # Set x,y to new values
                info_to_change.append((self.y,self.x,'7')) # Change new pos to kepper & target

            case '4': # If go to cargo
                if self.map_info[self.y + 2*dist[0]][self.x + 2*dist[1]] not in ('1', '4', '5'): # If after cargo air continue
                    if self.map_info[self.y][self.x] == '6': #If he was not on target
                        info_to_change.append((self.y,self.x,'2')) # Change the pos to empty
                    if self.map_info[self.y][self.x] == '7': # If he was on target
                        info_to_change.append((self.y,self.x,'3')) # Change the pos to target 
                    
                    self.y, self.x = self.y + dist[0], self.x + dist[1] # Set x,y to new values
                    info_to_change.append((self.y, self.x, '6')) # Change new pos to kepper
                    
                    if self.map_info[self.y + dist[0]][self.x + dist[1]] == '2': # If after cargo empty
                        info_to_change.append((self.y + dist[0],self.x + dist[1],'4')) # Set new pos to cargo
                    if self.map_info[self.y + dist[0]][self.x + dist[1]] == '3': #If after cargo target
                        info_to_change.append((self.y + dist[0],self.x + dist[1],'5')) # Set new pos cargo & target

            case '5': # If go to cargo & target
                if self.map_info[self.y + 2*dist[0]][self.x + 2*dist[1]] not in ('1', '4', '5'): # If after cargo air continue
                    if self.map_info[self.y][self.x] == '6': #If he was not on target
                        info_to_change.append((self.y,self.x,'2')) # Change the pos to empty
                    if self.map_info[self.y][self.x] == '7': # If he was on target
                        info_to_change.append((self.y,self.x,'3')) # Change the pos to target  
                    
                    self.y, self.x = self.y + dist[0], self.x + dist[1] # Set x,y to new values
                    info_to_change.append((self.y, self.x, '7')) # Change new pos to kepper & target
                    
                    if self.map_info[self.y + dist[0]][self.x + dist[1]] == '2': # If after cargo empty
                        info_to_change.append((self.y + dist[0],self.x + dist[1],'4')) # Set new pos to cargo
                    if self.map_info[self.y + dist[0]][self.x + dist[1]] == '3': #If after cargo target
                        info_to_change.append((self.y + dist[0],self.x + dist[1],'5')) # Set new pos cargo & target

            case _: # Otherwise something went wrong
                raise Exception('Invalid map')
        self.change_map(info_to_change)
    
    # Change the game map info by the move (and graphics)
    def change_map(self, info_to_change):
        if info_to_change:
            for tile_info in info_to_change:
                (y, x, type) = tile_info
                self.map_info[y][x] = type
                self.game_map.change_tile(tile_info)

    # Check if the level ended
    def check_end(self):
        end_level = True
        for y, row in enumerate(self.map_info):
            for x, tile in enumerate(row):
                if tile in ('3', '4'):
                    end_level = False
        return end_level

if __name__ == '__main__':
    game = SokobanGame()
    while True:
        game.play_step()