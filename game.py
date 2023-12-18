#!/home/env/sokoban/bin/python3
# Control the flow of the game
import pygame
import csv
import os
from map.graphics import TileMap
from enum import Enum


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


FIRST_LEVEL = 1
LAST_LEVEL = 62

class SokobanGame:
    def __init__(self):
        self.level = 61
        self.reset_level()

    # Reset the game to the current level
    def reset_level(self):
        pygame.display.set_caption(f'Sokoban Level {self.level}')
        if self.level < FIRST_LEVEL or self.level > LAST_LEVEL:
            raise Exception('Invalid Level')

        absolute_path = os.path.dirname(__file__)
        relative_path_level = f'levels/Level{self.level}.csv'

        # Load the map info from file
        self.map_info = []
        with open(os.path.join(absolute_path, relative_path_level)) as f:
            data = csv.reader(f, delimiter=',')
            for row in data:
                row = [int(item) for item in list(row)]
                self.map_info.append(row)

        self.game_map = TileMap(self.map_info)
        self.search_keeper_pos()

    # Find the position of the keeper
    def search_keeper_pos(self):
        x, y = 0, 0
        for y, row in enumerate(self.map_info):
            for x, tile in enumerate(row):
                if tile in (6, 7): # Keeper type of tiles
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
        if action == 0:
            self.move((-1, 0))
        if action == 1:
            self.move((0, 1))
        if action == 2:
            self.move((1, 0))
        if action == 3:
            self.move((0, -1))

        self.game_map.update_ui()

        return self.check_end()  # done

    # Step to do difined by the keyboard
    # def play_step(self):
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             quit()
    #         if event.type == pygame.KEYDOWN:
    #             key_name = pygame.key.name(event.key)
    #             match key_name:
    #                 case 'd':
    #                     self.move((0,1))
    #                 case 'a':
    #                     self.move((0,-1))
    #                 case 'w':
    #                     self.move((-1,0))
    #                 case 's':
    #                     self.move((1,0))
    #                 case 'q':
    #                     pygame.quit()
    #                     quit()
    #                 case 'r':
    #                     self.reset_level()
    #                 case 'n':
    #                     self.next_level()
    #                 case 'p':
    #                     self.prev_level()

    #     if self.check_end():
    #         self.reset_level()
    #     self.game_map.update_ui()

    # Difine what changes need to be done by the step
    def move(self, dist):
        info_to_change = []
        if self.map_info[self.y + dist[0]][self.x + dist[1]] == 1:  # If go to wall do nothing
            return
        elif self.map_info[self.y + dist[0]][self.x + dist[1]] == 2:  # If go to empty tile
            if self.map_info[self.y][self.x] == 6:  # If he was not on target
                # Change the pos to empty
                info_to_change.append((self.y, self.x, 2))
            if self.map_info[self.y][self.x] == 7:  # If he was on target
                # Change the pos to target
                info_to_change.append((self.y, self.x, 3))

            # Set x,y to new values
            self.y, self.x = self.y + dist[0], self.x + dist[1]
            # Change new pos to kepper
            info_to_change.append((self.y, self.x, 6))
        elif self.map_info[self.y + dist[0]][self.x + dist[1]] == 3:  # If go to target
            if self.map_info[self.y][self.x] == 6:  # If he was not on target
                # Change the pos to empty
                info_to_change.append((self.y, self.x, 2))
            if self.map_info[self.y][self.x] == 7:  # If he was on target
                # Change the pos to target
                info_to_change.append((self.y, self.x, 3))

            # Set x,y to new values
            self.y, self.x = self.y + dist[0], self.x + dist[1]
            # Change new pos to kepper & target
            info_to_change.append((self.y, self.x, 7))
        elif self.map_info[self.y + dist[0]][self.x + dist[1]] == 4:  # If go to cargo
            # If after cargo air continue
            if self.map_info[self.y + 2*dist[0]][self.x + 2*dist[1]] not in (1, 4, 5):
                if self.map_info[self.y][self.x] == 6:  # If he was not on target
                    # Change the pos to empty
                    info_to_change.append((self.y, self.x, 2))
                if self.map_info[self.y][self.x] == 7:  # If he was on target
                    # Change the pos to target
                    info_to_change.append((self.y, self.x, 3))

                # Set x,y to new values
                self.y, self.x = self.y + dist[0], self.x + dist[1]
                # Change new pos to kepper
                info_to_change.append((self.y, self.x, 6))

                # If after cargo empty
                if self.map_info[self.y + dist[0]][self.x + dist[1]] == 2:
                    # Set new pos to cargo
                    info_to_change.append(
                        (self.y + dist[0], self.x + dist[1], 4))
                # If after cargo target
                if self.map_info[self.y + dist[0]][self.x + dist[1]] == 3:
                    # Set new pos cargo & target
                    info_to_change.append(
                        (self.y + dist[0], self.x + dist[1], 5))
        elif self.map_info[self.y + dist[0]][self.x + dist[1]] == 5:  # If go to cargo & target
            # If after cargo air continue
            if self.map_info[self.y + 2*dist[0]][self.x + 2*dist[1]] not in (1, 4, 5):
                if self.map_info[self.y][self.x] == 6:  # If he was not on target
                    # Change the pos to empty
                    info_to_change.append((self.y, self.x, 2))
                if self.map_info[self.y][self.x] == 7:  # If he was on target
                    # Change the pos to target
                    info_to_change.append((self.y, self.x, 3))

                # Set x,y to new values
                self.y, self.x = self.y + dist[0], self.x + dist[1]
                # Change new pos to kepper & target
                info_to_change.append((self.y, self.x, 7))

                # If after cargo empty
                if self.map_info[self.y + dist[0]][self.x + dist[1]] == 2:
                    # Set new pos to cargo
                    info_to_change.append(
                        (self.y + dist[0], self.x + dist[1], 4))
                # If after cargo target
                if self.map_info[self.y + dist[0]][self.x + dist[1]] == 3:
                    # Set new pos cargo & target
                    info_to_change.append(
                        (self.y + dist[0], self.x + dist[1], 5))
        else:  # Otherwise something went wrong
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
        for row in self.map_info:
            for tile in row:
                if tile in (3, 4):
                    end_level = False
        return end_level