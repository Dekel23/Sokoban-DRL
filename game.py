import csv
import os
import random
import numpy as np


FIRST_LEVEL = 1
LAST_LEVEL = 62

class SokobanGame:
    def __init__(self, level, graphics_enable=False, seed=0):
        self.graphics_enable = graphics_enable
        self.level = level
       
        self.x, self.y = 0, 0
        # self.target_x, self.target_y = 0, 0
        # self.cargo_x, self.cargo_y = 0, 0

        self.map_info = None
        self.seed = seed
        self.load_map_info()
        if self.graphics_enable:
            from map.graphics import TileMap
            self.game_map = TileMap(self.map_info)

    def process_state(self):  # Take the game information and transform it into a states
        state = self.map_info[1:-1]  # Cut the frame
        state = [row[1:-1] for row in state]

        state = np.array(state, dtype=np.float32)  # transform to np.array in 1d
        
        return state

    # def set_map_info(self, state):
    #     curr_state = state.reshape(len(self.map_info)-2, len(self.map_info[0])-2)

    #     self.map_info = np.ones((len(self.map_info), len(self.map_info[0])))
    #     self.map_info[1:-1, 1:-1] = curr_state

    #     self.search_target_and_keeper_pos()

    def load_map_info(self):
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
        
        #self.add_keeper_randomly()
        self.search_target_and_keeper_pos()        
    
    def add_keeper_randomly(self):
        if self.seed:
            random.seed(self.seed)
        
        while True:
            rand_x = random.randint(1, len(self.map_info[0]) - 2)
            rand_y = random.randint(1, len(self.map_info) - 2)

            if self.map_info[rand_y][rand_x] in (2, 3): # Possible place for keeper
                self.map_info[rand_y][rand_x] = 6 # Put keeper
                break

    # Find the position of the keeper
    def search_target_and_keeper_pos(self):
        x, y = 0, 0
        for y, row in enumerate(self.map_info):
            for x, tile in enumerate(row):
                if tile in (6, 7):  # Keeper type of tiles
                    self.x = x
                    self.y = y
                # if tile in (3, 5, 7):  # Target type of tiles
                #     self.target_x = x
                #     self.target_y = y
                # if tile in (4, 5):  # Cargo type of tiles
                #     self.cargo_x = x
                #     self.cargo_y = y
    
    # Reset the game to the current level
    def reset_level(self):
        self.load_map_info()

        if self.graphics_enable:
            self.game_map.load_level(self.map_info, self.level)

    # Reset the game to the next level
    def next_level(self):
        self.level += 1
        self.reset_level()

    # Reset the game to the previous level
    def prev_level(self):
        self.level -= 1
        self.reset_level()
    
    # Step to take defined by action
    def step_action(self, action):
        if action == 0:  # UP
            self.move((-1, 0))
        if action == 1:  # RIGHT
            self.move((0, 1))
        if action == 2:  # DOWN
            self.move((1, 0))
        if action == 3:  # LEFT
            self.move((0, -1))

        if self.graphics_enable:
            self.game_map.update_ui(self.map_info)

        return self.check_end()  # done
    
    # Define what changes need to be done by the step
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
            # Change new pos to keeper
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
            # Change new pos to keeper & target
            info_to_change.append((self.y, self.x, 7))
        elif self.map_info[self.y + dist[0]][self.x + dist[1]] == 4:  # If go to cargo
            # If after cargo air continue
            if self.map_info[self.y + 2 * dist[0]][self.x + 2 * dist[1]] not in (1, 4, 5):
                if self.map_info[self.y][self.x] == 6:  # If he was not on target
                    # Change the pos to empty
                    info_to_change.append((self.y, self.x, 2))
                if self.map_info[self.y][self.x] == 7:  # If he was on target
                    # Change the pos to target
                    info_to_change.append((self.y, self.x, 3))

                # Set x,y to new values
                self.y, self.x = self.y + dist[0], self.x + dist[1]

                # Change new pos to keeper
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
            if self.map_info[self.y + 2 * dist[0]][self.x + 2 * dist[1]] not in (1, 4, 5):
                if self.map_info[self.y][self.x] == 6:  # If he was not on target
                    # Change the pos to empty
                    info_to_change.append((self.y, self.x, 2))
                if self.map_info[self.y][self.x] == 7:  # If he was on target
                    # Change the pos to target
                    info_to_change.append((self.y, self.x, 3))

                # Set x,y to new values
                self.y, self.x = self.y + dist[0], self.x + dist[1]

                # Change new pos to keeper & target
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
            raise Exception('Invalid move')

        self.change_map(info_to_change)

    # Change the game map info by the move (and graphics)
    def change_map(self, info_to_change):
        if info_to_change:
            for tile_info in info_to_change:
                (y, x, _type) = tile_info
                self.map_info[y][x] = _type

    # Check if the level ended
    def check_end(self):
        end_level = True
        for row in self.map_info:
            for tile in row:
                if tile in (3, 4):
                    end_level = False
        return end_level
