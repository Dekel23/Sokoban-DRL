
class Keeper:
    def __init__(self, map_info):
        self.search_keeper_pos(map_info)
        self.map = map_info

    def search_keeper_pos(self, map_info):
        x, y = 0, 0
        for y, row in enumerate(map_info):
            for x, tile in enumerate(row):
                if tile == ('6' or '7'):
                    self.x = x
                    self.y = y

    def move(self, dist):  # 0 - no go, 1 - wall, 2 - empty, 3 - target, 4 - cargo, 5 - cargo & target, 6 - kepper, 7 - kepper & target
        info_to_change = []
        match self.map[self.y + dist[0]][self.x + dist[1]]:
            case '1':  # If go to wall do nothing
                return

            case '2':  # If go to empty tile
                if self.map[self.y][self.x] == '6':  # If he was not on target
                    # Change the pos to empty
                    info_to_change.append((self.y, self.x, '2'))
                if self.map[self.y][self.x] == '7':  # If he was on target
                    # Change the pos to target
                    info_to_change.append((self.y, self.x, '3'))

                # Set x,y to new values
                self.y, self.x = self.y + dist[0], self.x + dist[1]
                # Change new pos to kepper
                info_to_change.append((self.y, self.x, '6'))

            case '3':  # If go to target
                if self.map[self.y][self.x] == '6':  # If he was not on target
                    # Change the pos to empty
                    info_to_change.append((self.y, self.x, '2'))
                if self.map[self.y][self.x] == '7':  # If he was on target
                    # Change the pos to target
                    info_to_change.append((self.y, self.x, '3'))

                # Set x,y to new values
                self.y, self.x = self.y + dist[0], self.x + dist[1]
                # Change new pos to kepper & target
                info_to_change.append((self.y, self.x, '7'))

            case '4':  # If go to cargo
                # If after cargo air continue
                if self.map[self.y + 2*dist[0]][self.x + 2*dist[1]] not in ('1', '4', '5'):
                    if self.map[self.y][self.x] == '6':  # If he was not on target
                        # Change the pos to empty
                        info_to_change.append((self.y, self.x, '2'))
                    if self.map[self.y][self.x] == '7':  # If he was on target
                        # Change the pos to target
                        info_to_change.append((self.y, self.x, '3'))

                    # Set x,y to new values
                    self.y, self.x = self.y + dist[0], self.x + dist[1]
                    # Change new pos to kepper
                    info_to_change.append((self.y, self.x, '6'))

                    if self.map[self.y + dist[0]][self.x + dist[1]] == '2':  # If after cargo empty
                        # Set new pos to cargo
                        info_to_change.append(
                            (self.y + dist[0], self.x + dist[1], '4'))
                    if self.map[self.y + dist[0]][self.x + dist[1]] == '3':  # If after cargo target
                        # Set new pos cargo & target
                        info_to_change.append(
                            (self.y + dist[0], self.x + dist[1], '5'))

            case '5':  # If go to cargo & target
                # If after cargo air continue
                if self.map[self.y + 2*dist[0]][self.x + 2*dist[1]] not in ('1', '4', '5'):
                    if self.map[self.y][self.x] == '6':  # If he was not on target
                        # Change the pos to empty
                        info_to_change.append((self.y, self.x, '2'))
                    if self.map[self.y][self.x] == '7':  # If he was on target
                        # Change the pos to target
                        info_to_change.append((self.y, self.x, '3'))

                    # Set x,y to new values
                    self.y, self.x = self.y + dist[0], self.x + dist[1]
                    # Change new pos to kepper & target
                    info_to_change.append((self.y, self.x, '7'))

                    if self.map[self.y + dist[0]][self.x + dist[1]] == '2':  # If after cargo empty
                        # Set new pos to cargo
                        info_to_change.append(
                            (self.y + dist[0], self.x + dist[1], '4'))
                    if self.map[self.y + dist[0]][self.x + dist[1]] == '3':  # If after cargo target
                        # Set new pos cargo & target
                        info_to_change.append(
                            (self.y + dist[0], self.x + dist[1], '5'))

            case _:  # Otherwise something went wrong
                raise Exception('Invalid map')

        return info_to_change
