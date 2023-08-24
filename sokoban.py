
class Kepper:
    def __init__(self,map_info):
        self.searchKeeperPos(map_info)
        self.map = map_info
        
    def searchKeeperPos(self, map_info):
        x, y = 0, 0
        for y, row in enumerate(map_info):
            for x, tile in enumerate(row):
                if tile == ('6' or '7'):
                    self.x = x
                    self.y = y

    def Move(self, dist): # 0 - no go, 1 - wall, 2 - empty, 3 - target, 4 - cargo, 5 - cargo & target, 6 - kepper, 7 - kepper & target
        info_to_change = []
        match self.map[self.y + dist[0]][self.x + dist[1]]:
            case '1': # If go to wall do nothing
                return
            
            case '2': # If go to empty tile
                if self.map[self.y][self.x] == '6': # If he was not on target
                    info_to_change.append((self.y,self.x,'2')) # Change the pos to empty
                if self.map[self.y][self.x] == '7': # If he was on target
                    info_to_change.append((self.y,self.x,'3')) # Change the pos to target

                self.y, self.x = self.y + dist[0], self.x + dist[1] # Set x,y to new values
                info_to_change.append((self.y,self.x,'6')) # Change new pos to kepper

            case '3': # If go to target
                if self.map[self.y][self.x] == '6': # If he was not on target
                    info_to_change.append((self.y,self.x,'2')) # Change the pos to empty
                if self.map[self.y][self.x] == '7': # If he was on target
                    info_to_change.append((self.y,self.x,'3')) # Change the pos to target

                self.y, self.x = self.y + dist[0], self.x + dist[1] # Set x,y to new values
                info_to_change.append((self.y,self.x,'7')) # Change new pos to kepper & target

            case '4': # If go to cargo
                if self.map[self.y + 2*dist[0]][self.x + 2*dist[1]] not in ('1', '4', '5'): # If after cargo air continue
                    if self.map[self.y][self.x] == '6': #If he was not on target
                        info_to_change.append((self.y,self.x,'2')) # Change the pos to empty
                    if self.map[self.y][self.x] == '7': # If he was on target
                        info_to_change.append((self.y,self.x,'3')) # Change the pos to target 
                    
                    self.y, self.x = self.y + dist[0], self.x + dist[1] # Set x,y to new values
                    info_to_change.append((self.y, self.x, '6')) # Change new pos to kepper
                    
                    if self.map[self.y + dist[0]][self.x + dist[1]] == '2': # If after cargo empty
                        info_to_change.append((self.y + dist[0],self.x + dist[1],'4')) # Set new pos to cargo
                    if self.map[self.y + dist[0]][self.x + dist[1]] == '3': #If after cargo target
                        info_to_change.append((self.y + dist[0],self.x + dist[1],'5')) # Set new pos cargo & target

            case '5': # If go to cargo & target
                if self.map[self.y + 2*dist[0]][self.x + 2*dist[1]] not in ('1', '4', '5'): # If after cargo air continue
                    if self.map[self.y][self.x] == '6': #If he was not on target
                        info_to_change.append((self.y,self.x,'2')) # Change the pos to empty
                    if self.map[self.y][self.x] == '7': # If he was on target
                        info_to_change.append((self.y,self.x,'3')) # Change the pos to target  
                    
                    self.y, self.x = self.y + dist[0], self.x + dist[1] # Set x,y to new values
                    info_to_change.append((self.y, self.x, '7')) # Change new pos to kepper & target
                    
                    if self.map[self.y + dist[0]][self.x + dist[1]] == '2': # If after cargo empty
                        info_to_change.append((self.y + dist[0],self.x + dist[1],'4')) # Set new pos to cargo
                    if self.map[self.y + dist[0]][self.x + dist[1]] == '3': #If after cargo target
                        info_to_change.append((self.y + dist[0],self.x + dist[1],'5')) # Set new pos cargo & target

            case _: # Otherwise something went wrong
                raise Exception('Invalid map')
            
        return info_to_change


               
