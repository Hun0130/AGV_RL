import random
from OBJ import obj

# AGV Object
class AGV():
    turns = {}
    
    # POS is given as coordinates on the grid ex (1,5)
    def __init__(self, pos, color):
        # Color of AGV
        self.color = color
        self.head = obj(pos, 1, 0, color)
        self.dirnx = 0
        self.dirny = 1
        
        # Check the AGV has load or not
        self.load = 0
    
    def move(self, move, others_pos_list = []):        
        keys = 0
        if move == [1, 0, 0, 0]:
            keys = 1
        if move == [0, 1, 0, 0]:
            keys = 2
        if move == [0, 0, 1, 0]:
            keys = 3
        if move == [0, 0, 0, 1]:
            keys = 4
            
        # Move Left
        if keys == 1:
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        
        # Move Right
        elif keys == 2:
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        
        # Move Up
        elif keys == 3:
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        
        # Move Down
        elif keys == 4:
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        
        next_pos = self.head.pos[0] + self.dirnx, self.head.pos[1] + self.dirny
        
        if others_pos_list:
            for pos in others_pos_list:
                if next_pos == pos:
                    return self.head.pos

        self.head.move(self.dirnx, self.dirny)
        return self.head.pos
    
    def reset(self, pos, color):
        # Color of AGV
        self.head = obj(pos, 1, 0, color)
        self.dirnx = 0
        self.dirny = 1
    
    def draw(self, surface):
        self.head.draw(surface, True)
        
    def get_load(self):
        self.load = 1
        
    def pop_load(self):
        self.load = 0
        return True
