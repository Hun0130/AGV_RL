import pygame
from OBJ import obj
from AGV import AGV
import DQN

class ENV():
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    
    def __init__(self):
        # AGV objects
        self.agv1 = AGV((10, 5), self.RED)
        self.agv2 = AGV((10, 10), self.GREEN)
        self.agv3 = AGV((10, 15), self.BLUE)
        
        # Buffer objects
        self.buffer1 = obj((3, 5), color = self.RED)
        self.buffer2 = obj((3, 10), color = self.GREEN)
        self.buffer3 = obj((3, 15), color = self.BLUE)
        
        # Machine objects
        self.machine1 = obj((17, 5), color = self.RED)
        self.machine2 = obj((17, 10), color = self.GREEN)
        self.machine3 = obj((17, 15), color = self.BLUE)
        
        # All products produced
        self.products_num = [0, 0, 0]
        
        # Running Option : 0 = random, 1 = deterministic, 2 = DQN, 3 = DQN Learned model
        self.running_opt = 0
        
        # State List
        self.state_list = []
        
        # Time
        self.time = 0
        
        # Use for training
        self.state_list = []

    # Check the AGV is out of factory of not
    def Out_Of_Factory(self, pos):
        return (pos[0] >= 20 or pos[0] < 0 or pos[1] >= 20 or pos[1] < 0)

    # Single Process Step
    def Run(self):  
        self.time += 1
        # Use for GUI
        info_list = []
        
        # Random Move
        if self.running_opt == 0:
            agvs_pos = []
            agvs_pos.append(self.agv1.random_move(agvs_pos))
            agvs_pos.append(self.agv2.random_move(agvs_pos))
            agvs_pos.append(self.agv3.random_move(agvs_pos))
            info_list.append(agvs_pos)
        
        # Deterministic Move
        if self.running_opt == 1:
            buffers_pos = [self.buffer1.pos, self.buffer2.pos, self.buffer3.pos]
            machines_pos = [self.machine1.pos, self.machine2.pos, self.machine3.pos]
            agvs_pos = []
            agvs_pos.append(self.agv1.deterministic_move(agvs_pos, buffers_pos[0], machines_pos[0]))
            agvs_pos.append(self.agv2.deterministic_move(agvs_pos, buffers_pos[1], machines_pos[1]))
            agvs_pos.append(self.agv3.deterministic_move(agvs_pos, buffers_pos[2], machines_pos[2]))
            info_list.append(agvs_pos)
        
        # Deep Q Network Learning
        if self.running_opt == 2:
            self.state_list = []
            pass
        
        if self.running_opt == 3:
            pass
        
        # AGV's position is available or not [0, 0, 0, 0, 0, 0, 0, 0, 0] (9)
        agvs_out = []
        if self.Out_Of_Factory(self.agv1.head.pos):
            self.agv1.reset((10, 5), (255, 0, 0))
            agvs_out.append(1)
        else:
            agvs_out.append(0)
        if self.Out_Of_Factory(self.agv2.head.pos):
            self.agv2.reset((10, 10), (0, 255, 0))
            agvs_out.append(1)
        else:
            agvs_out.append(0)
        if self.Out_Of_Factory(self.agv3.head.pos):
            self.agv3.reset((10, 15), (0, 0, 255))
            agvs_out.append(1)
        else:
            agvs_out.append(0)
            
        self.state_list.extend(agvs_out)
        info_list.append(agvs_out)

        # AGV's Loads : [0, 0, 0] (3)
        agvs_load = []
        if self.agv1.head.pos == self.buffer1.pos:
            self.agv1.get_load()
        if self.agv2.head.pos == self.buffer2.pos:
            self.agv2.get_load()
        if self.agv3.head.pos == self.buffer3.pos:
            self.agv3.get_load()
        agvs_load.append(self.agv1.load)
        agvs_load.append(self.agv2.load)
        agvs_load.append(self.agv3.load)
        self.state_list.extend(agvs_load)
        info_list.append(agvs_load)

        # Machine produced product [0, 0, 0] (3)
        machines_product = []
        if self.agv1.head.pos == self.machine1.pos:
            if self.agv1.load:
                self.agv1.pop_load()
                self.machine1.produce()
                machines_product.append(1)
                self.products_num[0] += 1
            else:
                machines_product.append(0)
        else:
            machines_product.append(0)
        if self.agv2.head.pos == self.machine2.pos:
            if self.agv2.load:
                self.agv2.pop_load()
                self.machine2.produce()
                machines_product.append(1)
                self.products_num[1] += 1
            else:
                machines_product.append(0)
        else:
            machines_product.append(0)
        if self.agv3.head.pos == self.machine3.pos:
            if self.agv3.load:
                self.agv3.pop_load()
                self.machine3.produce()
                machines_product.append(1)
                self.products_num[2] += 1
            else:
                machines_product.append(0)
        else:
            machines_product.append(1)
        self.state_list.extend(machines_product)
        info_list.append(machines_product)
        
        # (24)
        pos_relation = []
        pos_relation.append(self.agv1.head.pos[0] < self.machine1.pos[0])
        pos_relation.append(self.agv1.head.pos[0] > self.machine1.pos[0])
        pos_relation.append(self.agv1.head.pos[1] < self.machine1.pos[1])
        pos_relation.append(self.agv1.head.pos[1] > self.machine1.pos[1])
        pos_relation.append(self.agv2.head.pos[0] < self.machine2.pos[0])
        pos_relation.append(self.agv2.head.pos[0] > self.machine2.pos[0])
        pos_relation.append(self.agv2.head.pos[1] < self.machine2.pos[1])
        pos_relation.append(self.agv2.head.pos[1] > self.machine2.pos[1])
        pos_relation.append(self.agv3.head.pos[0] < self.machine3.pos[0])
        pos_relation.append(self.agv3.head.pos[0] > self.machine3.pos[0])
        pos_relation.append(self.agv3.head.pos[1] < self.machine3.pos[1])
        pos_relation.append(self.agv3.head.pos[1] > self.machine3.pos[1])
        
        pos_relation.append(self.agv1.head.pos[0] < self.buffer1.pos[0])
        pos_relation.append(self.agv1.head.pos[0] > self.buffer1.pos[0])
        pos_relation.append(self.agv1.head.pos[1] < self.buffer1.pos[1])
        pos_relation.append(self.agv1.head.pos[1] > self.buffer1.pos[1])
        pos_relation.append(self.agv2.head.pos[0] < self.buffer2.pos[0])
        pos_relation.append(self.agv2.head.pos[0] > self.buffer2.pos[0])
        pos_relation.append(self.agv2.head.pos[1] < self.buffer2.pos[1])
        pos_relation.append(self.agv2.head.pos[1] > self.buffer2.pos[1])
        pos_relation.append(self.agv3.head.pos[0] < self.buffer3.pos[0])
        pos_relation.append(self.agv3.head.pos[0] > self.buffer3.pos[0])
        pos_relation.append(self.agv3.head.pos[1] < self.buffer3.pos[1])
        pos_relation.append(self.agv3.head.pos[1] > self.buffer3.pos[1])
        self.state_list.extend(pos_relation)
        
        return info_list
        
    def Get_Obj(self):
        return [self.agv1, self.agv2, self.agv3, self.buffer1, self.buffer2, 
                self.buffer3, self.machine1, self.machine2, self.machine3] 
        
    def Reset(self):
        self.agv1 = AGV((10, 5), self.RED)
        self.agv2 = AGV((10, 10), self.GREEN)
        self.agv3 = AGV((10, 15), self.BLUE)
        
        self.buffer1 = obj((3, 5), color = self.RED)
        self.buffer2 = obj((3, 10), color = self.GREEN)
        self.buffer3 = obj((3, 15), color = self.BLUE)
        
        self.machine1 = obj((17, 5), color = self.RED)
        self.machine2 = obj((17, 10), color = self.GREEN)
        self.machine3 = obj((17, 15), color = self.BLUE)
        return
    
    def Get_product(self):
        return self.products_num
    
    def Get_throuput(self):
        product_num = 0
        for product in self.products_num:
            product_num += product
            
        return product_num / self.time