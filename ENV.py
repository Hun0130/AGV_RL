from distutils.log import info
from OBJ import obj
from AGV import AGV
import os

# Environment class
class env():
    # Color of AGVs
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    
    # Position of AGVs
    AGV1_POS = (10, 5)
    AGV2_POS = (10, 10)
    AGV3_POS = (10, 15)
    
    # Position of Buffers
    BUFFER1_POS = (8, 5)
    BUFFER2_POS = (8, 10)
    BUFFER3_POS = (8, 15)
    
    # Position of Machines
    MACHINE1_POS = (12, 5)
    MACHINE2_POS = (12, 10)
    MACHINE3_POS = (12, 15)
    
    def __init__(self):
        # AGV objects
        self.agv1 = AGV(self.AGV1_POS, self.RED)
        self.agv2 = AGV(self.AGV2_POS, self.GREEN)
        self.agv3 = AGV(self.AGV3_POS, self.BLUE)
        
        # Buffer objects
        self.buffer1 = obj(self.BUFFER1_POS, color = self.RED)
        self.buffer2 = obj(self.BUFFER2_POS, color = self.GREEN)
        self.buffer3 = obj(self.BUFFER3_POS, color = self.BLUE)
        
        # Machine objects
        self.machine1 = obj(self.MACHINE1_POS, color = self.RED)
        self.machine2 = obj(self.MACHINE2_POS, color = self.GREEN)
        self.machine3 = obj(self.MACHINE3_POS, color = self.BLUE)
        
        # Use for training
        self.state_list = []
        self.update_state()
        
        # All products produced
        self.products_num = [0, 0, 0]
        
        # highest reward
        self.high_reward = 0

        # previous product number
        self.prev_products_num = 0

        # whole_reward
        self.whole_reward = 0
        
        # Time
        self.time = 0

    # Check the AGV is out of factory or not
    def Out_Of_Factory(self, pos):
        return (pos[0] >= 20 or pos[0] < 0 or pos[1] >= 20 or pos[1] < 0)

    # 1 Step of ENV
    def step(self, action, episode):
        # the number of steps
        self.time += 1
        
        # # end of episode
        # if self.time == episode:
        #     self.time = 1
        #     self.Reset()
        #     return False
        
        # Use for GUI
        info_list = []
        
        agvs_pos = []
        agvs_pos.append(self.agv1.move(action[0:4]))
        agvs_pos.append(self.agv2.move(action[4:8]))
        agvs_pos.append(self.agv3.move(action[8:12]))
        info_list.append(agvs_pos)
        
        # AGV's position is available or not
        agvs_out = []
        if self.Out_Of_Factory(self.agv1.head.pos):
            self.agv1.reset(self.AGV1_POS, (255, 0, 0))
            agvs_out.append(1)
        else:
            agvs_out.append(0)
        if self.Out_Of_Factory(self.agv2.head.pos):
            self.agv2.reset(self.AGV2_POS, (0, 255, 0))
            agvs_out.append(1)
        else:
            agvs_out.append(0)
        if self.Out_Of_Factory(self.agv3.head.pos):
            self.agv3.reset(self.AGV3_POS, (0, 0, 255))
            agvs_out.append(1)
        else:
            agvs_out.append(0)
        info_list.append(agvs_out)

        # AGV's Loads
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
        info_list.append(agvs_load)

        # Machine produced product
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
        info_list.append(machines_product)
        
        return info_list
    
    # Get the list of object
    def Get_Obj(self):
        return [self.agv1, self.agv2, self.agv3, self.buffer1, self.buffer2, 
                self.buffer3, self.machine1, self.machine2, self.machine3] 
        
    def Get_AGV(self):
        return [self.agv1, self.agv2, self.agv3]
    
    def Get_Buffer(self):
        return [self.buffer1, self.buffer2, self.buffer3]
    
    def Get_Machine(self):
        return [self.machine1, self.machine2, self.machine3] 
    
    # Reset the environment
    def Reset(self):
        self.agv1 = AGV(self.AGV1_POS, self.RED)
        self.agv2 = AGV(self.AGV2_POS, self.GREEN)
        self.agv3 = AGV(self.AGV3_POS, self.BLUE)
        
        self.buffer1 = obj(self.BUFFER1_POS, color = self.RED)
        self.buffer2 = obj(self.BUFFER2_POS, color = self.GREEN)
        self.buffer3 = obj(self.BUFFER3_POS, color = self.BLUE)
        
        self.machine1 = obj(self.MACHINE1_POS, color = self.RED)
        self.machine2 = obj(self.MACHINE2_POS, color = self.GREEN)
        self.machine3 = obj(self.MACHINE3_POS, color = self.BLUE)
        
        # All products produced
        self.products_num = [0, 0, 0]
        
        # Time
        self.time = 1
        
        # Use for training
        self.update_state()
        
        # previous product number
        self.prev_products_num = 0
        
        # whole_reward
        self.whole_reward = 0
        
        # Use for GUI
        info_list = []
        info_list.append([self.agv1.head.pos, self.agv2.head.pos, self.agv3.head.pos])
        info_list.append([self.Out_Of_Factory(self.agv1.head.pos), self.Out_Of_Factory(self.agv2.head.pos), 
                        self.Out_Of_Factory(self.agv3.head.pos)])
        info_list.append([self.agv1.load, self.agv2.load, self.agv3.load])
        info_list.append([0, 0, 0])
        return info_list
    
    # Get product list of machines
    def Get_product(self):
        return self.products_num
    
    # Get whole products / running time
    def Get_throuput(self):
        product_num = 0
        for product in self.products_num:
            product_num += product
            
        return product_num / self.time
    
    # Update state for training
    def update_state(self):
        self.state_list = []     
        self.state_list.extend([self.Out_Of_Factory(self.agv1.head.pos), 
                                self.Out_Of_Factory(self.agv2.head.pos), self.Out_Of_Factory(self.agv3.head.pos)])
        self.state_list.extend([((self.agv1.head.pos == self.buffer1.pos) and (self.agv1.load == 0)),
                                ((self.agv2.head.pos == self.buffer2.pos) and (self.agv2.load == 0)),
                                ((self.agv3.head.pos == self.buffer3.pos) and (self.agv3.load == 0))])
        self.state_list.extend([((self.agv1.head.pos == self.machine1.pos) and self.agv1.load == 1), 
                                ((self.agv2.head.pos == self.machine2.pos) and self.agv2.load == 1), 
                                ((self.agv3.head.pos == self.machine3.pos) and self.agv3.load == 1)])
        self.state_list.extend([self.agv1.dirnx, self.agv1.dirny, self.agv2.dirnx, self.agv2.dirny, self.agv3.dirnx, self.agv3.dirny])
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
        
        for x in range(len(self.state_list)):
            if(self.state_list[x] == False):
                self.state_list[x] = 0
            if(self.state_list[x] == True):
                self.state_list[x] = 1
        return
    
    # Update reward for training
    def get_reward(self):
        # Reward 1
        reward = self.products_num[0] + self.products_num[1] + self.products_num[2] - self.prev_products_num
        self.prev_products_num = self.products_num[0] + self.products_num[1] + self.products_num[2]
        self.whole_reward += reward
        
        # Reward 2
        # reward = 0
        # if (self.agv1.head.pos == self.buffer1.pos) and (self.agv1.load == 0):
        #     reward += 10
        # if (self.agv2.head.pos == self.buffer2.pos) and (self.agv2.load == 0):
        #     reward += 10
        # if (self.agv3.head.pos == self.buffer3.pos) and (self.agv3.load == 0):
        #     reward += 10           
        # if (self.agv1.head.pos == self.machine1.pos) and (self.agv1.load == 1):
        #     reward += 10
        # if (self.agv2.head.pos == self.machine2.pos) and (self.agv2.load == 1):
        #     reward += 10
        # if (self.agv3.head.pos == self.machine3.pos) and (self.agv3.load == 1):
        #     reward += 10
        # self.whole_reward += reward
        # # if self.Out_Of_Factory(self.agv1.head.pos):
        # #     reward -= 0.1
        # # if self.Out_Of_Factory(self.agv2.head.pos):
        # #     reward -= 0.1
        # # if self.Out_Of_Factory(self.agv3.head.pos):
        # #     reward -= 0.1
        return reward
        