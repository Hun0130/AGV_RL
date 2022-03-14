from decimal import DecimalTuple
from distutils.log import info
import pygame
from OBJ import obj
from AGV import AGV
import DQN
import numpy as np
import os

class ENV():
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    
    AGV1_POS = (10, 5)
    AGV2_POS = (10, 10)
    AGV3_POS = (10, 15)
    
    BUFFER1_POS = (8, 5)
    BUFFER2_POS = (8, 10)
    BUFFER3_POS = (8, 15)
    
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
        
        # All products produced
        self.products_num = [0, 0, 0]
        
        # Running Option : 0 = random, 1 = deterministic, 2 = DQN, 3 = DQN Learned model
        self.running_opt = 0
        
        # Time
        self.time = 0
        
        # Use for training
        self.state_list = []
        self.update_state()
        
        # n_game
        self.n_game = 0
        
        # highest reward
        self.high_reward = 0
        
        # previous product number
        self.prev_products_num = 0
        
        # whole_reward
        self.whole_reward = 0
        
        # ================ Training parameters ===================
        # episode : the number of step of 1 episode
        self.episode = 1500
        
        # epoch : whole number of epoch with training
        self.epoch = 1000
        
        # learning rate
        self.learning_rate = 0.001
        
        # Gamma
        self.gamma = 0.9
        
        # batch_size
        self.batch_size = 30
        
        # training interval
        self.training_interval = 30
        
        # model name to be saved
        self.model_name = ""
        # ================ Training parameters ===================
        
        # DQN trainer
        self.trainer = DQN.QTrainer(DQN.Qnet(len(self.state_list), 256, 12), self.learning_rate, self.gamma, self.epoch, self.episode, self.batch_size)

    # Check the AGV is out of factory of not
    def Out_Of_Factory(self, pos):
        return (pos[0] >= 20 or pos[0] < 0 or pos[1] >= 20 or pos[1] < 0)

    # Single Process Step
    def Run(self, mode = 0):  
        self.time += 1
        # Use for GUI
        info_list = []
        
        # Random Move
        if self.running_opt == 0:
            if self.time == self.episode:
                self.time = 1
                self.Reset()
                return False
            
            agvs_pos = []
            agvs_pos.append(self.agv1.random_move(agvs_pos))
            agvs_pos.append(self.agv2.random_move(agvs_pos))
            agvs_pos.append(self.agv3.random_move(agvs_pos))
            info_list.append(agvs_pos)
        
        # Deterministic Move
        if self.running_opt == 1:
            if self.time == self.episode:
                self.time = 1
                self.Reset()
                return False
        
            buffers_pos = [self.buffer1.pos, self.buffer2.pos, self.buffer3.pos]
            machines_pos = [self.machine1.pos, self.machine2.pos, self.machine3.pos]
            agvs_pos = []
            agvs_pos.append(self.agv1.deterministic_move(agvs_pos, buffers_pos[0], machines_pos[0]))
            agvs_pos.append(self.agv2.deterministic_move(agvs_pos, buffers_pos[1], machines_pos[1]))
            agvs_pos.append(self.agv3.deterministic_move(agvs_pos, buffers_pos[2], machines_pos[2]))
            info_list.append(agvs_pos)
        
        # Deep Q Network Learning
        if self.running_opt == 2:
            self.update_state()
            state = self.state_list
            action = self.trainer.get_action(self.state_list)
            
            agvs_pos = []
            agvs_pos.append(self.agv1.dqn_move(action[0:4], agvs_pos))
            agvs_pos.append(self.agv2.dqn_move(action[4:8], agvs_pos))
            agvs_pos.append(self.agv3.dqn_move(action[8:12], agvs_pos))
            info_list.append(agvs_pos)
            # Update Reward
            reward = self.get_reward()
        
        if self.running_opt == 3:
            self.update_state()
            state = self.state_list
            action = self.trainer.get_action_test(self.state_list)
            
            agvs_pos = []
            agvs_pos.append(self.agv1.dqn_move(action[0:4], agvs_pos))
            agvs_pos.append(self.agv2.dqn_move(action[4:8], agvs_pos))
            agvs_pos.append(self.agv3.dqn_move(action[8:12], agvs_pos))
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
        
        # Training parts
        if self.running_opt == 2:
            self.update_state()
            next_state = self.state_list
            
            # train short memory
            # self.trainer.train_short_memory(state, action, reward, next_state, False)
            
            # memorize
            self.trainer.remember(state, action, reward, next_state, False)
            
            if self.time % self.training_interval == 0:
                self.trainer.train_long_memory()
            
            if self.time > self.episode:
                self.time = 1
                self.n_game += 1
                self.trainer.n_game += 1
                # new High score 
                if(self.whole_reward > self.high_reward):
                    self.high_reward = self.whole_reward
                    if os.path.isfile(('DQN_save/' + self.model_name)):
                        os.remove(('DQN_save/' + self.model_name))
                    self.model_name = 'model_' + str(self.n_game) + '.pth'
                    self.trainer.model.save(self.model_name)
                train_result = self.Get_training_record()
                self.Reset()
                self.update_state()
                if mode == 1:
                    return train_result  
                return [info_list, train_result]
                
                
            if self.n_game > self.trainer.epoch:
                self.time = 1
                self.Reset()
                return False     

        return info_list
    
    # Get the list of object
    def Get_Obj(self):
        return [self.agv1, self.agv2, self.agv3, self.buffer1, self.buffer2, 
                self.buffer3, self.machine1, self.machine2, self.machine3] 
        
    # Get training result
    def Get_training_record(self):
        record = ""
        record += 'Game: ' + str(self.n_game)
        record += ' Score: ' + str(self.whole_reward)
        record += ' Record: ' + str(self.high_reward)
        record += ' Random: ' + str(round(((self.trainer.epsilon / (self.trainer.epoch)) * 100), 1)) + '%'
        return record
    
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
        self.state_list = []
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
        