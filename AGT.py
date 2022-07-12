import DQN
import random

class agt():
    
    def __init__(self, env):
        # Running Option : 0 = random, 1 = deterministic, 2 = DQN, 3 = DQN Learned model
        self.running_opt = 0

        # n_game
        self.n_game = 0

        # highest reward
        self.high_reward = 0

        # previous product number
        self.prev_products_num = 0

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
        self.trainer = DQN.QTrainer(DQN.Qnet(len(env.state_list), 256, 12), self.learning_rate, self.gamma, self.epoch, self.episode, self.batch_size)
        
    def action(self, agv_list = [], buffer_pos = [], machine_pos = [], state = []):
        action_list = []
        
        # Random Move
        if self.running_opt == 0:
            for num in range(3):
                keys = random.randint(1, 4)
                if keys == 1:
                    action_list.extend([1, 0, 0, 0])
                if keys == 2:
                    action_list.extend([0, 1, 0, 0])
                if keys == 3:
                    action_list.extend([0, 0, 1, 0])
                if keys == 4:
                    action_list.extend([0, 0, 0, 1])
        
        # Deterministic Move
        if self.running_opt == 1:
            for num in range(3):
                if agv_list[num].load:
                    if (agv_list[num].head.pos[0] - machine_pos[num].pos[0] == 0) and (agv_list[num].head.pos[1] - machine_pos[num].pos[1] == 0):
                        action_list.extend([0, 0, 0, 0])
                    elif (agv_list[num].head.pos[0] - machine_pos[num].pos[0] > 0) and (agv_list[num].head.pos[1] - machine_pos[num].pos[1] > 0):
                        action_list.extend([1, 0, 0, 0])
                    elif (agv_list[num].head.pos[0] - machine_pos[num].pos[0] > 0) and (agv_list[num].head.pos[1] - machine_pos[num].pos[1] == 0):
                        action_list.extend([1, 0, 0, 0])
                    elif (agv_list[num].head.pos[0] - machine_pos[num].pos[0] == 0) and (agv_list[num].head.pos[1] - machine_pos[num].pos[1] > 0):
                        action_list.extend([0, 0, 1, 0])
                    elif (agv_list[num].head.pos[0] - machine_pos[num].pos[0] < 0) and (agv_list[num].head.pos[1] - machine_pos[num].pos[1] < 0):
                        action_list.extend([0, 1, 0, 0])
                    elif (agv_list[num].head.pos[0] - machine_pos[num].pos[0] < 0) and (agv_list[num].head.pos[1] - machine_pos[num].pos[1] == 0):
                        action_list.extend([0, 1, 0, 0])
                    elif (agv_list[num].head.pos[0] - machine_pos[num].pos[0] == 0) and (agv_list[num].head.pos[1] - machine_pos[num].pos[1] < 0):
                        action_list.extend([0, 0, 0, 1])
                else:
                    if (agv_list[num].head.pos[0] - buffer_pos[num].pos[0] == 0) and (agv_list[num].head.pos[1] - buffer_pos[num].pos[1] == 0):
                        action_list.extend([0, 0, 0, 0])
                    elif (agv_list[num].head.pos[0] - buffer_pos[num].pos[0] > 0) and (agv_list[num].head.pos[1] - buffer_pos[num].pos[1] > 0):
                        action_list.extend([1, 0, 0, 0])
                    elif (agv_list[num].head.pos[0] - buffer_pos[num].pos[0] > 0) and (agv_list[num].head.pos[1] - buffer_pos[num].pos[1] == 0):
                        action_list.extend([1, 0, 0, 0])
                    elif (agv_list[num].head.pos[0] - buffer_pos[num].pos[0] == 0) and (agv_list[num].head.pos[1] - buffer_pos[num].pos[1] > 0):
                        action_list.extend([0, 0, 1, 0])
                    elif (agv_list[num].head.pos[0] - buffer_pos[num].pos[0] < 0) and (agv_list[num].head.pos[1] - buffer_pos[num].pos[1] < 0):
                        action_list.extend([0, 1, 0, 0])
                    elif (agv_list[num].head.pos[0] - buffer_pos[num].pos[0] < 0) and (agv_list[num].head.pos[1] - buffer_pos[num].pos[1] == 0):
                        action_list.extend([0, 1, 0, 0])
                    elif (agv_list[num].head.pos[0] - buffer_pos[num].pos[0] == 0) and (agv_list[num].head.pos[1] - buffer_pos[num].pos[1] < 0):
                        action_list.extend([0, 0, 0, 1])
                        
        # Deep Q Network Learning
        if self.running_opt == 2:
            action_list = self.trainer.get_action(state)
            
        # Deep Q Network Evaluation
        if self.running_opt == 3:
            action_list = self.trainer.get_action(state)
            
        return action_list
    
    # Get training result
    def Get_training_record(self, whole_reward):
        record = ""
        record += 'Game: ' + str(self.n_game)
        record += ' Score: ' + str(whole_reward)
        record += ' Record: ' + str(self.high_reward)
        record += ' Random: ' + str(round(((self.trainer.epsilon / (self.trainer.epoch)) * 100), 1)) + '%'
        return record