from gettext import translation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import math
import numpy as np

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
    
# Neural Network Model Name: Qnet
class Qnet(nn.Module):
    # Model에서 사용될 module을 정의
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # modules ...
        # Linear: y=wx+b 형태의 선형 변환을 수행하는 메소드
        # 입력되는 x의 차원과 y의 차원
        self.model = nn.Sequential
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # some functions ...
        # ReLU는 max(0, x)를 의미하는 함수인데, 0보다 작아지게 되면 0이 되는 특징을 가지고 있습니다. 
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = 'DQN_save/'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer():
    def __init__(self, model, lr, gamma):        
        # learning rate for the optimizer.
        self.learning_rate = lr
        # the discount rate used in Bellman equation
        self.gamma         = gamma
        # Linear NN defined at Qnet
        self.model = model
        # Opimizer for weight and biases updation
        self.optimer = optim.Adam(model.parameters(), lr = self.learning_rate)
        # Mean Squared error loss function
        self.criterion = nn.MSELoss()
        
        self.n_game = 0
        # Randomness
        self.epsilon = 0 

        # popleft()
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Qnet(33,256,12) 
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n) 
        # self.model.to('cuda')   
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)         
        # TODO: model,trainer
    
    def train_step(self, state, action, reward, next_state, done):       
        state = torch.tensor(np.array(state, dtype = int), dtype = torch.float)
        next_state = torch.tensor(np.array(next_state, dtype = int), dtype = torch.float)
        action = torch.tensor(np.array(action, dtype = int), dtype = torch.float)
        reward = torch.tensor(np.array(reward, dtype = int), dtype = torch.float)
        
        if(len(state.shape) == 1):
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )
            
        # 1. Predicted Q value with current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
            
        # 2. Q_new = reward + gamma * max(next_predicted Qvalue)
        #pred.clone()
        #preds[argmax(action)] = Q_new
        self.optimer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward() # backward propagation of loss
        
        self.optimer.step()
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.train_step(state,action,reward,next_state,done)
    
    def get_action(self, state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 1100 - self.n_game
        
        final_move = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        result = []
        # Act randomly
        if(random.randint(0, 1200) < self.epsilon):
            for agv_num in range(len(final_move)):
                move = random.randint(0, 3)
                final_move[agv_num][move] = 1
                result.extend(final_move[agv_num])
    
        # NN network choose action
        else:
            # state0 = torch.tensor(state, dtype = torch.float).cuda()
            state0 = torch.tensor(state, dtype=torch.float)
            # prediction by model
            # prediction = self.model(state0).cuda()  
            prediction = self.model(state0)
            move1 = torch.argmax(prediction[0:4]).item()
            move2 = torch.argmax(prediction[4:8]).item()
            move3 = torch.argmax(prediction[8:12]).item()
            final_move[0][move1] = 1
            final_move[1][move2] = 1
            final_move[2][move3] = 1
            result.extend(final_move[0])
            result.extend(final_move[1])
            result.extend(final_move[2])
        return result