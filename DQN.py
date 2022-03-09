from gettext import translation
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import collections
import random
import os
import math
import numpy as np

# ReplayBuffer
class ReplayBuffer():
    buffer_limit  = 5000
    batch_size    = 32
    
    def __init__(self):
        self.buffer = collections.deque(maxlen = self.buffer_limit)
    
    # Save transition
    def push(self, transition):
        # transition: ('state', 'action', 'next_state', 'reward')
        self.buffer.append(transition)
    
    def sample(self):
        return random.sample(self.buffer, self.batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
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
    def __init__(self, model, learning_rate, gamma):
        # If gpu is to be used
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(torch.cuda.is_available())
        # print(torch.cuda.get_device_name(0))
        
        # learning rate for the optimizer.
        self.learning_rate = learning_rate
        # the discount rate used in Bellman equation
        self.gamma         = gamma
        # Linear NN defined at Qnet
        self.model = model
        # Opimizer for weight and biases updation
        self.optimer = optim.Adam(model.parameters(), lr = self.learning_rate)
        # Mean Squared error loss function
        self.criterion = nn.MSELoss()
        # Replay Buffer
        self.memory = ReplayBuffer()
        # Increase with training
        self.steps_done = 0
        
        # Random percent of agent when training start
        self.EPS_START = 0.9
        # Random percent of agent after training
        self.EPS_END = 0.05
        # Decreasing random parameter
        self.EPS_DECAY = 200

    
    def train_step(self, state, action, reward, next_state):       
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
            
        self.memory.push((state, action, next_state, reward))

        if len(self.memory) < self.memory.batch_size:
            return
        
        batch = self.memory.sample()
        state, action, reward, next_state = zip(*batch)
        state = torch.cat(state)
        next_state = torch.cat(next_state)
        action = torch.cat(action)
        reward = torch.cat(reward)
        
        # 1. Predicted Q value with current state
        q_out = self.model(state)
        current_q = q_out.gather(1, action)
        
        # 2. Q_new = reward + gamma * max(next_predicted Qvalue)
        max_next_q = self.model(next_state).detach().max(1)[0]
        expected_q = reward + (self.gamma * max_next_q)
        
        # backward propagation of loss
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimer.zero_grad()
        loss.backward() 
        self.optimer.step()
    
    # Save state in Replay buffer
    def memorize(self, state, action, reward, next_state):
        self.memory.push((state, action, reward, next_state))

    def get_action(self, state):
        # random moves: tradeoff explotation / exploitation
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        final_move = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        result = []
        # Act randomly
        if random.random() <= eps_threshold:
            for agv_num in range(len(final_move)):
                move = random.randint(0, 3)
                final_move[agv_num][move] = 1
                result.extend(final_move[agv_num])
    
        # NN network choose action
        else:
            # state0 = torch.tensor(state, dtype = torch.float).cuda()
            state0 = torch.tensor(state, dtype = torch.float)
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
    