import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import collections
import random
import os

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
        pass
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        
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
    
    def train_step(self,state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        #(1, x)
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
        
        # backward propagation of loss
        loss.backward() 
        self.optimer.step()
        

    def get_action(self, state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        if(random.randint(0, 200) < self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).cuda()
            prediction = self.model(state0).cuda()  # prediction by model
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    