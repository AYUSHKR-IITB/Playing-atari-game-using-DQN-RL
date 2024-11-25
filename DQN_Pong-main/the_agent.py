import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from agent_memory import Memory
from collections import deque

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the input features for the fully connected layer
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Agent:
    def __init__(self, possible_actions, starting_mem_len, max_mem_len, starting_epsilon, learn_rate, starting_lives=5, debug=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = Memory(max_mem_len)
        self.possible_actions = possible_actions
        self.epsilon = starting_epsilon
        self.epsilon_decay = .9/100000
        self.epsilon_min = .05
        self.gamma = .95
        self.learn_rate = learn_rate
        self.model = DQN(len(possible_actions)).to(self.device)
        self.model_target = DQN(len(possible_actions)).to(self.device)
        self.model_target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)
        self.total_timesteps = 0
        self.lives = starting_lives
        self.starting_mem_len = starting_mem_len
        self.learns = 0

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            # state = np.moveaxis(state, -1, 1)
            q_values = self.model(state)
            action_idx = q_values.max(1)[1].item()
            return self.possible_actions[action_idx]

    def _index_valid(self, index):
        if self.memory.done_flags[index-3] or self.memory.done_flags[index-2] or self.memory.done_flags[index-1] or self.memory.done_flags[index]:
            return False
        return True

    def learn(self, debug=False):
        # print(len(self.starting_mem_len))
        if len(self.memory.frames) < self.starting_mem_len:
            return

        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        # Sample batch
        while len(states) < 32:
            index = np.random.randint(4, len(self.memory.frames) - 1)
            if self._index_valid(index):
                state = [self.memory.frames[index-3], self.memory.frames[index-2],
                        self.memory.frames[index-1], self.memory.frames[index]]
                state = np.moveaxis(state, -1, 1)/255
                next_state = [self.memory.frames[index-2], self.memory.frames[index-1],
                            self.memory.frames[index], self.memory.frames[index+1]]
                next_state = np.moveaxis(next_state, -1, 1)/255

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index+1])
                next_done_flags.append(self.memory.done_flags[index+1])
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor([self.possible_actions.index(a) for a in actions_taken]).to(self.device)
        next_rewards = torch.FloatTensor(next_rewards).to(self.device)
        next_done_flags = torch.FloatTensor(next_done_flags).to(self.device)
        # print(states.size(),next_states.size())
        # Compute current Q values
        # state = np.moveaxis(state, -1, 0)

# Predict Q-values for current and next states
        labels = self.model(states)  # [32, 3]
        next_state_values = self.model_target(next_states)  # [32, 3]

        # Compute max Q-value for the next states
        max_next_q_values, _ = torch.max(next_state_values, dim=1)  # [32]

        # Update the Q-values using the Bellman equation
        updated_q_values = next_rewards + (1 - next_done_flags) * self.gamma * max_next_q_values  # [32]
        labels = labels.clone()  # Detach the tensor to prevent gradient issues
        labels[torch.arange(32), actions] = updated_q_values
        # print(current_q_values.size(),next_q_values.size(),target_q_values.size())
        # Compute loss and update
        
        criterion = nn.SmoothL1Loss()  # Loss function (e.g., Mean Squared Error)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)  # Optimizer

        # Forward pass and compute loss
        predicted_q_values = self.model(states)  # [32, 3]
        loss = criterion(predicted_q_values, labels)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update epsilon and learn counter
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1

        # Update target network
        if self.learns % 10000 == 0:
            self.model_target.load_state_dict(self.model.state_dict())
            print('\nTarget model updated')

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model_target.load_state_dict(self.model.state_dict())
