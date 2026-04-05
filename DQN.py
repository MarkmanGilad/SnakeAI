import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Constant import *
 

class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS, CONV_KERNEL_SIZE, CONV_STRIDE, padding=CONV_PADDING)
        self.conv2 = nn.Conv2d(CONV2_IN_CHANNELS, CONV2_OUT_CHANNELS, CONV_KERNEL_SIZE, CONV_STRIDE, padding=CONV_PADDING)
        self.conv3 = nn.Conv2d(CONV3_IN_CHANNELS, CONV3_OUT_CHANNELS, CONV_KERNEL_SIZE, CONV_STRIDE, padding=CONV_PADDING)
        self.fc = nn.Linear(FC_INPUT_SIZE, FC_HIDDEN_SIZE)
        self.output = nn.Linear(FC_HIDDEN_SIZE, OUTPUT_SIZE)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        x = F.relu(x)
        x = self.output(x)
        return x
    
    def loss (self, Q_values, rewards, Q_next_Values, dones ):
        Q_new = rewards.to(self.device) + GAMMA * Q_next_Values * (1- dones.to(self.device))
        return self.MSELoss(Q_values, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

    def __call__(self, states):
        return self.forward(states).to(self.device)