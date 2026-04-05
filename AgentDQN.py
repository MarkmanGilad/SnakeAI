import torch
import random
import math
from DQN import *
from Graphics import *
from Constant import *

class AgentDQN:
    def __init__(self, parametes_path = None, train = True, env= None, devive = torch.device('cpu')):
        self.DQN = DQN(device=devive)
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.train = train
        self.setTrainMode()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_action (self, state, epoch = 0, events= None, train = True) -> tuple:
        actions = ACTIONS
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        state_tensor = state.to_tensor(device=self.DQN.device)
        with torch.no_grad():
            Q_values = self.DQN(state_tensor)
        max_index = torch.argmax(Q_values)
        return actions[max_index]

    def get_Actions_Values (self, states):
        with torch.no_grad():
            Q_values = self.DQN(states)
            max_values, max_indices = torch.max(Q_values,dim=1) # best_values, best_actions
        actions = max_indices + 1  # actions are [1,2,3,4] indices are [0,1,2,3]
        
        return actions.reshape(-1,1), max_values.reshape(-1,1)

    def Q (self, states, actions):
        Q_values = self.DQN(states) # try: Q_values = self.DQN(states).gather(dim=1, actions) ; check if shape of actions is [-1, 1] otherwise dim=0
        rows = torch.arange(Q_values.shape[0]).reshape(-1,1)
        cols = actions.reshape(-1,1)
        return Q_values[rows, cols-1]

    def epsilon_greedy(self, epoch, start=EPSILON_START, final=EPSILON_FINAL, decay=EPSILON_DECAY):
        # res = final + (start - final) * math.exp(-1 * epoch/decay)
        if epoch < decay:
            return start - (start - final) * epoch/decay
        return final
        
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def fix_update (self, dqn):
        self.DQN.load_state_dict(dqn.state_dict())


    def __call__(self, events= None, state=None):
        return self.get_Action(state)
