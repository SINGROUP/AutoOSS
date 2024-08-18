import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque, namedtuple
import copy
from .initi_update import soft_update, hard_update, weights_init_


class Actor(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        """
        Initialize the Actor network.

        Parameters
        ----------
        num_inputs: int
        num_actions: int
        hidden_dim: int

        Returns
        -------
        None
        """
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(num_inputs , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)


        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = nn.Sigmoid()(x)

        return x
    
class Critic(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        """
        Initialize the Critic network.

        Parameters
        ----------
        num_inputs: int
        num_actions: int
        hidden_dim: int

        Returns
        -------
        None
        """
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)


        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)