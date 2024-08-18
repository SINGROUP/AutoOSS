import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque, namedtuple
import copy
from .initi_update import soft_update, hard_update, weights_init_


class QNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        """
        Initialize the Q network.

        Parameters
        ----------
        num_inputs: int
        num_actions: int
        hidden_dim: int

        Returns
        -------
        None
        """
        super(QNetwork,self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.fc4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state,action],1)
        x1 = F.relu(self.fc1(sa))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(sa))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2

