import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque, namedtuple
import copy

from .initi_update import soft_update, hard_update, weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
#LOG_SIG_MAX = -1
#LOG_SIG_MIN = 1
epsilon = 1e-6

class GaussianPolicy(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_actions: int,
                 hidden_dim: int,
                 action_space: namedtuple=None):
        """

        Parameters
        ----------
        num_inputs: int
        num_actions: int
        hidden_dim: int
        action_space: namedtuple

        Returns
        -------
        """
        super(GaussianPolicy,self).__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)

        if action_space is None:
            self.action_scale = torch.tensor([1, 1, 1/3, 0.25])
            self.action_bias = torch.tensor([0, 0, 2/3, 0.75])
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        self.apply(weights_init_)

    def forward(self, state: np.array):
        """

        Parameters
        ----------
        state: array_like

        Returns
        -------
        mean, log_std
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Parameters
        ----------
        state: array_like

        Returns
        -------
        action, log_prob, mean
        """
        mean, log_std = self.forward(state)

        with open('gaussian_mean.txt', 'a') as file_mean:
            file_mean.write('%s\n' % (mean))

        with open('gaussian_std.txt', 'a') as file_std:
            file_std.write('%s\n' % (log_std.exp()))
   
        # print('Gaussianplolicy mean:', mean, 'log_std:', log_std)
        std = log_std.exp()
        normal = Normal(mean,std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t*self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # print('Gaussianplolicy action:', action, 'log_prob:', log_prob, 'mean:', mean)
        with open('mean_std_after_reparam.txt', 'a') as file_mean_std_after:
            file_mean_std_after.write('%s \t %s \t %s\n' % (action, log_prob, mean))
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)