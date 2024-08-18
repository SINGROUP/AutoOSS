
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque, namedtuple
import copy


def soft_update(target,source,tau):
    """


    Parameters
    ----------

    Returns
    -------
    """
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)

def hard_update(target,source):
    """


    Parameters
    ----------

    Returns
    -------
    """
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(param.data)

def weights_init_(m):
    """
    Initialize weights in torch.nn object

    Parameters
    ----------
    m: torch.nn.Linear

    Returns
    -------
    None
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)