import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque, namedtuple
import copy
from .replay_memory import ReplayMemory
from .replay_memory import HerReplayMemory 
from .gaussianpolicy import GaussianPolicy
from .qnetwork import QNetwork
from .initi_update import soft_update, hard_update, weights_init_
from .actor_critic_net import Actor, Critic
import os


class ddpg_agent():
    def __init__(self,
                 num_inputs: int,
                 num_actions: int,
                 hidden_size: int,
                 device: torch.device,
                 lr: float,
                 gamma: float,
                 tau: float,) -> None:
        
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # define actor networks
        args = num_inputs, num_actions, hidden_size
        self.actor = Actor(*args).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)
        self.actor_target = Actor(*args).to(self.device)
        hard_update(self.actor_target,self.actor)



        # define critic networks
        args = num_inputs, num_actions, hidden_size
        self.critic = Critic(*args).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = Critic(*args).to(self.device)
        hard_update(self.critic_target,self.critic)


        self.criterion = nn.MSELoss()


        # define the directory to save the model and results


    def select_action(self, state) -> None:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        action = action.detach().cpu().numpy()[0]
        return action


    def update_parameters(self, 
                          memory: HerReplayMemory,
                          batch_size: int,
                          c_k: float,
                          train_mode: bool = True) -> None:
        memories = memory.sample(batch_size, c_k)
        states, actions, rewards, next_states, masks = memories
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next_values= self.critic_target(next_states, next_actions)
            q_target = rewards + masks*self.gamma*q_next_values

        # Critic update
        self.critic.zero_grad()
        q_values = self.critic(states,actions)
        q_loss = self.criterion(q_values, q_target.detach())
        self.critic_optim.zero_grad()
        q_loss.backward()
        critic_norm = self.get_grad_norm(self.critic)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10, norm_type=2.0)
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        policy_loss = -self.critic(states, self.actor(states)).mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
  

    def save_model(self, filepath='output_model_ddpg', filename='ddpg'):
        if os.path.exists(filepath):
            pass
        else:
            os.mkdir(filepath)
        torch.save(self.actor.state_dict(), '{}/actor_{}.pkl'.format(filepath, filename))
        torch.save(self.critic.state_dict(), '{}/critic_{}.pkl'.format(filepath, filename))
        # torch.save(self.actor_target.state_dict(), '{}/actor_target_{}.pkl'.format(filepath, filename))
        # torch.save(self.critic_target.state_dict(), '{}/critic_target_{}.pkl'.format(filepath, filename))
        # torch.save(self.actor_optim.state_dict(), '{}/actor_optim_{}.pkl'.format(filepath, filename))
        # torch.save(self.critic_optim.state_dict(), '{}/critic_optim_{}.pkl'.format(filepath, filename))


        

    def load_model(self, filepath='output_model_ddpg', filename='ddpg') -> None:
        torch.load('{}/actor_{}.pkl'.format(filepath, filename))
        torch.load('{}/critic_{}.pkl'.format(filepath, filename))
        # torch.load('{}/actor_target_{}.pkl'.format(filepath, filename))
        # torch.load('{}/critic_target_{}.pkl'.format(filepath, filename))
        # torch.load('{}/actor_optim_{}.pkl'.format(filepath, filename))
        # torch.load('{}/critic_optim_{}.pkl'.format(filepath, filename))


    def get_grad_norm(self, net):
        """
        """
        total_norm = 0
        for p in net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm
    
    def train_mode(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def eval_mode(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def to_device(self):
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

