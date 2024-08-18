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




class sac_agent():
    def __init__(self,
                 num_inputs: int,
                 num_actions: int,
                 action_space: namedtuple,
                 device: torch.device,
                 hidden_size: int,
                 lr: float,
                 gamma: float,
                 tau: float,
                 alpha: float) -> None:
        """
        Initialize soft-actor critic agent for performing RL task and training

        Parameters
        ----------
        num_inputs: int
            number of input values to agent
        num_actions: int
            number of values output by agent
        action_space: namedtuple
            namedtuple called ACTION_SPACE with fields called 'high' and 'low'
            that are each torch tensors of length len(num_actions)
            which define the GaussianPolicy action_scale and action_bias
        device: torch.device
        hidden_size: int

        lr: float

        gamma: float

        tau: float

        alpha: float

        Returns
        -------
        None
        """

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device

        args = num_inputs, num_actions, hidden_size
        self.critic = QNetwork(*args).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = QNetwork(*args).to(self.device)
        hard_update(self.critic_target,self.critic)

        args = num_inputs, num_actions, hidden_size, action_space
        self.policy = GaussianPolicy(*args).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        kwargs = {'requires_grad':True,
                  'device':self.device,
                  'dtype':torch.float32}
        self.log_alpha = torch.tensor([np.log(alpha)], **kwargs)
        self.alpha = self.log_alpha.detach().exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        arg = torch.Tensor([num_actions]).to(self.device)
        self.target_entropy = -torch.prod(arg).item()

    def select_action(self,
                      state: np.array,
                      eval:bool=False):
        """

        Parameters
        ----------
        state: array_like
            should be of length num_inputs

        Returns
        -------
        action: array_like
            should be of length num_actions

        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self,
                          memory: HerReplayMemory,
                          batch_size: int,
                          c_k: float,
                          train_pi: bool = True,
                          save_loss_filename: str = None):
        """
        SAC agent training step

        Parameters
        ----------
        memory: HerReplayMemory or ReplayMemory object
        batch_size: int
            minibatch size, i.e. number of memories to sample per batch
        c_k: float
        train_pi: bool

        Returns
        -------
        """
        memories = memory.sample(batch_size, c_k)
        states, actions, rewards, next_states, masks = memories
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_state_log_pi, _ = self.policy.sample(next_states)
            q1_next_target, q2_next_target = self.critic_target(next_states, next_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)-self.alpha*next_state_log_pi
            next_q_value = rewards + masks*self.gamma*min_q_next_target

        q1, q2 = self.critic(states,actions)

        q_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        self.critic_optim.zero_grad()
        q_loss.backward() 
        critic_norm = self.get_grad_norm(self.critic)
        '''
        total_norm = 0
        for p in self.critic.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(total_norm)'''

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10, norm_type=2.0)
        self.critic_optim.step()

        if train_pi:
            pi, log_pi,_ = self.policy.sample(states)
            q1_pi, q2_pi = self.critic(states, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            policy_loss = (self.alpha*log_pi-min_q_pi).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            policy_norm = self.get_grad_norm(self.policy)
            print('Training','critic norm:', critic_norm, 'policy norm:', policy_norm)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 2, norm_type=2.0)
            self.policy_optim.step()

            alpha_loss = -(self.log_alpha*(log_pi+self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        if save_loss_filename is not None:
            with open('sac_training_log_%s.txt' % save_loss_filename,'a') as f:
                f.write('%s, %s, %s\n' % (q_loss.item(), policy_loss.item(), alpha_loss.item()))
        else:
            with open('sac_training_log.txt','a') as f:
                f.write('%s, %s, %s\n' % (q_loss.item(), policy_loss.item(), alpha_loss.item()))

        soft_update(self.critic_target,self.critic,self.tau)

    def get_grad_norm(self, net):
        """


        Parameters
        ----------
        net:

        Returns
        -------
        total_norm:
        """
        total_norm = 0
        for p in net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm
