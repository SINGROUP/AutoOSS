import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque, namedtuple
import copy
import pandas as pd

class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        """
        Parameters
        ----------
        capacity: int
            max length of buffer deque

        Returns
        -------
        None
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self,
             state: list,
             action: list,
             reward: list,
             next_state: list,
             mask: list):
        """
        Insert a new memory into the end of the ReplayMemory buffer

        Parameters
        ----------
        state, action, reward, next_state, mask: array_like
        Returns
        -------
        None
        """
        self.buffer.insert(0, (state, action, reward, next_state, mask))

    def sample(self, batch_size, c_k):
        N = len(self.buffer)
        if c_k>N:
            c_k = N
        indices = np.random.choice(c_k, batch_size)
        batch = [self.buffer[idx] for idx in indices]
        #batch = random.sample(self.buffer,batch_size)
        state, action, reward, next_state, mask = map(np.stack,zip(*batch))
        return state, action, reward, next_state, mask

    def __len__(self):
        return len(self.buffer)

class HerReplayMemory(ReplayMemory):
    def __init__(self,
                 capacity: int,
                 env,
                 strategy: str='final'):
        """
        Initialize HerReplayMemory object

        Parameters
        ----------
        capacity: int
        env: AMRL.RealExpEnv
        strategy: str

        Returns
        -------
        None
        """
        super(HerReplayMemory, self).__init__(capacity)
        self.env = env
        self.n_sampled_goal = 2
        self.strategy = strategy


    def sample(self,
               batch_size: int,
               c_k: float) -> tuple:
        """
        Sample batch_size (state, action, reward, next_state, mask) # of memories
        from the HERReplayMemory, emphasizing the c_k most recent experiences.

        Also implemented: hindsight experience replay, which treats
        memories in which the achieved goal was different than the intended goal
        as 'succesful' in order to speed up training.


        Parameters
        ----------
        batch_size: int
        c_k: int
            select from the c_k most recent memories

        Returns
        -------
        tuple
        """
        N = len(self.buffer)
        if c_k>N:
            c_k = N
        indices = np.random.choice(c_k, int(batch_size))
        batch = []
        for idx in indices:
            # batch.append(self.buffer[idx])
            # state, action, reward, next_state, mask = self.buffer[idx]
            #print('old state:', state, 'old next state:', next_state, 'old reward:', reward)
            final_idx = self.sample_goals(idx)
            for fi in final_idx:
                new_state, action, new_reward, new_next_state, mask = self.buffer[fi]
                m = (new_state, action, new_reward, new_next_state, mask)
                batch.append(m)
        print('No. of samples:', len(batch))
        state, action, reward, next_state, mask = map(np.stack,zip(*batch))
        return state, action, reward, next_state, mask

    def sample_goals(self, idx):
        """
        Sample memories in the same episode

        Parameters
        ----------
        idx: int

        Returns
        -------
        array_like
            list of final_idx HerReplayMemory buffer indices
        """
        #get done state idx
        i = copy.copy(idx)

        while True:
            # print('#######sample goal i:', i)
            _,_,_,_,m = self.buffer[i]
            if not m:
                break
            else:
                i-=1
        if self.strategy == 'final' or i == idx:
            return [i]
        elif self.strategy == 'future':
            iss = np.random.choice(np.arange(i, idx+1), min(idx-i+1, 3))
            return iss

    # def sample(self,
    #            batch_size: int,
    #            c_k: float) -> tuple:
    #     """
    #     Sample batch_size (state, action, reward, next_state, mask) # of memories
    #     from the HERReplayMemory, emphasizing the c_k most recent experiences.

    #     Also implemented: hindsight experience replay, which treats
    #     memories in which the achieved goal was different than the intended goal
    #     as 'succesful' in order to speed up training.


    #     Parameters
    #     ----------
    #     batch_size: int
    #     c_k: int
    #         select from the c_k most recent memories

    #     Returns
    #     -------
    #     tuple
    #     """
    #     N = len(self.buffer)
    #     if c_k>N:
    #         c_k = N
    #     indices = np.random.choice(c_k, int(batch_size))
    #     batch = []
    #     for idx in indices:
    #         batch.append(self.buffer[idx])
    #         # state, action, reward, next_state, mask = self.buffer[idx]
    #         #print('old state:', state, 'old next state:', next_state, 'old reward:', reward)
    #         # final_idx = self.sample_goals(idx)
    #         # for fi in final_idx:
    #         #     new_state, action, new_reward, new_next_state, mask = self.buffer[fi]
    #         #     m = (new_state, action, new_reward, new_next_state, mask)
    #         #     batch.append(m)
    #     print('No. of samples:', len(batch))
    #     state, action, reward, next_state, mask = map(np.stack,zip(*batch))
    #     return state, action, reward, next_state, mask

    # def sample_goals(self, idx):
    #     """
    #     Sample memories in the same episode

    #     Parameters
    #     ----------
    #     idx: int

    #     Returns
    #     -------
    #     array_like
    #         list of final_idx HerReplayMemory buffer indices
    #     """
    #     #get done state idx
    #     i = copy.copy(idx)

    #     while True:
    #         print('#######sample goal i:', i)
    #         _,_,_,_,m = self.buffer[i]
    #         if not m:
    #             break
    #         else:
    #             i-=1
    #     if self.strategy == 'final' or i == idx:
    #         return [i]
    #     elif self.strategy == 'future':
    #         iss = np.random.choice(np.arange(i, idx+1), min(idx-i+1, 3))
    #         return iss
        
    # def calculate_value(self, idx):
    #     """
    #     Calculate the value of a memory

    #     Parameters
    #     ----------
    #     idx: int

    #     Returns
    #     -------
    #     float
    #     """
    #     _, _, reward, _, _ = self.buffer[idx]
    #     return reward
    
    # def sample(self, batch_size: int,   
    #            c_k: float, bolzmann_beta=0.6, random=True) -> tuple:           #  prioritized sample
    #     """
    #     Prioritize memories in the buffer based on boltzan distribution

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     -------
    #     None

    #     """
    #     N = len(self.buffer)
    #     if c_k>N:
    #         c_k = N

    #     # calculate priority
    #     idxs=[i for i in range(c_k)]

        
    #     batch = []


    #     if random:
    #         select_idxs_random = np.random.choice(idxs, batch_size)
            
    #         for idx in select_idxs_random:
    #             state, action, reward, next_state, mask = self.buffer[idx]
    #             print('select_idxs_random:', idx, 'reward:', reward)
    #             batch.append(self.buffer[idx])
    #     else:
    #         priority = np.array([self.calculate_value(i) for i in range(c_k)])
    #         done_idxs = np.where(priority==1)[0].tolist()
    #         messy_idxs = np.where(priority!=1)[0].tolist()

    #         print('done_idxs:', done_idxs)
    #         if len(done_idxs) < batch_size:
    #             for idx in done_idxs:
    #                 state, action, reward, next_state, mask = self.buffer[idx]
    #                 batch.append(self.buffer[idx])
    #                 print('batch_reward_done', reward)
                

    #             select_idxs_undone = np.random.choice(messy_idxs, batch_size-len(done_idxs))
    #             print('select_idxs_undone:', select_idxs_undone)
    #             for idx in select_idxs_undone:
    #                 state, action, reward, next_state, mask = self.buffer[idx]
    #                 # batch.append(self.buffer[idx])       # adjust reward  
    #                 if reward>=0 and reward<1:
    #                     reward = 0
    #                 elif reward==-0.5:
    #                     reward = 0
    #                 elif reward==-2:
    #                     reward = -1
    #                 batch.append([state, action, reward, next_state, mask])  
                    
    #                 print('batch_reward_undone', reward)
    #         else:
    #             select_index_done = np.random.choice(done_idxs, batch_size)
    #             print('select_index_done:', select_index_done)
    #             for idx in select_index_done:
    #                 state, action, reward, next_state, mask = self.buffer[idx]
    #                 batch.append(self.buffer[idx])

    #     print('No. of samples:', len(batch))

    #     state, action, reward, next_state, mask = map(np.stack,zip(*batch))
    #     return state, action, reward, next_state, mask

#  Boltzmann distribution
        # boltzmann_distribution = np.exp(priority/bolzmann_beta)/np.sum(np.exp(priority/bolzmann_beta))
        # select_index=np.random.choice(idxs, batch_size, p=boltzmann_distribution)
        # print('select_index:', select_index)
        # batch = []
        # for idx in select_index:
        #     batch.append(self.buffer[idx])
        # print('No. of samples:', len(batch))

        
        
        

# selected based on the reward values.
        # priority_pd=pd.DataFrame({'idxs':idxs, 'priority':priority})
        # idxs_order=priority_pd.sort_values(by='priority', ascending=False)['idxs'].tolist()
        # selected_idxs = idxs_order[:batch_size]
        # batch = []
        # for idx in selected_idxs:
        #     batch.append(self.buffer[idx])
        # print('No. of samples:', len(batch))
        # state, action, reward, next_state, mask = map(np.stack,zip(*batch))
        # return state, action, reward, next_state, mask



