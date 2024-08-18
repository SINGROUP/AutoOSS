# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.preprocessing import MinMaxScaler

import glob
from tqdm import tqdm


# Define dataset class


class TopoDataset():
    def __init__(self, data_path, label_true=True):
        # .pkl file path
        with open(os.path.join(data_path, 'succ_diss_dataset.pkl'), 'rb') as file:
            # Serialize and write the variable to the file
            self.data_list_true=pickle.load(file)
            
        with open(os.path.join(data_path, 'failure_dataset.pkl'), 'rb') as file:
            # Serialize and write the variable to the file
            self.data_list_false=pickle.load(file)
        
        self.data_list=self.data_list_true+self.data_list_false
        self.len=len(self.data_list)
  
        
    def __getitem__(self, index): 
        topography=self.data_list[index]
        if index < len(self.data_list_true):
            label=np.array([1, 0])
        else:
            label=np.array([0, 1])
            
        scaler = MinMaxScaler(feature_range=(-1, 1))
        topography = scaler.fit_transform(topography.reshape(-1, 1))
        return topography, label

        # return topography, label
    
    def __len__(self):
        return self.len





# Define dataset loader
data_path='/scratch/work/wun2/stm_signal_img/dataset/topography/'
dataset=TopoDataset(data_path)
train_dataset, test_dataset=train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
device=torch.cuda.is_available()
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True, num_workers=0)


