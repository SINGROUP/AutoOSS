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


import cv2
import glob
from tqdm import tqdm



# Define dataset class

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list, img_channel=1):
        # X is dissassemble data, here choose topography as input, y is the label to check if dissociation happens
        self.dataset_list = dataset_list

        self.img_channel=img_channel
        
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        if self.img_channel==1:
            x=cv2.imread(self.dataset_list[idx], cv2.IMREAD_GRAYSCALE)
        elif self.img_channel==3:
            x=cv2.imread(self.dataset_list[idx])
        x=cv2.resize(x, (128, 128), interpolation = cv2.INTER_AREA)  # resize to 128*128
        
        data=self.dataset_list[idx]
        data_type=data.split('/')[-2]

        if data_type=='br2me4dpp':
            label=np.array([1, 0, 0])
            
        elif data_type=='brme4dpp':
            label=np.array([0, 1, 0])
            
        # elif data_type=='me4dpp':
        #     label=np.array([0, 1, 0])
            
        else:
            label=np.array([0, 0, 1])
            
        # if data_type=='br2me4dpp':
        #     label=np.array([1, 0])
            
        # elif data_type=='brme4dpp':
        #     label=np.array([1, 0])
            
        # elif data_type=='me4dpp':
        #     label=np.array([1, 0])
            
        # else:
        #     label=np.array([0, 1])
        
        # if data_type=='br2me4dpp':
        #     label=np.array([1, 0])

        # elif data_type=='brme4dpp':
        #     label=np.array([0, 1])

        # elif data_type=='me4dpp':
        #     label=np.array([0, 1])
 
        return x, label


# Define dataset class
dataset_path='/scratch/work/wun2/stm_signal_img/dataset'


# Obtain list of images
# dataset_train_list = glob.glob(os.path.join(dataset_path, 'imgs/train/*dpp/*.png'), recursive = True)
# dataset_test_list = glob.glob(os.path.join(dataset_path, 'imgs/test/*dpp/*.png'), recursive = True)

# # # Obtain list of images
dataset_train_list = glob.glob(os.path.join(dataset_path, 'imgs/train/**/*.png'), recursive = True)
dataset_test_list = glob.glob(os.path.join(dataset_path, 'imgs/test/**/*.png'), recursive = True)

# Define dataset loader

train_dataset=ImgDataset(dataset_train_list)
test_dataset=ImgDataset(dataset_test_list)

device=torch.cuda.is_available()
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True, num_workers=0)


