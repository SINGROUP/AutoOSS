import os
import time
import numpy as np

import pathlib
from PIL import Image
from typing import Tuple, Dict, List

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms

class ResNet(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.stg1 = nn.Sequential(
                                   nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3),
                                             stride=(1), padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        
        ##stage 2
        self.convShortcut2 = self.conv_shortcut(64,256,1)
        
        self.conv2 = self.block(64,[64,256],3,1,conv=True)
        self.ident2 = self.block(256,[64,256],3,1)

        
        ##stage 3
        self.convShortcut3 = self.conv_shortcut(256,512,2)
        
        self.conv3 = self.block(256,[128,512],3,2,conv=True)
        self.ident3 = self.block(512,[128,512],3,2)

        
        ##stage 4
        self.convShortcut4 = self.conv_shortcut(512,1024,2)
        
        self.conv4 = self.block(512,[256,1024],3,2,conv=True)
        self.ident4 = self.block(1024,[256,1024],3,2)
        
        
        ##Classify
        self.classifier = nn.Sequential(
                                       nn.AvgPool2d(kernel_size=(4)),
                                       nn.Flatten(),
                                       nn.Linear(16384, num_classes))
        
    def forward(self,inputs):
        out = self.stg1(inputs)
        
        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)
        
        #stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        
        #stage4             
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        
        #Classify
        out = self.classifier(out)#100x1024
        out = F.softmax(out, dim=1)
        
        return out
    

    def conv_shortcut(self, in_channel, out_channel, stride):
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(stride, stride)),
                 nn.BatchNorm2d(out_channel)]
        return nn.Sequential(*layers)
    

    def block(self, in_channel, out_channel, k_size,stride, conv=False):
        layers = None
        first_layers = [nn.Conv2d(in_channel,out_channel[0], kernel_size=(1,1),stride=(1,1)),
                        nn.BatchNorm2d(out_channel[0]),
                        nn.ReLU(inplace=True)]
        if conv:
            first_layers[0].stride=(stride,stride)
        
        second_layers = [nn.Conv2d(out_channel[0], out_channel[1], kernel_size=(k_size, k_size), stride=(1,1), padding=1),
                        nn.BatchNorm2d(out_channel[1])]
        layers = first_layers + second_layers
        return nn.Sequential(*layers)



class NormalNet(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
    
    
class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()     
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.conv1=nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.model=model
    def forward(self, x):
        x=self.model(x)
        x=F.softmax(x, dim=1)
        return x



