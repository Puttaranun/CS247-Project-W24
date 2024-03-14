import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch import Tensor
from typing import Type
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from tqdm import tqdm
import random

cudnn.benchmark = True
plt.ion()   # interactive mode

# More info can be found at:
# https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
# Each Resnet layer has two Basic Blocks. Each Basic Block has two convolution layers.
class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: tuple = (0,1),
        dropout: float = 0.0
    ) -> None:
        super(BasicBlock, self).__init__() #call the constructor of the parent class


        # Since we are making Basic Blocks for all four different layers, each using a different
        # number of filters, we can have the user specify in out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=stride, padding=padding,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(p=dropout)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), stride=1, padding='same',bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out += self.shortcut(identity)
        out = self.elu(out)
        return  out
    
class ResNetLSTM(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        time_dim: int,
        block: Type[BasicBlock],
        num_classes: int  = 1000,
        dropout: float = 0.0
    ) -> None:
        super(ResNetLSTM, self).__init__()
        
        #self.in_channels = 48
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=48, kernel_size=(1,3), 
            stride=1, padding='same', bias=False) #48x22x1000
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(22,1), 
            stride=1, padding=0, bias=False) #48x1x1000
        self.bn2 = nn.BatchNorm2d(48)
        self.elu = nn.ELU(inplace=True)
        
        self.res1 = BasicBlock(in_channels=48,out_channels=48,stride=1,padding ='same',dropout=dropout)
        self.res2 = BasicBlock(in_channels=48,out_channels=48,stride=1,padding ='same',dropout=dropout) #48x1x400
        self.res3 = BasicBlock(in_channels=48,out_channels=96,stride=(1,2),padding=(0,1),dropout=dropout) #96x1x200
        self.res4 = BasicBlock(in_channels=96,out_channels=96,stride=1,padding=(0,1),dropout=dropout) #96x1x200
        self.res5 = BasicBlock(in_channels=96,out_channels=144,stride=(1,2),padding=(0,1),dropout=dropout) #144x1x100
        #self.res6 = BasicBlock(in_channels=144,out_channels=144,stride=1,padding='same',dropout=dropout) #144x1x100
        #self.res7 = BasicBlock(in_channels=144,out_channels=144,stride=(1,2),padding=(0,1),dropout=dropout) #144x1x50
        #self.res8 = BasicBlock(in_channels=144,out_channels=144,stride=1,padding='same',dropout=dropout) #144x1x50
        
        #self.res9 = BasicBlock(in_channels=144,out_channels=144,stride=(1,2),padding=(0,1),dropout=dropout) #144x1x25
        #self.res10 = BasicBlock(in_channels=144,out_channels=144,stride=1,padding='same',dropout=dropout) #144x1x25
        #self.res11 = BasicBlock(in_channels=144,out_channels=144,stride=(1,2),padding=(0,1),dropout=dropout) #144x1x13
        #self.res12 = BasicBlock(in_channels=144,out_channels=144,stride=1,padding='same',dropout=dropout) #144x1x13

        self.avgpool = nn.AvgPool2d(kernel_size=(1,10),stride=1)#96x1x91
        #self.maxpool = nn.MaxPool2d(kernel_size=(1,16))
        self.conv3 = nn.Conv2d(in_channels=144,out_channels=num_classes,kernel_size=(1,1),bias=False) #4x1x91

        self.fc1 = nn.Linear(int(num_classes*(time_dim/4 - 9)), 10)
        self.lstm = nn.LSTM(input_size=10, hidden_size=10, num_layers=1, dropout=dropout, batch_first=True)
        self.fc2 = nn.Linear(10, num_classes)
        #self.fc = nn.Linear(144, num_classes)
        

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        #x = self.res6(x)
        #x = self.res7(x)
        #x = self.res8(x)

        #x = self.res9(x)
        #x = self.res10(x)
        #x = self.res11(x)
        #x = self.res12(x)
        #x = self.res13(x)
        #x = self.res14(x)

        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        #x = self.maxpool(x)

        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        x = self.conv3(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x,_ = self.lstm(x)
        x = self.fc2(x)
        return x

def sample(X_train, y_train, bsz):
    sample_indices = random.sample(list(np.arange(X_train.shape[0])), bsz)
    X_train_sample = X_train[sample_indices]
    y_train_sample = y_train[sample_indices]
    return X_train_sample, y_train_sample



def train_data_prep(X,y,sub_sample,average,noise):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:800]
    print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
    total_X = X_max
    total_y = y
    print('Shape of X after maxpooling:',total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    
    print('Shape of X after subsampling and concatenating:',total_X.shape)
    print('Shape of Y:',total_y.shape)
    return total_X,total_y


def test_data_prep(X):
    
    total_X = None
    
    
    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:800]
    print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)
    
    
    total_X = X_max
    print('Shape of X after maxpooling:',total_X.shape)
    
    return total_X
