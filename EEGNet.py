import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, T):
        super(EEGNet, self).__init__()
        
        # Declare fixed parameters for EEGNet
        C = 22 #number of channels
        T = 1000 #number of time points
        F1 = 4 #number of temporal filters
        D = 2 #depth multiplier
        F2 = 16 #number of pointwise filters
        kernel_len = 64
        dropout_rate = 0.25

        # First block: Conv2D - BN - DWConv2D - BN - ELU - AvgPool - Dropout
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_len), bias=False, padding='same'),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, kernel_size=(C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(dropout_rate)
        )
        
        # Second block: SeparableConv2D - BN - ELU - AvgPool - Dropout
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(F1*D, F2, kernel_size=(1,16), padding='same', bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(dropout_rate)
        )
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Dense
        self.dense = nn.Linear(F2*(T//32), 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.flatten(x)
        print(x.shape)
        x = self.dense(x)
#         x = nn.Linear(x.shape[-1], 4)(x)
        x = self.softmax(x)
        
        return x