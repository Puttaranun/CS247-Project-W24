import torch
import torch.nn as nn

class EEGNet_LSTM(nn.Module):
    def __init__(self, T, num_layers=1, hidden_size=32):
        super(EEGNet_LSTM, self).__init__()
        
        # Declare fixed parameters for EEGNet
        
        C = 22 #number of channels
        # T = 1000 #number of time points
        F1 = 8 #number of temporal filters
        D = 2 #depth multiplier
        F2 = 16 #number of pointwise filters
        kernel_len = 64
        dropout_rate = 0.25
        
        # Declare parameters for LSTM
        input_size = F2
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        dropout_lstm = 0.25

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
        self.LSTM1 = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_lstm, batch_first=True)
        self.BN_DO = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_lstm)
        )
        
        self.LSTM2 = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout_lstm, batch_first=True)

        # Flatten
        self.flatten = nn.Flatten()
        
        # Dense
        self.dense = nn.Linear(hidden_size*(T//32), 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        
        x = x.squeeze(dim=2)
        x = x.permute(0, 2, 1)
        
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # x, _ = self.LSTM1(x, (h0,c0))
        x, _ = self.LSTM1(x)
        
        x = x.permute(0,2,1)
        x = self.BN_DO(x)
            
        x = x.permute(0,2,1)
        # x, _ = self.LSTM2(x, (h0,c0))
        x, _ = self.LSTM2(x)
        
        x = x.permute(0,2,1)
        x = self.BN_DO(x)
        
        x = self.flatten(x)
        
        x = self.dense(x)
        x = self.softmax(x)
        
        return x