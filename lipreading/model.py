import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
from torchvision.models import densenet161, resnet152, vgg19

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
class EncoderDecoder(nn.Module):
    def __init__(self, network='vgg19'):
        super(EncoderDecoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'densenet161':
            self.net = densenet161(pretrained=True)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        else:
            self.net = vgg19(pretrained=True)
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 512
        self.temporalNet = TemporalConvNet(512 * 8 * 8, [1024, 512, 256], kernel_size= 3, dropout= 0.5)
        self.last_fc = nn.Linear(256, 38)
        self.dropout = nn.Dropout()
    def forward(self, xs):
        lstm_input = []
        for step_t in range(xs.shape[1]):
            x = xs[:,step_t]
            x = self.net(x)
            print (x.shape)
            x = x.view(x.size(0),  -1)
            print (x.shape)
            lstm_input.append(x)
        lstm_input = torch.stack(lstm_input, dim = 1)
        print (lstm_input.shape)
        out = self.temporalNet(lstm_input.transpose(1,2))
        print ('++++' , out.shape)
        out = out.transpose(1,2)
        preds = []
        for step_t in range(out.shape[1]):
            preds.append( self.last_fc(out[:,step_t]))
    
        return torch.stack(preds, dim = 1)

tcn = EncoderDecoder()
print (tcn)
from torch.autograd import Variable

x = Variable(torch.ones(5,4, 3, 128,128), requires_grad=True)
a = tcn(x)
print (a.shape)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

class LipNet(torch.nn.Module):
    def __init__(self, dropout_p=0.5):
        super(LipNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))     
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.gru1  = nn.GRU(96* 8 *8, 256, 1, bidirectional=True)
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        
        self.FC    = nn.Linear(512, 27+1)
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)        
        self.dropout3d = nn.Dropout3d(self.dropout_p)  
        self._init()
    
    def _init(self):
        weight_init(self.conv1)
        weight_init(self.conv2)
        weight_init(self.conv3)
        weight_init(self.FC)
        weight_init(self.gru1)
        weight_init(self.gru2)
        # init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        # init.constant_(self.conv1.bias, 0)
        
        # init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        # init.constant_(self.conv2.bias, 0)
        
        # init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        # init.constant_(self.conv3.bias, 0)        
        
        # init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        # init.constant_(self.FC.bias, 0)
        
        # for m in (self.gru1, self.gru2):
        #     stdv = math.sqrt(2 / (96 * 3* 6 + 256))
        #     for i in range(0, 256 * 3, 256):
        #         init.uniform_(m.weight_ih_l0[i: i + 256],
        #                     -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
        #         init.orthogonal_(m.weight_hh_l0[i: i + 256])
        #         init.constant_(m.bias_ih_l0[i: i + 256], 0)
        #         init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
        #                     -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
        #         init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
        #         init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool3(x)
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        h0 = torch.randn(1, x.shape[0] , 96 * 8 * 8 * 2)
        x, h = self.gru1(x)     
        x = self.dropout(x)
        x, h = self.gru2(x)   
        x = self.dropout(x)
        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()
        return x
        

