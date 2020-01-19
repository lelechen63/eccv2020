

import torch.nn as nn
import torch.nn.functional as F


class Cropped_VGG19(nn.Module):
    def __init__(self):
        super(Cropped_VGG19, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3,64,3)
        self.conv1_2 = nn.Conv2d(64,64,3)
        self.conv2_1 = nn.Conv2d(64,128,3)
        self.conv2_2 = nn.Conv2d(128,128,3)
        self.conv3_1 = nn.Conv2d(128,256,3)
        self.conv3_2 = nn.Conv2d(256,256,3)
        self.conv3_3 = nn.Conv2d(256,256,3)
        self.conv4_1 = nn.Conv2d(256,512,3)
        self.conv4_2 = nn.Conv2d(512,512,3)
        self.conv4_3 = nn.Conv2d(512,512,3)
        self.conv5_1 = nn.Conv2d(512,512,3)
        #self.conv5_2 = nn.Conv2d(512,512,3)
        #self.conv5_3 = nn.Conv2d(512,512,3)
        
    def forward(self, x):
        conv1_1_pad     = F.pad(x, (1, 1, 1, 1))
        conv1_1         = self.conv1_1(conv1_1_pad)
        relu1_1         = F.relu(conv1_1)
        conv1_2_pad     = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2         = self.conv1_2(conv1_2_pad)
        relu1_2         = F.relu(conv1_2)
        pool1_pad       = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1           = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad     = F.pad(pool1, (1, 1, 1, 1))
        conv2_1         = self.conv2_1(conv2_1_pad)
        relu2_1         = F.relu(conv2_1)
        conv2_2_pad     = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2         = self.conv2_2(conv2_2_pad)
        relu2_2         = F.relu(conv2_2)
        pool2_pad       = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2           = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad     = F.pad(pool2, (1, 1, 1, 1))
        conv3_1         = self.conv3_1(conv3_1_pad)
        relu3_1         = F.relu(conv3_1)
        conv3_2_pad     = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2         = self.conv3_2(conv3_2_pad)
        relu3_2         = F.relu(conv3_2)
        conv3_3_pad     = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3         = self.conv3_3(conv3_3_pad)
        relu3_3         = F.relu(conv3_3)
        pool3_pad       = F.pad(relu3_3, (0, 1, 0, 1), value=float('-inf'))
        pool3           = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad     = F.pad(pool3, (1, 1, 1, 1))
        conv4_1         = self.conv4_1(conv4_1_pad)
        relu4_1         = F.relu(conv4_1)
        conv4_2_pad     = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2         = self.conv4_2(conv4_2_pad)
        relu4_2         = F.relu(conv4_2)
        conv4_3_pad     = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3         = self.conv4_3(conv4_3_pad)
        relu4_3         = F.relu(conv4_3)
        pool4_pad       = F.pad(relu4_3, (0, 1, 0, 1), value=float('-inf'))
        pool4           = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv5_1_pad     = F.pad(pool4, (1, 1, 1, 1))
        conv5_1         = self.conv5_1(conv5_1_pad)
        
        return [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]
