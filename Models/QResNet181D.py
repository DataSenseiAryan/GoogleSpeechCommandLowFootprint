import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torch
import time
from torchinfo import summary
from tqdm import tqdm


from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

################################################################################################################################
import torchaudio

from quaternion_layers import QuaternionConv

class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = QuaternionConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = QuaternionConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity 
        x = self.relu(x)
        return x
    

class QResNet_181d(nn.Module):
    
    def __init__(self, image_channels, num_classes):
        
        super(QResNet_181d, self).__init__()
        
        self.in_channels = 64
        self.conv1 = QuaternionConv(image_channels, 64, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        # print(x.shape) # 0,1,2,3
        x= x.view(x.shape[0],4*x.shape[2],x.shape[3])
        # print(x.shape)
        
        x = self.conv1(x)
        # print("after conv1",x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print("after maxpool",x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        #x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            QuaternionConv(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm1d(out_channels)
        )
        
        
 
# model = QResNet_181d(image_channels=160, num_classes=35) #.to(device)

# print(model(torch.randn(32,4,40,32)))
# summary(model, (32,4,40,32))
# Total params: 3,168,931
# Total params: 1,306,211 conv1d