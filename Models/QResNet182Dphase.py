import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torch
import time
import librosa
from torchinfo import summary
from tqdm import tqdm


from sklearn.utils import class_weight
import pandas
import numpy

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

################################################################################################################################
import torchaudio

from quaternion_layers import QuaternionLinear, QuaternionConv2d

class Block (nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = QuaternionConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = QuaternionConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
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
    

class QResNet_182d(nn.Module):
    
    def __init__(self, image_channels, num_classes):
        
        super(QResNet_182d, self).__init__()
        
        self.in_channels = 64
        self.conv1 = QuaternionConv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.instancenorm   = nn.InstanceNorm2d(40)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=40)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1))
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
        
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                freqs, times, mags = librosa.reassigned_spectrogram(x.numpy(), sr=16000, S=None,n_fft=512, hop_length=160, win_length=400, window='hann', center=True, reassign_frequencies=True, reassign_times=True, ref_power=1e-06, fill_nan=False,clip=True, dtype=None, pad_mode='constant')
                mags= librosa.amplitude_to_db(mags, ref=numpy.max)
                
                freqs = torch.tensor(freqs.astype('float32')).squeeze() #convert into tensor 
                mags = torch.tensor(mags.astype('float32')).squeeze()
                # print("freqs", freqs.shape)
                # print("mags", mags.shape)
                
                # y  = self.torchfb(x)+1e-6
                # y  = y.log().squeeze()
                # print("y mel", y.shape)
                # y_ = torchaudio.functional.compute_deltas(y)
                # print("y_ mel der", y_.shape)
                
                # y  = self.instancenorm(y) #.squeeze(1).detach()
                # print("y instance norm mel", y.shape)
                # y_ = self.instancenorm(y_) #.unsqueeze(1).detach()
                # print("y_ instance norm mel", y_.shape)
                freqs_ = torchaudio.functional.compute_deltas(freqs)
                mags_ = torchaudio.functional.compute_deltas(freqs)
                
                # normalise everyhng
                freqs  = self.instancenorm(freqs).unsqueeze(1).detach()
                freqs_ = self.instancenorm(freqs_).unsqueeze(1).detach()
                mags  = self.instancenorm(mags).unsqueeze(1).detach()
                mags_ = self.instancenorm(mags_).unsqueeze(1).detach()
                
                # print("freqs after instance norm", freqs.shape,freqs_.shape, mags.shape,mags_.shape)
                
                quaterion_input = torch.cat([mags,freqs,mags_,freqs_], dim=1)
        
        x = self.conv1(quaterion_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
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
            QuaternionConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )
        
        
 
# model = QResNet_182d(image_channels=4, num_classes=35)  #.to(device)

# print(model(torch.randn(32,1,16000)))