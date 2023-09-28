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

'''
Mathboxnet Non Modular 
author @aryanchaudhary crosscaps Labs
Date @Feb 16 2023 00:30
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchaudio

###############################################################################################################################


class QMatchBoxNet85k2Dcompatible(nn.Module):    
    def __init__(self, preprocessing_mode:str) -> None:
        ''' mode ['cnn', 'audio' , 'off']'''
        super().__init__()
        
        self.preprocessing_mode = preprocessing_mode
        
        self.audioToMFCC = nn.Sequential(
            
            torchaudio.transforms.MelSpectrogram( sample_rate = 16240, n_fft = 512, win_length = 400, hop_length = 160,
                                            n_mels = 64),
            torchaudio.transforms.AmplitudeToDB('power')
        )
        
        
        self.preprocessing_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=(1,), stride=(1,), padding =0, bias = False)
        )
        
        
        
        self.QJasperBlock0 = nn.Sequential(
            QuaternionConv(in_channels=160, out_channels=64, kernel_size=5, stride=(1,), padding=2, groups =1 , bias= False),
            nn.BatchNorm1d(num_features=64,eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace=False)                                                                          
        )
        
        
        
        
        ###########################################################
        self.QJasperBlock1_mconv = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=13, stride=1, padding=5, groups =1 , bias= False),
            
            nn.BatchNorm1d(num_features=64,eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
        )
        
        self.QJasperBlock1_res = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    
        )
        self.QJasperBlock1_mout = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        
        #########################################################
        self.QJasperBlock2_mconv = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=15, stride=1, padding=7, groups =1 , bias= False),
            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(1,), stride= (1,), bias=False),
            nn.BatchNorm1d(num_features=64,eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
        )
        self.QJasperBlock2_res = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    
        )
        self.QJasperBlock2_mout = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        
        #######################################################
        self.QJasperBlock3_mconv = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=17, stride=1, padding=8, groups =1, bias= False),
            # nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(1,), stride= (1,), bias=False),
            nn.BatchNorm1d(num_features=64,eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
        )
        self.QJasperBlock3_res = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    
        )
        self.QJasperBlock3_mout = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        
        #######################################################
        self.QJasperBlock4 = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=96, kernel_size=11, stride=1, padding=10, dilatation=2, groups=1, bias= False),
            # nn.Conv1d(in_channels=64, out_channels=128, kernel_size=(1,), stride= (1,), bias=False),
            nn.BatchNorm1d(num_features=96, eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        ######################################################
        self.QJasperBlock5 = nn.Sequential(
            QuaternionConv(in_channels=96, out_channels=96, kernel_size=1, stride=1, bias= False),
            nn.BatchNorm1d(num_features=96, eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        #####################################################
        self.adaptivepool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=96, out_features=35)
            )
        self.softmax = nn.Softmax(dim=1)
        
        
        
    def forward(self,x):
        
        
        
        if self.preprocessing_mode == "audio":
            o = self.audioToMFCC(x)
            o = torch.squeeze(o) # MFCC one extra dimension of "1". we are just squeezing it out!
            # o = torch.cat([o,o,o,o], dim=1)
            # print("output shape Audio" ,o.shape)
        elif self.preprocessing_mode =="cnn":
            o = self.preprocessing_cnn(x)
            # print("hi")
            
        else:
            # print(x.shape)
            
            o=x.view(x.shape[0],4*x.shape[2],x.shape[3]) # since you are choosing off make sure jasper block 0 takes 4 as input channel
            # print("hello")
        o = self.QJasperBlock0(o)
        # print("output shape Jasperbloack0" ,o.shape)
        
        ox =o #saving o as it will be overwritten in next layer and we need it as residue. Hence ox is residue for next layer
        
        #####################################################
        o = self.QJasperBlock1_mconv(o)
        # print("output shape Jasperbloack1_mconv" ,o.shape)
        o_res = self.QJasperBlock1_res(ox) # residual connection  ox will have original sequcence length 
        o_pad = F.pad(o, (0, o_res.shape[-1] - o.shape[-1]))
        o_pad_res = o_res + o_pad # adding output of conv and residual connection
        o = self.QJasperBlock1_mout(o_pad_res)
        
        ox = o #saving o as it will be overwritten in next layer and we need it as residue. Hence ox is residue for next layer      
        
        ####################################################
        
        o = self.QJasperBlock2_mconv(o)
        o_res = self.QJasperBlock2_res(ox) # residual connection  ox will have original sequcence length 
        o_pad = F.pad(o, (0, o_res.shape[-1] - o.shape[-1]))
        o_pad_res = o_res + o_pad # adding output of conv and residual connection
        o = self.QJasperBlock2_mout(o_pad_res)       
        ox = o #saving o as it will be overwritten in next layer and we need it as residue. Hence ox is residue for next layer      
        # print("output shape Jasperbloack2" ,o.shape)
        
        ####################################################
        
        
        o = self.QJasperBlock3_mconv(o)
        o_res = self.QJasperBlock3_res(ox) # residual connection  ox will have original sequcence length 
        o_pad = F.pad(o, (0, o_res.shape[-1] - o.shape[-1]))
        o_pad_res = o_res + o_pad # adding output of conv and residual connection
        o = self.QJasperBlock3_mout(o_pad_res)       
        # print("output shape Jasperbloack3" ,o.shape)
        ####################################################
        
        o = self.QJasperBlock4(o)
        # print("output shape Jasperbloack4" ,o.shape)
        
        ####################################################
        
        o = self.QJasperBlock5(o)
        # print("output shape Jasperbloack5" ,o.shape)
        
        
        ####################################################
        o = self.adaptivepool(o)
        # print("output shape adaptive pooling" ,o.shape)
        o = torch.flatten(o, 1) # Flatten so that i can feed it to Linear net else their would be an extra dimension 
        
        # print("output shape Falttening Adaptive pooling" ,o.shape)
        o = self.classifier(o)
        # o = self.softmax(o)
        
        return F.log_softmax(o)
        
        
        
    
    
    
# m = QMatchBoxNet85k2Dcompatible(preprocessing_mode='off')
# print(m(torch.randn(32,4,40,32)).shape)
# print(summary(m, (32,4,40,32)))
    
  

