import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchaudio
from quaternion_layers import QuaternionConv
###############################################################################################################################



class QMatchBoxNetBB(nn.Module):
    
    def __init__(self, preprocessing_mode:str) -> None:
        ''' mode ['audio','cnn', 'off]'''
        super().__init__()
        
        self.preprocessing_mode = preprocessing_mode
        
        self.audioToMFCC = nn.Sequential(
           
            torchaudio.transforms.MelSpectrogram( sample_rate = 16240, n_fft = 512, win_length = 400, hop_length = 160,
                                            n_mels = 64),
            torchaudio.transforms.AmplitudeToDB('power')
        )
        
        self.preprocessing_cnn = nn.Sequential(
            QuaternionConv(in_channels=40, out_channels=64, kernel_size=1, stride=1, padding =0, bias = False)
        )
        
        
        
        self.JasperBlock0 = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=11, stride=1, padding=5, groups =1 , bias= False),
            QuaternionConv(in_channels=64, out_channels=128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=128,eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace=False)
        )
        
        
        
        
        ###########################################################
        self.JasperBlock1_mconv = nn.Sequential(
            QuaternionConv(in_channels=128, out_channels=128, kernel_size=13, stride=1, padding=5, groups =1 , bias= False),
            QuaternionConv(in_channels=128, out_channels=64, kernel_size=1, stride= 1, bias=False),
            nn.BatchNorm1d(num_features=64,eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
        )
        
        self.JasperBlock1_res = nn.Sequential(
            QuaternionConv(in_channels=128, out_channels=64, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    
        )
        self.JasperBlock1_mout = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        
        #########################################################
        self.JasperBlock2_mconv = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=15, stride=1, padding=7, groups =1 , bias= False),
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=1, stride= 1, bias=False),
            nn.BatchNorm1d(num_features=64,eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
        )
        self.JasperBlock2_res = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    
        )
        self.JasperBlock2_mout = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        
        #######################################################
        self.JasperBlock3_mconv = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=17, stride=1, padding=8, groups =1 , bias= False),
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=1, stride= 1, bias=False),
            nn.BatchNorm1d(num_features=64,eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
        )
        self.JasperBlock3_res = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias = False),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                    
        )
        self.JasperBlock3_mout = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        
        #######################################################
        self.JasperBlock4 = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=29, stride=1, padding=28,dilatation=2, groups=1, bias= False),
            QuaternionConv(in_channels=64, out_channels=128, kernel_size=1, stride= 1, bias=False),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        ######################################################
        self.JasperBlock5 = nn.Sequential(
            QuaternionConv(in_channels=128, out_channels=128, kernel_size=1, stride=1, bias= False),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.1,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p =0.0, inplace =False)
        )
        
        
        
        #####################################################
        self.adaptivepool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=35)
            )
        # self.softmax = nn.Softmax(dim=1)
        
        
        
    def forward(self,x):
        if self.preprocessing_mode == "audio":
            o = self.audioToMFCC(x)
            # print("MFCC output:", o.shape)
            o = torch.squeeze(o) # MFCC one extra dimension of "1". we are just squeezing it out!
            print("MFCC output:", o.shape)
            
        elif self.preprocessing_mode == "cnn":
             o = x.squeeze()
             o = self.preprocessing_cnn(o)
        # else:
        #     o = x.squeeze()
        
        o = self.JasperBlock0(o)
        # print("output shape Jasperbloack0" ,o.shape)
        
        ox =o #saving o as it will be overwritten in next layer and we need it as residue. Hence ox is residue for next layer
        
        #####################################################
        o = self.JasperBlock1_mconv(o)
        # print("output shape Jasperbloack1_mconv" ,o.shape)
        o_res = self.JasperBlock1_res(ox) # residual connection  ox will have original sequcence length 
        o_pad = F.pad(o, (0, o_res.shape[-1] - o.shape[-1]))
        o_pad_res = o_res + o_pad # adding output of conv and residual connection
        o = self.JasperBlock1_mout(o_pad_res)
        
        ox = o #saving o as it will be overwritten in next layer and we need it as residue. Hence ox is residue for next layer      
        
        ####################################################
        
        o = self.JasperBlock2_mconv(o)
        o_res = self.JasperBlock2_res(ox) # residual connection  ox will have original sequcence length 
        o_pad = F.pad(o, (0, o_res.shape[-1] - o.shape[-1]))
        o_pad_res = o_res + o_pad # adding output of conv and residual connection
        o = self.JasperBlock2_mout(o_pad_res)       
        ox = o #saving o as it will be overwritten in next layer and we need it as residue. Hence ox is residue for next layer      
        # print("output shape Jasperbloack2" ,o.shape)
        
        ####################################################
        
        
        o = self.JasperBlock3_mconv(o)
        o_res = self.JasperBlock3_res(ox) # residual connection  ox will have original sequcence length 
        o_pad = F.pad(o, (0, o_res.shape[-1] - o.shape[-1]))
        o_pad_res = o_res + o_pad # adding output of conv and residual connection
        o = self.JasperBlock3_mout(o_pad_res)       
        # print("output shape Jasperbloack3" ,o.shape)
        ####################################################
        
        o = self.JasperBlock4(o)
        # print("output shape Jasperbloack4" ,o.shape)
        
        ####################################################
        
        o = self.JasperBlock5(o)
        # print("output shape Jasperbloack5" ,o.shape)
        
        
        ####################################################
        o = self.adaptivepool(o)
        # print("output shape adaptive pooling" ,o.shape)
        o = torch.flatten(o, 1) # Flatten so that i can feed it to Linear net else their would be an extra dimension 
        
        # print("output shape Falttening Adaptive pooling" ,o.shape)
        o = self.classifier(o)
        #o = self.softmax(o)
        
        return F.log_softmax(o)
        
        
        
    
    
    
# m = QMatchBoxNetBB(preprocessing_mode='cnn')



# print(m (torch.randn(4,1,40,32)).shape)
# Total params: 150,051
# summary(m, (4,1,40,32))
