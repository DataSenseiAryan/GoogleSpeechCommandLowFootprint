import torch
from torch import nn
import torch.nn.functional as F
from subspectral_norm import SubSpectralNorm
from quaternion_layers import QuaternionConv2d
import torchaudio
import librosa 
import numpy
DROPOUT = 0.1



class NormalBlock(nn.Module):
    def __init__(self, n_chan: int, *, dilation: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()
        norm_layer = SubSpectralNorm(n_chan, 5) if use_subspectral else nn.BatchNorm2d(n_chan)
        self.f2 = nn.Sequential(
            QuaternionConv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=(3,1), padding="same",stride =1, groups=1),
            # nn.Conv2d(n_chan, n_chan, kernel_size=(3, 1), padding="same", groups=n_chan),
            norm_layer,
        )
        self.f1 = nn.Sequential(
            QuaternionConv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=(1,3), padding="same",stride =1, groups=1),
            # nn.Conv2d(n_chan, n_chan, kernel_size=(1, 3), padding="same", groups=n_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            # nn.Conv2d(n_chan, n_chan, kernel_size=1),
            QuaternionConv2d(in_channels=n_chan, out_channels=n_chan, kernel_size=(1,1), padding=0,stride =1, groups=1) ,
            nn.Dropout2d(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        n_freq = x.shape[2]
        x1 = self.f2(x)

        x2 = torch.mean(x1, dim=2, keepdim=True)
        x2 = self.f1(x2)
        x2 = x2.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1 + x2)


class TransitionBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, *, dilation: int = 1, stride: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        if stride == 1:
            # conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), groups=out_chan, padding="same")
            conv = QuaternionConv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=(3,1), padding="same", groups= 1, stride=(1,1))
            
        else:
            conv = QuaternionConv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=(3,1), stride= (stride,1) ,padding=(1,1), groups=1)            
            # conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), stride=(stride, 1), groups=out_chan, padding=(1, 0))

        norm_layer = SubSpectralNorm(out_chan, 5) if use_subspectral else nn.BatchNorm2d(out_chan)
        self.f2 = nn.Sequential(
            # nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            QuaternionConv2d(in_channels=in_chan, out_channels=out_chan,stride =(1,1), kernel_size=(1,1), padding=0, groups= 1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            conv,
            norm_layer,
        )

        self.f1 = nn.Sequential(
            # nn.Conv2d(out_chan, out_chan, kernel_size=(1, 3), padding="same", groups=out_chan, dilation=(1, dilation)),
            QuaternionConv2d(in_channels=out_chan, out_channels=out_chan,stride =(1,1), kernel_size=(1,3), padding="same",dilatation=(1,dilation), groups= 1),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            # nn.Conv2d(out_chan, out_chan, kernel_size=1),
            QuaternionConv2d(in_channels=out_chan, out_channels=out_chan,stride =(1,1), kernel_size=(1,1), padding=0, groups= 1),
            nn.Dropout2d(dropout)
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.f2(x)
        n_freq = x.shape[2]
        x1 = torch.mean(x, dim=2, keepdim=True)
        x1 = self.f1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1)


class QBcResNetModel(nn.Module):
    def __init__(self, n_class: int = 35, *, scale: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()
        
        self.instancenorm   = nn.InstanceNorm2d(40)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=40)

        
        
        self.input_conv = QuaternionConv2d(in_channels=4, out_channels=16*scale, stride =(2,1), kernel_size=(5,5), padding=2, groups= 1)
        # self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock(8*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t2 = TransitionBlock(8*scale, 12*scale, dilation=2, stride=2, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock(12*scale, dilation=2, dropout=dropout, use_subspectral=use_subspectral) 

        self.t3 = TransitionBlock(12*scale, 16*scale, dilation=4, stride=2, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)

        self.t4 = TransitionBlock(16*scale, 20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)

        # self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        
        self.dw_conv = QuaternionConv2d(in_channels=20*scale, out_channels=20*scale,stride =(1,1), kernel_size=(5,5), padding=0,dilatation=(1,1), groups= 1)
        # self.onexone_conv = nn.Conv2d(20*scale, 32*scale, kernel_size=1)
        
        self.onexone_conv = QuaternionConv2d(in_channels=20*scale, out_channels=32*scale,stride =(1,1), kernel_size=(1,1), padding=0,dilatation=(1,1), groups= 1)

        self.head_conv = nn.Conv2d(32*scale, n_class, kernel_size=1)
        # self.head_conv = QuaternionConv2d(in_channels=32*scale, out_channels=20*scale,stride =(1,1), kernel_size=(5,5), padding=0,dilation=(1,1), groups= 20)
    
    def forward(self, x: torch.Tensor):
        # print("begining", x.shape)
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
                # print("quaterion input shape",quaterion_input.shape)
                
        x = self.input_conv(quaterion_input)
        # print(" input conv ",x.shape)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=3, keepdim=True)
        x = self.head_conv(x)
        x = x.squeeze()
        # print(x.shape)
        # x = F.adaptive_avg_pool2d(x, output_size=35)
        # print(x.shape)
        

        return F.log_softmax(x, dim=-1)


# model = QBcResNetModel(n_class=35, scale=6,dropout=0.2,use_subspectral=False)
# print(model(torch.randn(32,1,16000)).shape)


# from torchinfo import summary
# print(summary(model, (32,4,40,32)))
#41,723 scale 2 
#Total params: 87,059 scale 3
# Total params: 148,723 scale 4
# Total params: 226,715 scale 5
# Total params: 321,035 scale 6
