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


from sklearn.utils import class_weight
import pandas as pd
import numpy as np

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
        x = self.conv1(x)
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
        
        
 
model = QResNet_182d(image_channels=4, num_classes=35)  #.to(device)


# print(model(torch.randn(4,4,512,33)).shape)
# summary(model, (4,4,40,32))
# Total params: 3,168,931
       

###############################################################################################################################

# batch_size = 256
# random_seed= 42



# log = Logger('/home/abrol/aryan/' , 'ResNet182DQmelspec3.txt')

# log.write(" ResNet18 with Quaterionic features augmentations but Mel feature 512,160,400 only and their dervatives, hann window, log_softmax, nll loss, adamW , cosine annealing")

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# print(device)


# train_df = pd.read_csv('/home/abrol/aryan/train_metadata_speeechcommand.csv')
# class_weights = torch.FloatTensor(class_weight.compute_class_weight(
#     class_weight='balanced',classes = np.unique(train_df['NumLabel']),y= train_df['NumLabel'])).to(torch.device(device))


# batch_size = 128

# if device == "cuda:2":
#     num_workers = 1
#     pin_memory = True
# else:
#     num_workers = 0
#     pin_memory = False


# # composed_transforms = ComposeTransform( 
# #         [
# #     Soxtranforms(),
# #    # RandomBackgroundNoise(16000, '/home/abrol/aryan/Dataset/_background_noise_')
# #         ]
# #     )

# from custom_dataloader_mel import CustomDataloader #ye previous scripts ke lie tha
# # from custom_dataloader_mel_spec import CustomDataloader # this is augmentation on spctrogram

# train_dataset = CustomDataloader( '/home/abrol/aryan/train_metadata_speeechcommand.csv', root_dir='/',isquaterionic=True,conv1dcompatible=False,transform=None)

# test_dataset = CustomDataloader( '/home/abrol/aryan/test_metadata_speeechcommand.csv', root_dir='/', isquaterionic=True,conv1dcompatible=False,transform= None)




# train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers= num_workers)
# test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True, num_workers= num_workers)

# model = QResNet_182d(image_channels=4, num_classes=35).to(device)


# # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
# # optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
# # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # reduce the learning after 10 epochs by a factor of 10
# # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=4)
# # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# # criterion = nn.CrossEntropyLoss(weight=class_weights)



# optimizer = optim.AdamW(model.parameters(),  lr=0.0002, betas=(0.5, 0.999))
# # optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
# # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# # scheduler = ExponentialLR(optimizer, gamma=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

# # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
# # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# criterion = nn.NLLLoss()





# def test():
#     correct = 0
#     total = 0
#     for data in test_loader:
#         signal, labels = data['Audio'].to(device),data['label'].to(device)
#         outputs = model(signal) #.squeeze()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()

#     log.write('Accuracy of the network on the test set: %d %%' % (
#         100 * correct / total))
#     print('Accuracy of the network on the test set: %d %%' % (
#         100 * correct / total))


# def train():    
#             start_time = time.time()
#             running_loss=0
#             running_loss_10_batch =0.0
#             correct=0
#             total=0

#             torch.autograd.set_detect_anomaly(True)
#             # log.write(torch.autograd.set_detect_anomaly(True))
#             # scheduler.step()
#             for i, data in enumerate(train_loader, 0):
#                 # print("data dtype",data['Audio'].dtype)
#                 inputs, labels = data['Audio'].to(device) ,data['label'].to(device)
#                 # print("inputs dtype",inputs.dtype)

#                 #print("input dtype",inputs.dtype)
#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward + backward + optimize
#                 outputs = model(inputs)
#                 # outputs = outputs.squeeze()
#                 # print("input dtype",outputs.dtype)

#                 # print(outputs.shape,labels.shape)
#                 # print("output dtype", outputs.dtype)

#                 loss = criterion(outputs, labels)
#                 loss.backward()
                
#                 #sprint("backward pass done for minibatch in epoch ", i , epoch)
#                 optimizer.step()

#                 # print statistics
#                 running_loss += loss.item()
#                 running_loss_10_batch += loss.item()
                
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
                
#                 if i % 10==0:    # print every 10 mini-batches
#                     log.write(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss_10_batch / 10:.3f}')
#                     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss_10_batch / 10:.3f}')
#                     running_loss_10_batch = 0.0
            
            
#             train_loss=running_loss/len(train_loader)
#             # scheduler.step(metrics=train_loss)
#             scheduler.step()
#             accu=100.*correct/total
#             log.write('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu), f"Time Taken {round((time.time()-start_time)/3600, 4)}")
#             print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu), f"Time Taken {round((time.time()-start_time)/3600, 4)}")
            
#             torch.save(model.state_dict(), os.path.join("/home/abrol/aryan/savedModels", f"ResNet182DQmelspec3{epoch}.pt"))
            
            
#             test()
    

 
 
# for epoch in range(100):  # loop over the dataset multiple times
#     train()
            

# log.write('Finished Training')
# print('Finished Training')







################################################################################################################################