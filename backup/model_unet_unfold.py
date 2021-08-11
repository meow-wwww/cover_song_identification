#!/usr/bin/env python
# coding: utf-8

# # U-Net unfinished!

# In[3]:


import numpy as np
import torch
import torch.nn as nn


# In[22]:


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.MaxPool = nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True)
        
        self.DownConv0 = nn.Sequential(
            nn.BatchNorm2d(6),
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.DownConv1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.DownConv2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.DownConv3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 64*2->1 128*2->1 256*2->1 512->1
        self.GetOut0 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.GetOut1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.GetOut2 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.GetOut3 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 'placeholder' 128*2->64 256*2->128 512->256
        self.Up_T_Conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=0) # (1,1))
        )
        self.Up_T_Conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=512),
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(2,2), stride=(2,2), padding=0) # (1,1))
        )
        self.Up_T_Conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=512),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=(2,2), padding=0) # (1,1))
        )
        
        # 64*2 128*2 256*2
        self.Up_Conv0 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.Up_Conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.Up_Conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.SoftmaxLayer = nn.Softmax(dim=2)
        
    def forward(self, x, out_floor):
        save_for_concat = []
        x = self.DownConv0(x)
        #print(x.shape)
        save_for_concat.append(x)
        x = self.MaxPool(x)
        x = self.DownConv1(x)
        #print(x.shape)
        save_for_concat.append(x)
        x = self.MaxPool(x)
        x = self.DownConv2(x)
        #print(x.shape)
        save_for_concat.append(x)
        x = self.MaxPool(x)
        x = self.DownConv3(x)
        
        
        for floor in range(3,-1,-1): # floor: 3 2 1 0
            #print(f'[floor = {floor}]')
            if out_floor == floor:
                x = self.GetOut[floor](x)
                x = self.SoftmaxLayer(x)
                return x
            else:
                x = self.Up_T_Conv[floor](x)
                if x.shape[-1] != save_for_concat[floor-1].shape[-1]:
                    if x.shape[-1]-1 == save_for_concat[floor-1].shape[-1]:
                        x = x[:,:,:,:-1]
                    assert(x.shape[-1] == save_for_concat[floor-1].shape[-1])
                    #print(f'after cut, x.shape = {x.shape}')
                #else:
                    #print(f'no need to cut, x.shape = {x.shape}')
                x = self.Up_Conv[floor-1](torch.cat((save_for_concat[floor-1], x), dim=1)) # 可能有维数问题
                #print(x.shape)
