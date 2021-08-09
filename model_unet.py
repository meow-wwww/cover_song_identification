#!/usr/bin/env python
# coding: utf-8

# # U-Net

# In[2]:


import numpy as np
import torch
import torch.nn as nn


# In[ ]:


class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class out_conv(nn.Module):
    def __init__(self, ch_in):
        super(out_conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(in_channels=ch_in, out_channels=1, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class up_T_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_T_conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_features=ch_in),
            nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(2,2), stride=(2,2), padding=0) # (1,1))
            # TODO: ConvTranspose2d的padding怎么设置??
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    

class up_conv(nn.Module):
    def __init__(self, ch_in):
        super(up_conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_features=ch_in),
            nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=3, stride=1, padding=(1,1), bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


# In[4]:


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.channel_layers = [64, 128, 256, 512]
        
        self.MaxPool = nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True)
        
        self.DownConv0 = down_conv(6, 64)
        self.DownConv1 = down_conv(64, 128)
        self.DownConv2 = down_conv(128, 256)
        self.DownConv3 = down_conv(256, 512)
        
        self.GetOut = nn.ModuleList([out_conv(self.channel_layers[i]*2) for i in range(0,3)] + [out_conv(self.channel_layers[3])])
        # 64*2->1 128*2->1 256*2->1 512->1
        
        self.Up_T_Conv = nn.ModuleList([up_T_conv(self.channel_layers[i]*2, self.channel_layers[i-1]) for i in range(1,3)]+[up_T_conv(self.channel_layers[3], self.channel_layers[3-1])])
        # 'placeholder' 128*2->64 256*2->128 512->256
        
        self.Up_Conv = nn.ModuleList([up_conv(self.channel_layers[i]*2) for i in range(3)])
        # 64*2 128*2 256*2
        
        self.SoftmaxLayer = nn.Softmax(dim=2)
        
    def forward(self, x, out_floor):
        save_for_concat = []
        x = self.DownConv0(x)
        save_for_concat.append(x)
        x = self.MaxPool(x)
        x = self.DownConv1(x)
        save_for_concat.append(x)
        x = self.MaxPool(x)
        x = self.DownConv2(x)
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
                x = self.Up_T_Conv[floor-1](x)
                if x.shape[-1] != save_for_concat[floor-1].shape[-1]:
                    if x.shape[-1]-1 == save_for_concat[floor-1].shape[-1]:
                        x = x[:,:,:,:-1]
                    assert(x.shape[-1] == save_for_concat[floor-1].shape[-1])
                    #print(f'after cut, x.shape = {x.shape}')
                #else:
                    #print(f'no need to cut, x.shape = {x.shape}')
                x = self.Up_Conv[floor-1](torch.cat((save_for_concat[floor-1], x), dim=1)) # 可能有维数问题
                #print(x.shape)


# In[5]:


'''class UNet(nn.Module):
    def __init__(self, UNet_core):
        super(UNet, self).__init__()
        self.l1 = nn.Sequential(*list(UNet_core.children())[:-1]).to('cuda:0')
        self.last = list(UNet_core.children())[-1]

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size()[0], -1)
        x = self.last(x)
        return x'''

