#!/usr/bin/env python
# coding: utf-8

# # U-Net

# In[3]:


import numpy as np
import torch
import torch.nn as nn

from model_unet_subnet import *



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.channel_layers = [64, 128, 256, 512]
        
        self.MaxPool = nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True)
        
        self.DownConv0 = down_conv(6, 64)
        self.DownConv1 = down_conv(64, 128)
        self.DownConv2 = down_conv(128, 256)
        self.DownConv3 = down_conv(256, 512)
        
        self.GetOut = [out_conv(self.channel_layers[i]*2) for i in range(0,3)] + [out_conv(self.channel_layers[3])]
        # 64*2->1 128*2->1 256*2->1 512->1
        
        self.Up_T_Conv = ['placeholder'] + [up_T_conv(self.channel_layers[i]*2, self.channel_layers[i-1]) for i in range(1,3)]+[up_T_conv(self.channel_layers[3], self.channel_layers[3-1])]
        # 'placeholder' 128*2->64 256*2->128 512->256
        
        self.Up_Conv = [up_conv(self.channel_layers[i]*2) for i in range(3)]
        # 64*2 128*2 256*2
        
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
        
        #print('down finish', x.shape)
        #print('save_for_concat:', [array.shape for array in save_for_concat])
        
        for floor in range(3,-1,-1): # floor: 3 2 1 0
            #print(f'[floor = {floor}]')
            if out_floor == floor:
                x = self.GetOut[floor](x)
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



# In[28]:


# In[ ]:




