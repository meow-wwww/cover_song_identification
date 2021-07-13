#!/usr/bin/env python
# coding: utf-8

# # U-Net

# In[3]:


import numpy as np
import torch
import torch.nn as nn


# In[4]:


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