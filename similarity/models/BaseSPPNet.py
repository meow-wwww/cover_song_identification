# -*- coding: utf-8 -*-

import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
import math
import numpy as np

from torch.nn import init
from .basic_module import BasicModule
from .FCN import *

# wxy: 
class CQTSPPNet_seq_dilation_SPP(BasicModule):
    def __init__(self):
        super().__init__()
        self.features1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(12, 3),
                                dilation=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),

            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        #('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

        self.features2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),

            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        self.features3 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),

            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        self.features4 = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        self.features5 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None)))
        ]))

    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        
        # x1-x5的形状都是[N, c, f, t]
        # 下面是每个t的所有f做平均，重排，最后每个样本输出的深度序列都是二维的[c, t]，f压缩成一个数以后，通道变成了纵向的维度
        
        x1 = nn.AdaptiveMaxPool2d((1, None))(
            x1).squeeze(dim=2).permute(0, 2, 1)
        x2 = nn.AdaptiveMaxPool2d((1, None))(
            x2).squeeze(dim=2).permute(0, 2, 1)
        x3 = nn.AdaptiveMaxPool2d((1, None))(
            x3).squeeze(dim=2).permute(0, 2, 1)
        x4 = nn.AdaptiveMaxPool2d((1, None))(
            x4).squeeze(dim=2).permute(0, 2, 1)
        x5 = x5.squeeze(dim=2).permute(0, 2, 1)
        return x2, x3, x4, x5
    
    
class NeuralDTW_CNN_Mask_dilation_SPP6(BasicModule):
    """
    55000it [75:34:19,  4.73s/it]train_loss: 0.007741250745826789
    Youtube350:
                             0.9561629266311203 0.1928 1.876
    CoverSong80:, 513.07it/s]
                             0.902245744393522 0.0925 4.0875
    SH100K:
                             0.7396470873452758 0.49920356499478524 54.79321133971745
    *****************BEST*****************
    model name 0819_01:03:39.pth
    """
    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP()

        self.VGG_Conv1 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')

        self.fc = nn.Linear( 37376, 1)

    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqp.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self, seqa, seqb):
        seqa1, seqa2, seqa3, seqa4, = self.model(seqa)
        seqb1, seqb2, seqb3, seqb4 = self.model(seqb)
        # p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)
        # p_a = torch.cat((p_a2,p_a3,p_a4),3)
        # torch.Size([1, 1, 84, 400])
        # torch.Size([1, 194, 64])
        # torch.Size([1, 94, 128])
        # torch.Size([1, 44, 256])
        # torch.Size([1, 38, 512])
        VGG_out0 = self.VGG_Conv1(p_a1)
        VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x4'].shape[0], -1)
        VGG_out = torch.cat((VGG_out0,VGG_out1,VGG_out2, VGG_out3), 1)
        samil = torch.sigmoid(self.fc(VGG_out))

        return samil

    def forward(self, seqa, seqp, seqn):
        # seqa
        # seqp: 正例cqt
        # seqn: 负例cqt
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)