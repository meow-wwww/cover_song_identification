#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn


# In[2]:


def salience_to_output(temp, threshold=0):
    '''
    Suitable for multisample.
    temp: [N, 1, f, t] or [N, f, t]（已经softmax）
    [OUTPUT]:每个时间步只有一个激活的0/1矩阵
    计算损失函数的时候不用这个，要真实的输出时才会用这个函数
    '''
    
    out = torch.zeros_like(temp)
    sm = temp
    sm[sm < threshold] = 0
    # 到这里sm是部分为0，其余都小于等于1的矩阵

    _, maxi = sm.topk(1, dim=-2)

    index0 = torch.tensor([list(range(sm.shape[0]))]*sm.shape[-1]).T.reshape(-1)
    index1 = torch.zeros((sm.shape[0]*sm.shape[-1],), dtype=torch.int64)
    index2 = maxi.reshape(-1)
    index3 = torch.tensor([list(range(sm.shape[-1]))]*sm.shape[0]).reshape(-1)

    if len(temp.shape) == 4:
        sm[index0, index1, index2, index3] += 1
    elif len(temp.shape) == 3:
        sm[index0, index2, index3] += 1
        
    out = torch.zeros_like(sm)
    out[sm > 1] = 1
    
    return out


# In[4]:


def downsample(batch_data, num_floor):
    '''
    batch_data: a batch of y. [N, f, t]
    num_floor: 降采样到哪一层
    返回值一定是0/1的，且每个时间点最多只有一个1
    '''
    rst = batch_data
    with torch.no_grad():
        for floor in range(0, num_floor):
            rst = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)(rst)        
    rst = rst.bool().int()
    
    # 到这里rst一定是0/1的
    
    _, maxi = rst.topk(1, dim=1)
    
    rst_only = torch.zeros_like(rst)

    index0 = torch.tensor([list(range(rst.shape[0]))]*rst.shape[2]).T.reshape(-1)
    index1 = maxi.reshape(-1)
    index2 = torch.tensor([list(range(rst.shape[2]))]*rst.shape[0]).reshape(-1)
    
    rst[index0, index1, index2] += 1
    
    rst_only[rst==2] = 1
    
    return rst_only

