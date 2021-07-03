#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import torch
import torch.nn as nn


# def salience_to_output(temp, threshold):
#     out = np.zeros_like(temp)
#     sm = nn.functional.softmax(torch.tensor(temp), dim=0)
#     sm[sm < threshold] = 0
#     idx = sm.argmax(axis=0)
#     for time, frq in enumerate(idx):
#         out[frq, time] = 1
#     return out

# In[ ]:


def salience_to_output(temp, threshold=0):
    '''
    Suitable for multisample.
    temp: [N, c, f, t]
    '''
    out = torch.zeros_like(temp)
    sm = nn.functional.softmax(temp, dim=2)    
    sm[sm < threshold] = 0
    indices = sm.max(axis=2).indices
    for n in range(temp.shape[0]):
        out[n,0].T.scatter_(dim=1, index=indices[n,0].reshape(33,-1), value=1)
    return out


# In[ ]:


def downsample(batch_data, num_floor):
    rst = batch_data
    with torch.no_grad():
        for floor in range(0, num_floor):
            rst = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)(rst)
    return rst


# In[19]:

'''
try:
    get_ipython().system('jupyter nbconvert --to python utils.ipynb')
except:
    pass
'''

# In[ ]:




