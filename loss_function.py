#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as func


# In[ ]:


'''
原始函数，CrossEntropyLoss_Origin的内核
def cross_entropy_multilayer_simplify(sm, one_hot): 
    # sm one_hot都是多片的
    # 论文原始版本
        # 可以优化的方向：对unvoicing的部分加以控制：平方误差or dual loss?
    assert(sm.shape == one_hot.shape)
    batch_size, case, _ = sm.shape
    print(batch_size, case)
    none_zero_lines = one_hot.bool().any(len(one_hot.shape)-1)
    # print(none_zero_lines)
    return (-torch.log(sm)*one_hot).sum()/none_zero_lines.sum()
'''


# In[10]:


class CrossEntropyLoss_Origin(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_Origin, self).__init__()
        return
    def forward(self, output, one_hot):
        '''
        output: network output(not softmax yet)
                shape: [N, 1, f, t]
        one_hot: ground truth(has not been gaussian blurred)
                shape: [N, f, t]
        [OUTPUT]是一个batch中不同样本的均值
        '''
        output = output.squeeze(dim=1) # output: [N, f, t]
        
        output -= output.max() # 防止softmax溢出
        sm = func.softmax(output, dim=-2)

        none_zero_lines = one_hot.bool().any(1)
        
        return (-torch.log(sm)*one_hot).sum()/none_zero_lines.sum()

