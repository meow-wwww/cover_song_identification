#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as func


# In[ ]:


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
        
        sm = output + 1e-20

        none_zero_lines_num = one_hot.bool().any(1).sum()
        
        if none_zero_lines_num == 0: # 在这里曾经因为除以0发生了一次NaN (╥﹏╥)
            return (-torch.log(sm)*one_hot).sum()*0 # 其实直接return一个0tensor也行，但考虑到device，还是这样写吧
        else:
            return (-torch.log(sm)*one_hot).sum()/none_zero_lines_num


# In[ ]:


class CrossEntropyLoss_for_FA_CE(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_for_FA_CE, self).__init__()
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
        sm = output + 1e-20
        
        # 对one-hot进行处理，没有label的全零列改成1/360

        none_zero_lines = one_hot.bool().any(1).reshape(-1)
        one_hot = one_hot.float()
        index0 = torch.tensor([list(range(one_hot.shape[0]))]*one_hot.shape[2]).T.reshape(-1)[none_zero_lines == False]
        index2 = torch.tensor([list(range(one_hot.shape[2]))]*one_hot.shape[0]).reshape(-1)[none_zero_lines == False]
        one_hot[index0, :, index2] = 1/one_hot.shape[1]
        
        return (-torch.log(sm)*one_hot).sum()/(output.shape[2]*output.shape[0])


# In[ ]:


class CrossEntropyLoss_for_FA_CESQ(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_for_FA_CESQ, self).__init__()
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
        sm = output + 1e-20
        
        ce_loss = (-torch.log(sm)*one_hot).sum(dim=1)
        sq_loss = (sm**2).sum(dim=1)
        
        non_zero = one_hot.bool().any(1).to(torch.int32)
        loss_all = ((1-non_zero)*sq_loss).sum()+ce_loss.sum()
        
        return loss_all/(output.shape[0]*output.shape[2])


# In[ ]:


class CrossEntropyLoss_for_FA_CESQ_TF(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_for_FA_CESQ_TF, self).__init__()
        return
    def forward(self, output, one_hot, threshold):
        '''
        output: network output(not softmax yet)
                shape: [N, 1, f, t]
        one_hot: ground truth(has not been gaussian blurred)
                shape: [N, f, t]
        [OUTPUT]是一个batch中不同样本的均值
        '''
        
        output = output.squeeze(dim=1) # output: [N, f, t]
        sm = output + 1e-20
        
        ce_loss = (-torch.log(sm)*one_hot).sum(dim=1)
        sq_loss = (sm**2).sum(dim=1)
        non_zero = one_hot.bool().any(1).to(torch.int32)
        loss_all = ((1-non_zero)*sq_loss) + ce_loss
        
        Xout = utils.salience_to_output(output.clone().detach(), threshold=threshold).to(torch.int32)
        cmp_time = (Xout != one_hot).to(torch.int32).sum(dim=1).bool().to(torch.int32) # 判断错误的列

        False_loss = (loss_all * cmp_time).sum()
        True_loss = loss_all.sum() - False_loss
        False_num = cmp_time.sum()
        True_num = output.shape[2]*output.shape[0] - False_num
        
        return False_loss, False_num, True_loss, True_num


# In[8]:


class CrossEntropyLoss_for_FA_CE_VNV(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_for_FA_CE_VNV, self).__init__()
    def forward(self, output, one_hot):
        '''
        output: network output(not softmax yet)
                shape: [N, 1, f, t]
        one_hot: ground truth(has not been gaussian blurred)
                shape: [N, f, t]
        [OUTPUT]是一个batch中不同样本的均值
        '''
        output = output.squeeze(dim=1) # output: [N, f, t]
        sm = output + 1e-20
        
        # 对one-hot进行处理，没有label的全零列改成1/f

        none_zero_lines = one_hot.bool().any(1).reshape(-1)
        one_hot_float = one_hot.float()
        index0 = torch.tensor([list(range(one_hot_float.shape[0]))]*one_hot_float.shape[2]).T.reshape(-1)[none_zero_lines == False]
        index2 = torch.tensor([list(range(one_hot_float.shape[2]))]*one_hot_float.shape[0]).reshape(-1)[none_zero_lines == False]
        one_hot_float[index0, :, index2] = 1/one_hot_float.shape[1]
        
        loss_all = (-torch.log(sm)*one_hot_float).sum() # scalar tensor
        num_all = output.shape[2]*output.shape[0]
        
        none_zero_lines_num = none_zero_lines.sum()
        if none_zero_lines_num == 0: # 全是unvoicing
            loss_voicing = loss_all*0
            num_voicing = loss_all*0
        else:
            loss_voicing = (-torch.log(sm)*one_hot).sum()
            num_voicing = none_zero_lines_num
            
        loss_unvoicing = loss_all - loss_voicing
        num_unvoicing = loss_all*0 + (num_all - num_voicing)
        return loss_voicing, num_voicing, loss_unvoicing, num_unvoicing


# In[ ]:


import utils

class CrossEntropyLoss_for_FA_CE_TF(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_for_FA_CE_TF, self).__init__()
    def forward(self, output, one_hot, threshold):
        '''
        output: network output(not softmax yet)
                shape: [N, 1, f, t]
        one_hot: ground truth(has not been gaussian blurred)
                shape: [N, f, t]
        [OUTPUT]是一个batch中不同样本的均值
        '''
        output = output.squeeze(dim=1) # output: [N, f, t]
        sm = output + 1e-20
        
        # 预测结果 T or F
        Xout = utils.salience_to_output(output.clone().detach(), threshold=threshold).to(torch.int32)
        cmp_time = (Xout != one_hot).to(torch.int32).sum(dim=1).bool().to(torch.int32)
        
        # 对one-hot进行处理，没有label的全零列改成1/f

        none_zero_lines = one_hot.bool().any(1).reshape(-1)
        one_hot_float = one_hot.float()
        index0 = torch.tensor([list(range(one_hot_float.shape[0]))]*one_hot_float.shape[2]).T.reshape(-1)[none_zero_lines == False]
        index2 = torch.tensor([list(range(one_hot_float.shape[2]))]*one_hot_float.shape[0]).reshape(-1)[none_zero_lines == False]
        one_hot_float[index0, :, index2] = 1/one_hot_float.shape[1]
        
        loss_all = (-torch.log(sm)*one_hot_float).sum() # scalar tensor
        num_all = output.shape[2]*output.shape[0]
        # ----------------------------------------------------------------------
        loss_time = (-torch.log(sm)*one_hot_float).sum(dim=1) # scalar tensor
        
        False_loss = (loss_time * cmp_time).sum()
        False_num = cmp_time.sum()
        True_loss = loss_time.sum() - False_loss
        True_num = num_all - False_num
        
        # ----------------------------------------------------------------------
        '''none_zero_lines_num = none_zero_lines.sum()
        if none_zero_lines_num == 0: # 全是unvoicing
            loss_voicing = loss_all*0
            num_voicing = loss_all*0
        else:
            loss_voicing = (-torch.log(sm)*one_hot).sum()
            num_voicing = none_zero_lines_num
            
        loss_unvoicing = loss_all - loss_voicing
        num_unvoicing = loss_all*0 + (num_all - num_voicing)'''
        
        return False_loss, False_num, True_loss, True_num

