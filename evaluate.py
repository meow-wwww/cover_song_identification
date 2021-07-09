#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mir_eval
import torch
import numpy as np
from prepare_data_and_label import get_freq_grid

def evaluate(y_pred_clean, y, num_floor):
    '''
    多样本(返回测评结果的均值) or 单样本
    y_pred_clean: [N, 1, f, t] or [f, t], 需要先peak picking, 并且把unvoicing屏蔽掉
    y:            [N, f, t]    or [f, t], label
    num_floor:    0/1/2/3
    y_pred_clean和y缩放无影响
    '''
    
    if len(y_pred_clean.shape) == 2 and y_pred_clean.shape == y.shape:
        y_pred_clean = y_pred_clean.cpu()
        y = y.cpu()
    elif len(y_pred_clean.shape) == 4 and len(y.shape) == 3 and y_pred_clean.shape[-2:] == y.shape[-2:]:
        y_pred_clean = y_pred_clean.squeeze(1)
        y_pred_clean = torch.cat(tuple(y_pred_clean), dim=1).cpu()
        y = torch.cat(tuple(y), dim=1).cpu()
    else:
        assert False, ('输入有误，既不是规范的单样本也不是规范的多样本')
    
    est_time = np.arange(y_pred_clean.shape[1])
    est_freq = y_pred_clean.argmax(0)
    ref_time = est_time
    ref_freq = y.argmax(0)
    
    # 标记unvoicing
    ref_freq[y[ref_freq, ref_time]<=0] = -1
    est_freq[y_pred_clean[est_freq, est_time]<=0] = -1 # 这里的-1只是下标
                                                       # 要转换成表示unvoicing的0Hz还在后面
    
    f = get_freq_grid()
    idx = np.arange(0,360,2**num_floor)
    freq = f[idx]
    freq = np.append(freq, 0)
    
    ref_freq = freq[ref_freq]
    est_freq = freq[est_freq]
    
    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time, ref_freq, est_time, est_freq, base_frequency=32.7)
    
    overall_accuracy = mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c)
    voicing_recall = mir_eval.melody.voicing_recall(ref_v, est_v)
    voicing_false_alarm = mir_eval.melody.voicing_false_alarm(ref_v, est_v)
    raw_pitch_accuracy = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
    raw_chroma_accuracy = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
    
    return overall_accuracy,  voicing_recall, voicing_false_alarm, raw_pitch_accuracy, raw_chroma_accuracy


# In[5]:
