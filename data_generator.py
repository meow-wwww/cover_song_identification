#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import os
import hparams
import random
import dataset_track_id
import utils


# In[ ]:


def chunk_data_with_same_padding(data,
                                 data_chunks_duration_in_bins,
                                 data_chunks_overlap_in_bins,
                                 ):
    """
    Chunks data.
    Args:
        data: ndarray, [t, f, h]
        data_chunks_duration_in_bins:
        data_chunks_overlap_in_bins:
    Returns:
    """

    if type(data_chunks_overlap_in_bins) is list or type(data_chunks_overlap_in_bins) is tuple:
        chunks_beg_overlap_in_bins = data_chunks_overlap_in_bins[0]
        chunks_end_overlap_in_bins = data_chunks_overlap_in_bins[1]
    else:
        chunks_beg_overlap_in_bins = 0
        chunks_end_overlap_in_bins = data_chunks_overlap_in_bins

    start_bin = 0
    end_bin = start_bin + data_chunks_duration_in_bins
    chunks = []

    while end_bin < data.shape[0]:

        chunks.append(data[start_bin:end_bin].T)

        start_bin = end_bin - (chunks_beg_overlap_in_bins + chunks_end_overlap_in_bins)
        end_bin = start_bin + data_chunks_duration_in_bins

    # save last chunk
    end_bin = data.shape[0]
    if end_bin > start_bin:
        last_chunk = data[start_bin:end_bin].T

    else:
        last_chunk = None

    return chunks, last_chunk


# In[21]:


# 这个用在最后测试时更合适
def hcqt_to_melody(data, model, threshold): 
    """
    Chunks hcqt data.
    Args:
        data: ndarray, [h, f ,t]
        # data_chunks_duration_in_bins: 每一个训练样本的时间步长度
        # data_chunks_overlap_in_bins: 不同训练样本间跳步长度
    Returns: a tuple:(list, np-array) (include_last_chunk=True)
    """
    data = data.T # [t, f, h]
    
    data_chunks_duration_in_bins, data_chunks_overlap_in_bins = hparams.data_chunks_duration_in_bins, hparams.data_chunks_overlap_in_bins
    chunks_beg_overlap_in_bins = data_chunks_overlap_in_bins[0]
    chunks_end_overlap_in_bins = data_chunks_overlap_in_bins[1]

    chunks, last_chunk = chunk_data_with_same_padding(data, data_chunks_duration_in_bins, data_chunks_overlap_in_bins)
                                      
    chunks = torch.tensor(chunks) # [N, h, f, t]
    last_chunk = torch.tensor(last_chunk).unsqueeze(0) # [1, h, f, t]
    
    model.eval()

    # now apply network
    num_chunks = chunks.shape[0]
    batch_size = 16

    outputs = []

    for i in range(0, num_chunks, batch_size):

        if i + batch_size < num_chunks:
            dataset_placeholders = chunks[i:i+batch_size] # [batch_size, h, f, t]
        else:
            dataset_placeholders = chunks[i:]

        batched_output = model(dataset_placeholders, 0) # [num_batch, 1, f, t]
        batched_output = utils.salience_to_output(batched_output, threshold)

        # we might have overlapped, so we cut that part, except for the very first one
        if i == 0 and chunks_beg_overlap_in_bins > 0:
            outputs.append(batched_output[0, :, :, :chunks_beg_overlap_in_bins].cpu().detach().numpy().squeeze(0))

        # all the other chunks are trimmed at the beginning and the end
        if chunks_beg_overlap_in_bins > 0:
            batched_output = batched_output[:, :, :, chunks_beg_overlap_in_bins:]
        if chunks_end_overlap_in_bins > 0:
            batched_output = batched_output[:, :, :, :-chunks_end_overlap_in_bins]
        batched_output = batched_output.cpu().detach().numpy()
        batched_output = np.squeeze(batched_output, 1)

        # concatenate along time axis
        batched_output = np.concatenate(batched_output.tolist(), axis=1) # [num_batch*t]
        # batched_output = batched_output.astype(np.float32) # important: concatenate changes the dtype !!!
        outputs.append(batched_output)
        
    if last_chunk is not None:
        batched_output = model(last_chunk, 0)  #[1, 1, f, t]
        batched_output = utils.salience_to_output(batched_output, threshold)
        last_output = batched_output.squeeze(0).squeeze(0) #[f, t]

        # remove the beginning
        if chunks_beg_overlap_in_bins > 0:
            last_output = last_output[:, chunks_beg_overlap_in_bins:].cpu().detach().numpy()
        outputs.append(last_output)

    # concatenate all
    output = np.concatenate(outputs, axis=-1) #
    # output = output.astype(np.float32) # important, concatenate changes the dtype !!!

    return np.array(output)


# In[ ]:





# In[ ]:


def chunk_data(data, data_chunks_duration_in_bins,
                     data_chunks_overlap_in_bins,
                     include_last_chunk):
    """
    Chunks data.
    Args:
        data: ndarray, [h, f ,t] or [f, t]
        data_chunks_duration_in_bins: 每一个训练样本的时间步长度
        data_chunks_overlap_in_bins: 不同训练样本间跳步长度
        include_last_chunk: bool
    Returns: a list (include_last_chunk=False)
             or a tuple:(list, np-array) (include_last_chunk=True)
    """
    data = data.T # [t,f] or [t,f,h]

    # data_chunks_duration_in_bins = hparams.data_chunks_duration_in_bins
    
    # chunks_beg_overlap_in_bins = hparams.data_chunks_overlap_in_bins[0]
    # chunks_end_overlap_in_bins = hparams.data_chunks_overlap_in_bins[1]

    start_bin = 0
    end_bin = start_bin + data_chunks_duration_in_bins
    chunks = []

    while end_bin < data.shape[0]:

        chunks.append(data[start_bin:end_bin].T)

        start_bin = start_bin + data_chunks_overlap_in_bins
        end_bin = start_bin + data_chunks_duration_in_bins

    # save last chunk
    end_bin = data.shape[0]
    if end_bin > start_bin:
        last_chunk = data[start_bin:end_bin].T
    else:
        last_chunk = None
        
    if include_last_chunk:
        return chunks, last_chunk
    else:
        return chunks

def source_index_to_chunk_list(source_list, data_chunks_duration_in_bins, data_chunks_overlap_in_bins, label):
    '''
    For training.
    Params:
        source_list: a list of index eg.[0,1,4,5,6,7,8,9], meaning the songs to be chunked
        data_chunks_duration_in_bins: 每一个训练样本的时间步长度
        data_chunks_overlap_in_bins: 不同训练样本间跳步长度
        label:str,由于标签有多版，label指示用哪一版。
            'origin':
            'real_one_hot':
    Return:
        chunk_list: a list of tuple(X, y), X is hcqt, y is annotation
    doesn't include last_chunk.
    '''
    if label == 'origin':
        prefix_input = './inputs'
        prefix_output = './outputs'
    elif label == 'real_one_hot':
        prefix_input = os.path.join('./inputs', '')
        prefix_output = os.path.join('./outputs', label)
    
    chunk_list = []
    for fold_index in source_list:
        fold_list = dataset_track_id.dataset_track_id_list[fold_index]
        for track_id in fold_list:
            X = np.load(os.path.join(prefix_input, f'{track_id}_mel2_input.hcqt.npy'))
            y = np.load(os.path.join(prefix_output, f'{track_id}_mel2_output.npy'))
            X_chunk = chunk_data(X, 
                                 data_chunks_duration_in_bins=data_chunks_duration_in_bins, 
                                 data_chunks_overlap_in_bins=data_chunks_overlap_in_bins, 
                                 include_last_chunk=False)
            y_chunk = chunk_data(y, 
                                 data_chunks_duration_in_bins=data_chunks_duration_in_bins, 
                                 data_chunks_overlap_in_bins=data_chunks_overlap_in_bins, 
                                 include_last_chunk=False)
            chunk_list += list(zip(X_chunk, y_chunk))
    # random.shuffle(chunk_list)
    return chunk_list


# In[ ]:


def track_id_list_to_chunk_list(track_id_list_name, data_chunks_duration_in_bins, data_chunks_overlap_in_bins, label):
    '''
    For training.
    Params:
        track_id_list_name: str, variable name in dataset_track_id.py
        data_chunks_duration_in_bins: 每一个训练样本的时间步长度
        data_chunks_overlap_in_bins: 不同训练样本间跳步长度
        label:str,由于标签有多版，label指示用哪一版。
            'origin':
            'real_one_hot':
    Return:
        chunk_list: a list of tuple(X, y), X is hcqt, y is annotation
    doesn't include last_chunk.
    '''
    chunk_list = []
    
    fold_list = getattr(dataset_track_id, track_id_list_name)
    
    if label == 'origin':
        prefix_input = './inputs'
        prefix_output = './outputs'
    elif label == 'real_one_hot':
        prefix_input = os.path.join('./inputs', '')
        prefix_output = os.path.join('./outputs', label)
        
    for track_id in fold_list:
        X = np.load(os.path.join(prefix_input, f'{track_id}_mel2_input.hcqt.npy'))
        y = np.load(os.path.join(prefix_output, f'{track_id}_mel2_output.npy'))
        X_chunk = chunk_data(X, 
                             data_chunks_duration_in_bins=data_chunks_duration_in_bins, 
                             data_chunks_overlap_in_bins=data_chunks_overlap_in_bins, 
                             include_last_chunk=False)
        y_chunk = chunk_data(y, 
                             data_chunks_duration_in_bins=data_chunks_duration_in_bins, 
                             data_chunks_overlap_in_bins=data_chunks_overlap_in_bins, 
                             include_last_chunk=False)
        chunk_list += list(zip(X_chunk, y_chunk))
    return chunk_list


# In[ ]:




