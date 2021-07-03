#!/usr/bin/env python
# coding: utf-8

# In[137]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import hparams
import random
import dataset_track_id


# In[ ]:


# 这个用在最后测试时更合适
def chunk_and_concat(src_data): 
    '''
    chunk hcqt or target
    src_data: [t, f] or [t, f, h]
    '''
    data_chunks_duration_in_bins,     data_chunks_overlap_in_bins = hparams.data_chunks_duration_in_bins, hparams.data_chunks_overlap_in_bins

    data_chunks_beg_overlap_in_bins = data_chunks_overlap_in_bins[0]
    data_chunks_end_overlap_in_bins = data_chunks_overlap_in_bins[1]

    chunks,     last_chunk = chunker.chunk_data_with_same_padding(src_data,
                                                      data_chunks_duration_in_bins=data_chunks_duration_in_bins,
                                                      data_chunks_overlap_in_bins=data_chunks_overlap_in_bins) # [t, ...]

    # pile all the chunks (the number of chunk will be the batch size).
    src_data_chunks = np.stack(chunks)  # [num_chunks, h, f, t] or [num_chunks, f, t]

    # now apply network
    num_chunks = src_data_chunks.shape[0]
    batch_size = hparams.dataset_eval_batch_size

    outputs = []

    for i in range(0, num_chunks, batch_size):

        if i + batch_size < num_chunks:
            dataset_placeholders = [src_data_chunks[i:i+batch_size]] # [num_batch, (h,) f, t]
        else:
            dataset_placeholders = [src_data_chunks[i:]]

        batched_output = model.apply(dataset_placeholders=dataset_placeholders) # [num_batch, t]

        # we might have overlapped, so we cut that part, except for the very first one
        if i == 0 and data_chunks_beg_overlap_in_bins > 0:
            outputs.append(batched_output[0, :data_chunks_beg_overlap_in_bins])

        # all the other chunks are trimmed at the beginning and the end
        if data_chunks_beg_overlap_in_bins > 0:
            batched_output = batched_output[:, data_chunks_beg_overlap_in_bins:]
        if data_chunks_end_overlap_in_bins > 0:
            batched_output = batched_output[:, :-data_chunks_end_overlap_in_bins]

        # concatenate along time axis
        batched_output = np.concatenate(batched_output.tolist(), axis=0) # [num_batch*t]
        batched_output = batched_output.astype(np.float32) # important: concatenate changes the dtype !!!
        outputs.append(batched_output)

    # add the last chunk (时间尺度短了，但同样可以feed给网络并获得结果)
    if last_chunk is not None:
        last_data = np.expand_dims(last_chunk, 0) # [b=1, t, ...]
        dataset_placeholders = [last_data]
        batched_output = model.apply(dataset_placeholders=dataset_placeholders)  #[b=1, t, ...]

        last_output = np.squeeze(batched_output, 0) #[t,...]

        # remove the beginning
        if data_chunks_beg_overlap_in_bins > 0:
            last_output = last_output[data_chunks_beg_overlap_in_bins:]
        outputs.append(last_output)

    # concatenate all
    output = np.concatenate(outputs, axis=0) # [t,...]
    output = output.astype(np.float32) # important, concatenate changes the dtype !!!

    return output


# In[100]:


def chunk_data(data, # data_chunks_duration_in_bins=None,
                     # data_chunks_overlap_in_bins=None,
                     include_last_chunk ):
    """
    Chunks data.
    Args:
        data: ndarray, [h, f ,t] or [f, t]
        data_chunks_duration_in_bins: # 暂时废弃
        data_chunks_overlap_in_bins: # 暂时废弃
        include_last_chunk: bool
    Returns: a list(include_last_chunk=True)
            or a tuple:(list, np-array)(include_last_chunk=False)
    """
    data = data.T

    data_chunks_duration_in_bins = hparams.data_chunks_duration_in_bins
    
    chunks_beg_overlap_in_bins = hparams.data_chunks_overlap_in_bins[0]
    chunks_end_overlap_in_bins = hparams.data_chunks_overlap_in_bins[1]

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
        
    if include_last_chunk:
        return chunks, last_chunk
    else:
        return chunks


# In[128]:


def source_index_to_chunk_list(source_list):
    '''
    source_list: a list of index eg.[0,1,4,5,6,7,8,9]
    doesn't include last_chunk
    '''
    chunk_list = []
    for fold_index in source_list:
        fold_list = dataset_track_id.dataset_track_id_list[fold_index]
        for track_id in fold_list:
            X = np.load(f'./inputs/{track_id}_mel2_input.hcqt.npy')
            y = np.load(f'./outputs/{track_id}_mel2_output.npy')
            X_chunk = chunk_data(X, include_last_chunk=False)
            y_chunk = chunk_data(y, include_last_chunk=False)
            chunk_list += list(zip(X_chunk, y_chunk))
    random.shuffle(chunk_list)
    return chunk_list


# In[138]:





# In[ ]:




