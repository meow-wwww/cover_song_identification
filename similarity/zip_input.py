#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
from tqdm import tqdm
import numpy as np


# In[4]:


to_be_zip = os.listdir('/S3/DAA/gdoras_dataset/train/multif0_cqts_padded_1937x72')


# In[ ]:


for file in tqdm(to_be_zip):
    try:
        data = np.load(os.path.join('/S3/DAA/gdoras_dataset/train/multif0_cqts_padded_1937x72', file))
        mean_size = 5
        height, length = data.shape
        new_cqt = np.zeros((height, int(length/mean_size)),dtype=np.float64)
        for i in range(int(length/mean_size)):
            new_cqt[:, i] = data[:, i*mean_size:(i+1)*mean_size].mean(axis=1)
        np.save(os.path.join(f'/S3/DAA/gdoras_dataset/avg{mean_size}/train', file), new_cqt)
    except:
        print(f'error: {file}')