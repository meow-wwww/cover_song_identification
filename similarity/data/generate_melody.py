#!/usr/bin/env python
# coding: utf-8

# In[27]:


import sys 
sys.path.append("../..")
from prepare_data_and_label import compute_hcqt
import data_generator
import model_unet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm


# In[ ]:


you350_path = '/S3/DAA/jcy/SCNN/data/you350/audio/'
you350_list = os.listdir(you350_path)

model = model_unet.UNet().cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('../../model/th_0_4/0/model_floor0_best.pth'))
model.eval()

for file in tqdm(you350_list):
    hcqt = compute_hcqt(os.path.join(you350_path, file))
    out = data_generator.hcqt_to_melody(hcqt, model, threshold=0.4)
    mean_size = 20
    height, length = out.shape
    new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
    for i in range(int(length/mean_size)):
        new_cqt[:,i] = out[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
    np.save(os.path.join('./you350/', file.split('.')[0]+'.npy'), new_cqt)


# In[ ]:




