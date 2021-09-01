#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from multiprocessing import Pool
from tqdm import tqdm


# In[2]:


import gzip
 
def un_gz(file_name):
    
    # 获取文件的名称，去掉后缀名
    f_name = file_name.replace(".gz", "")
    # 开始解压
    g_file = gzip.GzipFile(file_name)
    #读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


# In[3]:


in_dir = '/media/data/data_share/new_youtube/crawl_data/data/'
out_dir = '/media/data/data_share/new_youtube/melody_line/'


# In[4]:


model = model_unet.UNet().cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('../../model/model_floor0_best.pth'))
model.eval()
pass


# In[5]:


def CQT(args):
    try:
        gz = False
        in_path, out_path = args
        if in_path.split('.')[-1] == 'gz':
            gz = True
            un_gz(in_path)
            in_path = in_path[:-3]
        hcqt = compute_hcqt(in_path)
        out = data_generator.hcqt_to_melody(hcqt, model, threshold=0.4)
        mean_size = 20
        height, length = out.shape
        new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
        for i in range(int(length/mean_size)):
            new_cqt[:,i] = out[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
        np.save(out_path, new_cqt)
        if gz:
            os.remove(in_path)
        print(f"saved {out_path.split('/')[-1]}")
    except:
        print('wa', in_path)


# In[6]:


params =[]
for ii, (root, dirs, files) in tqdm(enumerate(os.walk(in_dir))):
    if len(files):
        for file in files:
            in_path = os.path.join(root,file)
            set_id = root.split('/')[-1]
            out_path = out_dir + set_id + '_' + file.split('.')[0] + '.npy'
            params.append((in_path, out_path))
            
params = params[77:500]


# In[ ]:


print('begin')
for param in tqdm(params):
    CQT(param)



# multi-process
# pool = Pool(40)
# pool.map(CQT, params)
# pool.close()
# pool.join()


# you350_path = '/S3/DAA/jcy/SCNN/data/you350/audio/'
# you350_list = os.listdir(you350_path)
# 
# for file in tqdm(you350_list):
#     hcqt = compute_hcqt(os.path.join(you350_path, file))
#     out = data_generator.hcqt_to_melody(hcqt, model, threshold=0.4)
#     mean_size = 20
#     height, length = out.shape
#     new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
#     for i in range(int(length/mean_size)):
#         new_cqt[:,i] = out[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
#     np.save(os.path.join('./you350/', file.split('.')[0]+'.npy'), new_cqt)
