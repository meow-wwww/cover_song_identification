#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys 
sys.path.append("../..")
from less_bins.prepare_data_and_label import compute_hcqt
import data_generator
import model_unet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--wan', help='目标存储目录')

args = parser.parse_args()
start = int(args.wan)*10000
end = (int(args.wan)+1)*10000

print(f'[{start}:{end}]')


# In[ ]:


import gzip
 
def un_gz(file_name):
    
    # 获取文件的名称，去掉后缀名
    f_name = file_name.replace(".gz", "")
    # 开始解压
    g_file = gzip.GzipFile(file_name)
    #读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


# In[ ]:


in_dir = '/S3/DAA/data/'
out_dir = '/S3/DAA/shs100k/melody-3bin/'


# In[ ]:


model = model_unet.UNet().cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('../../model/6.1/0/model_floor0_best.pth'))
model.eval()
pass


# In[ ]:


def CQT(args):
    try:
        in_path, out_path = args
        hcqt = compute_hcqt(in_path)
        out = data_generator.hcqt_to_melody(hcqt, model, threshold=0.4)
        mean_size = 20
        height, length = out.shape
        new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
        for i in range(int(length/mean_size)):
            new_cqt[:,i] = out[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
        np.save(out_path, new_cqt)
        print(f"saved {out_path.split('/')[-1]}")
    except:
        print('wa', in_path)


# In[ ]:


params =[]
for ii, (root, dirs, files) in tqdm(enumerate(os.walk(in_dir))):
    if len(files):
        for file in files:
            in_path = os.path.join(root,file)
            set_id = root.split('/')[-1]
            out_path = out_dir + set_id + '_' + file.split('.')[0] + '.npy'
            params.append((in_path, out_path))

params = params[start:end]


# In[ ]:


print('begin')
for param in tqdm(params):
    CQT(param)

