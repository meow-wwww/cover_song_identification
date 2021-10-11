#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random
import datetime
from tqdm import tqdm
import argparse, os, math
import numpy as np
from hpcp_loader_for_softdtw import *
import models.BaseSPPNet as models
from config import DefaultConfig, opt
from utility import *
import json
import torch.nn.functional as F
from torch.nn import *
import pandas as pd
import seaborn as sns
import resource
import librosa


# In[ ]:


# argparse
save_dir = None
saved_model_path = None
lr = 1e-3
weight_decay = 0
num_workers = 0
epochs = 0
epochs_finished = 0
BATCH_SIZE = 16
device_ids = [0]
datatype = None
train_scale = None
test_scale = None
test_source = None
train_cut = None

print('--------------ArgParse--------------')

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--save_dir', help='目标存储目录')
parser.add_argument('-s', '--saved_model', help='以训练过的模型')
parser.add_argument('--lr', type=float, help='学习率')
parser.add_argument('--weight_decay', type=float, help='正则化系数')
parser.add_argument('--workers', type=int, help='num_workers')
parser.add_argument('-e', '--epochs', type=int, help='有几个epoch')
parser.add_argument('-g', '--gpu', help='要用的gpu号')
parser.add_argument('-b', '--batch_size', type=int, help='batch_size')
parser.add_argument('--datatype', help='训练数据的形式,f0,multif0,cqt等')
parser.add_argument('--train_scale', help='训练集大小尺度')
parser.add_argument('--test_scale', help='测试集大小尺度')
parser.add_argument('--test_source', help='测试用数据来源')
parser.add_argument('--train_cut', help='训练数据切割方式')

args = parser.parse_args()

assert args.save_dir != None, ('请输入存储目录')
if not os.path.exists(args.save_dir):
    print(f'| Save_dir {args.save_dir} does not exist. Will create one.')
    save_dir = args.save_dir
    os.mkdir(save_dir)
else:
    print(f'| Save_dir {args.save_dir} already exist. Will save to this directory.')
    save_dir = args.save_dir

if args.saved_model == None:
    print('| train from scratch')
else:
    print(f'| saved model:\t{args.saved_model}')
saved_model_path = args.saved_model

if args.lr != None:
    lr = args.lr
print(f'| lr:\t{lr}')
    
if args.weight_decay != None:
    weight_decay = args.weight_decay
print(f'| weight_dacay:\t{weight_decay}')
    
assert args.workers != None, ('请输入workers数')
num_workers = args.workers
print(f'| num_workers:\t{num_workers}')

assert args.epochs != None, ('请输入epochs数')
epochs = args.epochs
print(f'| epochs:\t{epochs}')

assert args.gpu != None, ('请输入要用的gpu号')
device_ids = list(range(len(args.gpu.split(','))))
print(f'| gpu:\t{args.gpu}')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f'| device count:\t{torch.cuda.device_count()}')

device = torch.device("cuda:0")

if args.batch_size != None:
    BATCH_SIZE = args.batch_size
print(f'| batch_size:\t{BATCH_SIZE}')

loss_fn = nn.BCELoss()
print(f'| loss_function:\t{loss_fn.__class__.__name__}')

assert args.datatype != None
datatype = args.datatype
print(f'| datatype:\t{datatype}')

train_scale = args.train_scale
print(f'| train_scale:\t{train_scale}')

test_scale = args.test_scale
print(f'| test_scale:\t{test_scale}')

assert args.test_source != None
test_source = args.test_source
print(f'| test_source:\t{test_source}')

assert args.train_cut in ['random', 'front'] or args.train_cut[0:11] == 'semi-random'
train_cut = args.train_cut
print(f'| train_cut:\t{train_cut}')


# In[6]:


model = models.NeuralDTW_CNN_Mask_dilation_SPP6(None)
model = model.cuda()
model = nn.DataParallel(model)

if saved_model_path != None:
    print(f'loading model from {saved_model_path}...')
    model.load_state_dict(torch.load(saved_model_path))
    
from torch.nn import init
#define the initial function to init the layer's parameters for the network
def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight)
            if m.bias != None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias != None:
                init.constant(m.bias, 0)
initNetParams(model)

loss_fn = nn.BCELoss() # loss_fn返回的值已经对每个样本做平均了
loss_fn = loss_fn.cuda()
loss_fn = nn.DataParallel(loss_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,
                                                               verbose=True, min_lr=5e-10)


# In[7]:


# Dataloader
print(f'{datetime.datetime.now()} - Preparing train_dataloader...')
train_data = triplet_CQT(out_length=400, is_label=True, mode=(datatype, train_scale), cut=train_cut, is_random=True)
train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=num_workers)

print(f'{datetime.datetime.now()} - Preparing test_dataloader...')
test_data = CQT(mode=(test_source, datatype, test_scale), out_length=400)
test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=True, num_workers=num_workers)


# In[38]:


# 一个epoch的训练/测试
def train(dataloader, model, loss_fn, optimizer, scheduler):
    model.train()
    model.module.model.train()
    
    size = len(dataloader.dataset)
    batch_num = math.ceil(size/dataloader.batch_size)
    
    loss_total = 0
    num = 0
    
    for batch, (a, p, n, la, lp, ln) in tqdm(enumerate(dataloader)):
        B, _, _, _ = a.shape
        target = torch.cat((torch.ones(B), torch.zeros(B))).cuda()
        a, p, n = a.cuda(device=device_ids[0]), p.cuda(device=device_ids[0]), n.cuda(device=device_ids[0])
        optimizer.zero_grad()
        pred = model(a, p, n)
        pred = pred.squeeze(1)
#         print(pred, target)
        loss = loss_fn(pred, target)
        loss = loss.sum()
#         print(loss)
        loss.backward()
#         print(list(model.named_parameters())[0][1].grad[0])
        optimizer.step()

        loss_total += loss.item()
        num += B

        if (batch+1) % 100 == 0:
            print(f"Avg loss: {loss.item():.4f}  [{num:>5d}/{size:>5d}]")
            scheduler.step(loss.item())

    return loss_total/batch_num


# In[39]:


@torch.no_grad()
def val_slow_batch(softdtw, dataloader, batch=50, is_dis=False):
    softdtw.eval()
    softdtw.module.model.eval()
    seqs, labels = [], []
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data.cuda(device=device_ids[0])
        seqs.append(input)
        labels.append(label)
    seqs = torch.cat(seqs, dim=0)
    labels = torch.cat(labels, dim=0)
    N = labels.shape[0]
    if N == 350:
        query_l = [i // 100 for i in range(100 * 100, 350 * 100)]
        ref_l = [i for i in range(100)] * 250
    else:
        query_l = [i // N for i in range(N * N)] # [0,...,0,1,...,1,...,N,...,N]
        ref_l = [i for i in range(N)] * N # [0,...,N,0,...,N,...,0,...,N]
    dis2d = np.zeros((N, N))

    N = N * N if N != 350 else 100 * 250 # N = N*N
    for st in tqdm(range(0, N, batch)):
        fi = (st + batch) if st + batch <= N else N
        query = seqs[query_l[st: fi], :, :]
        ref = seqs[ref_l[st: fi], :, :]
        s = softdtw.module.multi_compute_s(query, ref).data.cpu().numpy()
        for k in range(st, fi):
            i, j = query_l[k], ref_l[k]
            if is_dis:
                dis2d[i, j] = s[k - st]
            else:
                dis2d[i, j] = -s[k - st]
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(f'MAP:\t{MAP:.4f}\ttop10:\t{top10:.4f}\trank1:\t{rank1:.4f}')

    softdtw.train()
    softdtw.module.model.train()
    return MAP

@torch.no_grad()
def val_quick(softdtw, dataloader):
    softdtw.eval()
    softdtw.module.model.eval()
    labels = []
    temp = []
    count = -1
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        labels.append(label)
    labels = torch.cat(labels, dim=0) # labels: 库里每首歌的song_id
    N = labels.shape[0]
    dis2d = np.zeros((N, N))
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        data = data.cuda(device=device_ids[0])
        count += 1
        if count == 0: # for中的第一轮循环
            temp.append((data, count)) # temp[0] = (data, 0)
        else:
            for i in range(len(temp)):
                dis = softdtw.module.multi_compute_s(data, temp[i][0]).data.cpu().numpy()
                dis2d[temp[i][1]][count], dis2d[count][temp[i][1]] = -dis, -dis
            temp.append((data, count))

    MAP, top10, rank1 = calc_MAP(dis2d[0:labels.shape[0], 0:labels.shape[0]], labels)
    print(f'MAP:\t{MAP:.4f}\ttop10:\t{top10:.4f}\trank1:\t{rank1:.4f}')
    softdtw.train()
    softdtw.module.model.train()
    return MAP


# In[ ]:


train_loss_list = []
test_MAP_list = []
best_MAP = val_slow_batch(model, test_dataloader)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------{datetime.datetime.now()}")
    train_loss = train(train_dataloader, model, loss_fn, optimizer, scheduler)
    print(f'train_loss:{train_loss:.6f}')

    MAP = 0
    print('gdoras_test:')
    MAP += val_slow_batch(model, test_dataloader)
    # test on various dataset
    # print("Youtube350:")
    # MAP += val_slow_batch(model, val_dataloader, batch=100, is_dis=kwargs['zo'])
    # print("CoverSong80:")
    # MAP += val_slow_batch(model, val_dataloader80, batch=100, is_dis=kwargs['zo'])
    # print("Youtube350:")
    # MAP += val_quick(model, val_dataloader_350)
    # print("CoverSong80:")
    # MAP += val_quick(model, val_dataloader_80)
    # print("Mazurkas:")
    # MAP += val_slow_batch(model, val_dataloader_marukars, batch=100, is_dis=kwargs['zo'])
    if MAP > best_MAP:
        print('*****************BEST*****************')
        print(f'[epoch {t+1}] {best_MAP:.4f} --- {MAP:.4f}. Save.')
        best_MAP = MAP
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_best.pth'))
        torch.save(model.state_dict(), os.path.join(save_dir, f'latest.pth'))

    train_loss_list.append(train_loss)
    test_MAP_list.append(MAP)
    # valid_oa_list.append(oa)
    # test_oa_list.append(test_oa)
    
    if optimizer.state_dict()['param_groups'][0]['lr']<=0:
        print(f'Early stop after {t+1} epochs.')
        break
    
    plt.cla()
    plt.figure(figsize=(7,10))
    ax = plt.subplot(2,1,1)
    ax.set_title('train_loss')
    plt.plot(range(1,len(train_loss_list)+1), train_loss_list, 'b')
    ax = plt.subplot(2,1,2)
    ax.set_title('test_MAP')
    plt.plot(range(1,len(test_MAP_list)+1), test_MAP_list, 'r')
    plt.savefig(os.path.join(save_dir, 'loss_and_MAP.png'))
    
print('Done!')

