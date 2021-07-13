#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random
import datetime
from tqdm import tqdm
import argparse, os, math

import data_generator
import hparams
import model_unet_unfold as model_unet
import numpy as np
import loss_function
import utils
import evaluate


# # argparse

# In[ ]:


save_dir = None
lr = 0
saved_model_path = None
epochs = 0
epochs_finished = 0
num_floor = -1
device_ids = []

print('--------------ArgParse--------------')

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', help='目标存储目录')
parser.add_argument('--saved_model', help='以训练过的模型')
parser.add_argument('--lr', type=float, help='学习率')
parser.add_argument('-e', '--epochs', type=int, help='有几个epoch')
# parser.add_argument('--epochs_finished', type=int, help='已经完成了几个epoch')
parser.add_argument('-o', '--out_floor', type=int, help='输出在第几层(0，1，2，3)')
parser.add_argument('--gpu', help='要用的gpu号')

args = parser.parse_args()

assert args.save_dir != None, ('请输入存储目录')
if not os.path.exists(args.save_dir):
    print(f'Save_dir {args.save_dir} does not exist. Will create one.')
    save_dir = args.save_dir
    os.mkdir(save_dir)
else:
    print(f'Save_dir {args.save_dir} already exist. Will save to this directory.')
    save_dir = args.save_dir


if args.saved_model == None:
    print('Don\'t use saved model, train from scratch')
else:
    print(f'Use saved model[{args.saved_model}], train from it')
saved_model_path = args.saved_model

if args.lr == None:
    lr = 1e-3
    print('Using default lr=1e-3')
else:
    lr = args.lr
    print(f'Using lr from command: lr={lr}')

assert args.epochs != None, ('请输入epochs数')
epochs = args.epochs

assert args.out_floor != None, ('请输入输出的层数')
assert args.out_floor in [0,1,2,3], ('输入的层数必须为0 1 2 3之一')
num_floor = args.out_floor

assert args.gpu != None, ('请输入要用的gpu号')
device_ids = list(map(lambda x: int(x), args.gpu.split(',')))
print(f'device_ids: {device_ids}')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# In[ ]:


model = model_unet.UNet()
if saved_model_path != None:
    print(f'loading model from {saved_model_path}...')
    model = torch.load(saved_model_path)
else:
    print('raw model')
    
# multi-gpu
model = nn.DataParallel(model)
model = model.cuda()
    
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
scheduler_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94, verbose=True)
scheduler_stop = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-3, factor=-1, patience=1000)

loss_fn = loss_function.CrossEntropyLoss_for_FA_CE()
loss_fn = nn.DataParallel(loss_fn)
loss_fn = loss_fn.cuda()


# # split data, generate train/test_dataloader

# In[ ]:


'''
fold_index = list(range(10))
random.shuffle(fold_index)
test_fold_index = fold_index[0]
validation_fold_index = fold_index[1]
train_fold_index_list = fold_index[2:]
'''
train_fold_index_list = hparams.train_set_fold_index
valid_fold_index_list = hparams.validation_set_fold_index

# prepare dataloader
print(f'{datetime.datetime.now()} - Preparing train_dataloader...')
train_dataloader = data_generator.source_index_to_chunk_list(source_list=train_fold_index_list, 
                                                             data_chunks_duration_in_bins=hparams.data_chunks_duration_in_bins,
                                                             data_chunks_overlap_in_bins=hparams.data_chunks_overlap_in_bins_for_training)
print(f'{datetime.datetime.now()} - Preparing valid_dataloader...')
valid_dataloader = data_generator.source_index_to_chunk_list(source_list=valid_fold_index_list,
                                                             data_chunks_duration_in_bins=hparams.data_chunks_duration_in_bins,
                                                             data_chunks_overlap_in_bins=hparams.data_chunks_overlap_in_bins_for_training)

train_dataloader = DataLoader(train_dataloader, batch_size=hparams.batch_size*len(device_ids), shuffle=True)
valid_dataloader = DataLoader(valid_dataloader, batch_size=hparams.batch_size*len(device_ids), shuffle=False)


# # train/test function

# In[ ]:


# 一个epoch的训练+测试
def train(dataloader, model, loss_fn, optimizer, scheduler, out_floor):
    model.train()
    size = len(dataloader.dataset)
    batch_num = math.ceil(size/dataloader.batch_size)
    loss_total = 0
    
    for batch, (X, y) in tqdm(enumerate(dataloader)): # 每次返回一个batch
        # X, y = X.to(device), y.to(device)
        X, y = X.cuda(), y.cuda()
        # Compute prediction error
        pred = model(X, out_floor)
        
        if out_floor == 0:
            loss = loss_fn(pred, y)
        else:
            # downsample y
            y_downsample = utils.downsample(y, out_floor)
            loss = loss_fn(pred, y_downsample)
            
        loss_total += loss.item()
            
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch+1) % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Avg loss: {loss:.4f}  [{current:>5d}/{size:>5d}]")
            
    scheduler.step()
    return loss_total/batch_num
            
def test(dataloader, model, loss_fn, out_floor):
    
    with torch.no_grad():
    
        size = len(dataloader.dataset)
        batch_num = math.ceil(size/dataloader.batch_size)

        model.eval()
        test_loss = 0
        oa_avg, vr_avg, vfa_avg, rpa_avg, rca_avg = 0, 0, 0, 0, 0
    
    
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device) # single-gpu
            X, y = X.cuda(), y.cuda()
            Xpred = model(X, out_floor)
            Xout = utils.salience_to_output(Xpred, threshold=hparams.threshold)
            
            if out_floor == 0:
                loss = loss_fn(Xpred, y)
                oa, vr, vfa, rpa, rca = evaluate.evaluate(Xout, y, out_floor)
            else:
                # downsample y
                y_downsample = utils.downsample(y, out_floor)
                loss = loss_fn(Xpred, y_downsample)
                oa, vr, vfa, rpa, rca = evaluate.evaluate(Xout, y_downsample, out_floor)
            
            test_loss += loss.item()
            oa_avg += oa
            vr_avg += vr
            vfa_avg += vfa
            rpa_avg += rpa
            rca_avg += rca
            
    test_loss /= batch_num # 每张图的loss
    
    oa_avg /= batch_num
    vr_avg /= batch_num
    vfa_avg /= batch_num
    rpa_avg /= batch_num
    rca_avg /= batch_num
    
    print(f"Test Error: Avg loss: {test_loss:.4f} \n")
    print(f"Test OA\t{oa_avg:.4f}\tVR\t{vr_avg:.4f}\tVFA\t{vfa_avg:.4f}\tRPA\t{rpa_avg:.4f}\tRCA\t{rca_avg:.4f}\n")
    
    return test_loss, oa_avg, vr_avg, vfa_avg, rpa_avg, rca_avg


# # 训练模型🌟

# In[ ]:


# 多个epoch训练，每个epoch后在验证集上测试

train_loss_list = []
valid_loss_list = []
best_oa = 0

for t in range(epochs_finished, epochs_finished+epochs):
    print(f"Epoch {t+1}\n-------------------------------{datetime.datetime.now()}")
    train_loss = train(train_dataloader, model, loss_fn, optimizer, scheduler_decay, num_floor)
    valid_loss, oa, _, _, _, _ = test(valid_dataloader, model, loss_fn, num_floor)

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    
    scheduler_stop.step(valid_loss)
    
    if optimizer.state_dict()['param_groups'][0]['lr']<=0:
        print(f'Early stop after {t+1} epochs.')
        break
    
    plt.plot(range(1,len(train_loss_list)+1), train_loss_list, c='b')
    plt.plot(range(1,len(valid_loss_list)+1), valid_loss_list, c='r')
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    
    # 保存最优模型
    if oa > best_oa:
        with torch.no_grad():
            print(f'[in epoch {t+1}, OA got new best: from {best_oa} to {oa}. Will save the model]')
            best_oa = oa
            torch.save(model, os.path.join(save_dir, f'model_floor{num_floor}_best.pth'))
    
print("Done!")


# In[ ]:

