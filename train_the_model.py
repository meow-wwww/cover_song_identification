#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import model_unet
import numpy as np
import loss_function
import utils
import evaluate


# In[ ]:


# argparse
save_dir = None
lr = 1e-4
saved_model_path = None
epochs = 0
epochs_finished = 0
num_floor = -1
BATCH_SIZE = 16
overlap = 4
threshold = 0.05

print('--------------ArgParse--------------')

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--save_dir', help='目标存储目录')
parser.add_argument('-s', '--saved_model', help='以训练过的模型')
parser.add_argument('--lr', type=float, help='学习率')
parser.add_argument('-e', '--epochs', type=int, help='有几个epoch')
# parser.add_argument('--epochs_finished', type=int, help='已经完成了几个epoch')
parser.add_argument('-f', '--out_floor', type=int, help='输出在第几层(0，1，2，3)')
parser.add_argument('-g', '--gpu', help='要用的gpu号')
parser.add_argument('-b', '--batch_size', type=int, help='batch_size')
parser.add_argument('-o', '--overlap', type=int, help='切出训练数据时，跳步占全部长度的几分之一')
parser.add_argument('-t','--threshold', type=float, help='生成结果用的阈值')
parser.add_argument('--loss', type=int, help='损失函数')
parser.add_argument('--vt', help='验证/测试集')
parser.add_argument('--label', help='label的类型')
parser.add_argument('--fold', help='fold的类型')

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
    print('Using default lr=1e-4')
else:
    lr = args.lr
    print(f'Using lr from command: {lr}')

assert args.epochs != None, ('请输入epochs数')
epochs = args.epochs

assert args.out_floor != None, ('请输入输出的层数')
assert args.out_floor in [0,1,2,3], ('输入的层数必须为0 1 2 3之一')
num_floor = args.out_floor

assert args.gpu != None, ('请输入要用的gpu号')
device_ids = list(map(lambda x: int(x), args.gpu.split(',')))
print(f'using device_ids: {device_ids}')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f'device count: {torch.cuda.device_count()}')
print(f'current device: {torch.cuda.current_device()}')

device = torch.device("cuda:0")

if args.batch_size == None:
    print(f'using default batch_size {BATCH_SIZE}')
else:
    BATCH_SIZE = args.batch_size
    print(f'using batch_size from command: {BATCH_SIZE}')
    
if args.overlap == None:
    print(f'using default overlap {overlap}')
else:
    overlap = args.overlap
    print(f'using overlap from command: {overlap}')
overlap = 258//overlap

if args.threshold == None:
    print(f'using default threshold {threshold}')
else:
    threshold = args.threshold
    print(f'using threshold from command: {threshold}')
    
if args.loss == 0:
    loss_fn = loss_function.CrossEntropyLoss_Origin()
elif args.loss == 1:
    loss_fn = loss_function.CrossEntropyLoss_for_FA_CE()
elif args.loss == 2:
    loss_fn = loss_function.CrossEntropyLoss_for_FA_CE_VNV()
elif args.loss == 3:
    loss_fn = loss_function.CrossEntropyLoss_for_FA_CE_TF()
elif args.loss == 4:
    loss_fn = loss_function.CrossEntropyLoss_for_FA_CESQ_TF()
else:
    assert False, ('损失函数代号不在范围内')
print(f'Using loss_function: {loss_fn.__class__.__name__}')

assert args.vt[0]!=args.vt[1], ('验证集、测试集不能一样')
assert args.vt[0] in '0123456789', ('验证集index out of range')
assert args.vt[1] in '0123456789', ('测试集index out of range')
valid_fold_index_list = [int(args.vt[0])]
test_fold_index_list = [int(args.vt[1])]
train_fold_index_list = [i for i in range(10) if (i not in valid_fold_index_list and i not in test_fold_index_list)]
print(f'\ttrain: {train_fold_index_list}\n\tvalid: {valid_fold_index_list}\n\ttest: {test_fold_index_list}')

assert args.label in ['origin', 'real_one_hot'], ('label类型不在规定范围内')
label_kind = args.label
print(f'label kind: {label_kind}')

fold_version = args.fold
print(f'fold version: {fold_version}')


# In[ ]:


model = model_unet.UNet()
model = model.cuda()
model = nn.DataParallel(model)
# model = model.to(device)

if saved_model_path != None:
    print(f'loading model from {saved_model_path}...')
    #model = torch.load(saved_model_path)
    model.load_state_dict(torch.load(saved_model_path))
else:
    print('raw model')

loss_fn = loss_fn.cuda()
loss_fn = nn.DataParallel(loss_fn)
# loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
scheduler_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94, verbose=True)
scheduler_stop = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-3, factor=-1, patience=1000)


# In[ ]:


# split data, generate train/test_dataloader

# prepare dataloader
print(f'{datetime.datetime.now()} - Preparing train_dataloader...')
train_dataloader = data_generator.source_index_to_chunk_list(source_list=train_fold_index_list, 
                                                             fold_version=fold_version,
                                                             data_chunks_duration_in_bins=hparams.data_chunks_duration_in_bins,
                                                             data_chunks_overlap_in_bins=overlap,
                                                             label=label_kind)
print(f'{datetime.datetime.now()} - Preparing valid_dataloader...')
valid_dataloader = data_generator.source_index_to_chunk_list(source_list=valid_fold_index_list,
                                                             fold_version=fold_version,
                                                             data_chunks_duration_in_bins=hparams.data_chunks_duration_in_bins,
                                                             data_chunks_overlap_in_bins=overlap,
                                                             label=label_kind)
print(f'{datetime.datetime.now()} - Preparing test_dataloader...')
test_dataloader = data_generator.source_index_to_chunk_list(source_list=test_fold_index_list,
                                                             fold_version=fold_version,
                                                             data_chunks_duration_in_bins=hparams.data_chunks_duration_in_bins,
                                                             data_chunks_overlap_in_bins=overlap,
                                                             label=label_kind)

train_dataloader = DataLoader(train_dataloader, batch_size=BATCH_SIZE*len(device_ids), shuffle=True)
valid_dataloader = DataLoader(valid_dataloader, batch_size=BATCH_SIZE*len(device_ids), shuffle=True)
test_dataloader = DataLoader(test_dataloader, batch_size=BATCH_SIZE*len(device_ids), shuffle=True)


# In[ ]:


# 一个epoch的训练+测试
def train(dataloader, model, loss_fn, optimizer, scheduler, out_floor):
    model.train()
    size = len(dataloader.dataset)
    batch_num = math.ceil(size/dataloader.batch_size)
    loss_total = 0
    
    for batch, (X, y) in tqdm(enumerate(dataloader)): # 每次返回一个batch
        # X, y = X.to(device), y.to(device)
        X, y = X.cuda(device=device_ids[0]), y.cuda(device=device_ids[0])
        # Compute prediction error
        pred = model(X, out_floor)
        
        if out_floor == 0:
            loss_gpus = loss_fn(pred, y, threshold)
        else:
            # downsample y
            y_downsample = utils.downsample(y, out_floor)
            loss_gpus = loss_fn(pred, y_downsample, threshold)
            
        # 对于某些特殊的损失函数：
        if args.loss in [2,3,4]:
            loss = (loss_gpus[0].sum()+loss_gpus[2].sum())/(loss_gpus[1].sum()+loss_gpus[3].sum())
        elif args.loss in [0,1]:
            loss = loss_gpus.sum()
            
        if loss.item() != loss.item():
            print('Train NaN!')
            print(loss_gpus)
            print('====================================================')
            torch.save((X,y), os.path.join(save_dir, 'Xy_nan.pt'))
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_nan.pth'))
            exit()
            
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
        test_loss_v, test_loss_nv = 0, 0
        test_loss_f, test_loss_t = 0, 0
        oa_avg, vr_avg, vfa_avg, rpa_avg, rca_avg = 0, 0, 0, 0, 0
    
    
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            X, y = X.cuda(device=device_ids[0]), y.cuda(device=device_ids[0])
            Xpred = model(X, out_floor)
            Xout = utils.salience_to_output(Xpred.clone().detach(), threshold=threshold)
            
            if out_floor == 0:
                loss = loss_fn(Xpred, y, threshold)
                oa, vr, vfa, rpa, rca = evaluate.evaluate(Xout, y, out_floor)
            else:
                y_downsample = utils.downsample(y, out_floor)
                loss = loss_fn(Xpred, y_downsample, threshold)
                oa, vr, vfa, rpa, rca = evaluate.evaluate(Xout, y_downsample, out_floor)
                
            if args.loss == 2:
                loss_gather = (loss[0].sum()+loss[2].sum())/(loss[1].sum()+loss[3].sum())
                loss_v_gather = loss[0].sum()/loss[1].sum() if loss[1].sum() else loss[1].sum()*0
                loss_nv_gather = loss[2].sum()/loss[3].sum() if loss[3].sum() else loss[3].sum()*0
                test_loss_v += loss_v_gather.item()
                test_loss_nv += loss_nv_gather.item()
            elif args.loss in [3,4]:
                loss_gather = (loss[0].sum()+loss[2].sum())/(loss[1].sum()+loss[3].sum())
                loss_f_gather = loss[0].sum()/loss[1].sum() if loss[1].sum() else loss[1].sum()*0
                loss_t_gather = loss[2].sum()/loss[3].sum() if loss[3].sum() else loss[3].sum()*0
                test_loss_f += loss_f_gather.item()
                test_loss_t += loss_t_gather.item()
            elif args.loss in [0,1]:
                loss_gather = loss.sum()
                
            if loss_gather.item() != loss_gather.item():
                print('Test NaN!')
                print(loss)
                print('====================================================')
                torch.save((X,y), os.path.join(save_dir, 'Xy_nan.pt'))
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_nan.pth'))
                exit()
            
            test_loss += loss_gather.item()
            
            
            oa_avg += oa
            vr_avg += vr
            vfa_avg += vfa
            rpa_avg += rpa
            rca_avg += rca
            
    test_loss /= batch_num # 每张图的loss
    test_loss_v /= batch_num
    test_loss_nv /= batch_num
    test_loss_f /= batch_num
    test_loss_t /= batch_num
    
    oa_avg /= batch_num
    vr_avg /= batch_num
    vfa_avg /= batch_num
    rpa_avg /= batch_num
    rca_avg /= batch_num
    
    print(f"Test Error: Avg loss: {test_loss:.4f} \n")
    print(f"Test OA\t{oa_avg:.4f}\tVR\t{vr_avg:.4f}\tVFA\t{vfa_avg:.4f}\tRPA\t{rpa_avg:.4f}\tRCA\t{rca_avg:.4f}\n")
    
    if args.loss in [0,1,2]:
        return (test_loss, test_loss_v, test_loss_nv), oa_avg, vr_avg, vfa_avg, rpa_avg, rca_avg
    elif args.loss in [3,4]:
        return (test_loss, test_loss_f, test_loss_t), oa_avg, vr_avg, vfa_avg, rpa_avg, rca_avg


# In[ ]:


# # 训练模型🌟
# 多个epoch训练，每个epoch后在验证集上测试

valid_oa_list = []
test_oa_list = []

train_loss_list = []
valid_loss_list = []
test_loss_list = []

valid_loss_v_list = []
valid_loss_nv_list = []
test_loss_v_list = []
test_loss_nv_list = []

valid_loss_f_list = []
valid_loss_t_list = []
test_loss_f_list = []
test_loss_t_list = []

best_oa = 0

_, oa, _, _, _, _ = test(valid_dataloader, model, loss_fn, num_floor)
print(f'验证集原始OA: {oa:.4f}.')
best_oa = oa

for t in range(epochs_finished, epochs_finished+epochs):
    print(f"Epoch {t+1}\n-------------------------------{datetime.datetime.now()}")
    train_loss = train(train_dataloader, model, loss_fn, optimizer, scheduler_decay, num_floor)
    valid_loss, oa, _, _, _, _ = test(valid_dataloader, model, loss_fn, num_floor)
    test_loss, test_oa, _, _, _, _ = test(test_dataloader, model, loss_fn, num_floor)

    train_loss_list.append(train_loss)
    valid_oa_list.append(oa)
    test_oa_list.append(test_oa)
    if args.loss in [0,1]:
        valid_loss_list.append(valid_loss)
        test_loss_list.append(test_loss)
        scheduler_stop.step(valid_loss)
    elif args.loss == 2:
        valid_loss_list.append(valid_loss[0])
        valid_loss_v_list.append(valid_loss[1])
        valid_loss_nv_list.append(valid_loss[2])
        scheduler_stop.step(valid_loss[0])
        test_loss_list.append(test_loss[0])
        test_loss_v_list.append(test_loss[1])
        test_loss_nv_list.append(test_loss[2])
    elif args.loss in [3,4]:
        valid_loss_list.append(valid_loss[0])
        valid_loss_f_list.append(valid_loss[1])
        valid_loss_t_list.append(valid_loss[2])
        scheduler_stop.step(valid_loss[0])
        test_loss_list.append(test_loss[0])
        test_loss_f_list.append(test_loss[1])
        test_loss_t_list.append(test_loss[2])
    
    if optimizer.state_dict()['param_groups'][0]['lr']<=0:
        print(f'Early stop after {t+1} epochs.')
        break
    
    if args.loss in [0,1]:
        plt.plot(range(1,len(train_loss_list)+1), train_loss_list, 'y')
        plt.plot(range(1,len(valid_loss_list)+1), valid_loss_list, 'm')
        plt.plot(range(1,len(test_loss_list)+1), test_loss_list, 'r')
        plt.plot(range(1, len(valid_oa_list)+1), valid_oa_list, 'b')
        plt.plot(range(1, len(test_oa_list)+1), test_oa_list, 'b--')
        plt.legend(['train loss', 'valid loss', 'test_loss','valid acc','test acc'])
    elif args.loss == 2:
        plt.plot(range(1,len(train_loss_list)+1), train_loss_list, 'y')
        plt.plot(range(1,len(valid_loss_list)+1), valid_loss_list, 'm')
        plt.plot(range(1,len(valid_loss_list)+1), valid_loss_v_list, 'm--')
        plt.plot(range(1,len(valid_loss_list)+1), valid_loss_nv_list, 'm-.')
        plt.plot(range(1,len(test_loss_list)+1), test_loss_list, 'r')
        plt.plot(range(1,len(test_loss_list)+1), test_loss_v_list, 'r--')
        plt.plot(range(1,len(test_loss_list)+1), test_loss_nv_list, 'r-.')
        plt.plot(range(1, len(valid_oa_list)+1), valid_oa_list, 'b')
        plt.plot(range(1, len(test_oa_list)+1), test_oa_list, 'b--')
        plt.legend(['train loss', 'valid loss', 'valid loss(V)', 'valid loss(NV)', 'test loss', 'test loss(V)', 'test loss(NV)','valid acc','test acc'])
    elif args.loss in [3,4]:
        plt.plot(range(1,len(train_loss_list)+1), train_loss_list, 'y')
        plt.plot(range(1,len(valid_loss_list)+1), valid_loss_list, 'm')
        plt.plot(range(1,len(valid_loss_list)+1), valid_loss_f_list, 'm--')
        plt.plot(range(1,len(valid_loss_list)+1), valid_loss_t_list, 'm-.')
        plt.plot(range(1,len(test_loss_list)+1), test_loss_list, 'r')
        plt.plot(range(1,len(test_loss_list)+1), test_loss_f_list, 'r--')
        plt.plot(range(1,len(test_loss_list)+1), test_loss_t_list, 'r-.')
        plt.plot(range(1, len(valid_oa_list)+1), valid_oa_list, 'b')
        plt.plot(range(1, len(test_oa_list)+1), test_oa_list, 'b--')
        plt.legend(['train loss', 'valid loss', 'valid loss(F)', 'valid loss(T)', 'test loss', 'test loss(F)', 'test loss(T)','valid acc','test acc'])
    
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    
    # 保存最优模型
    if oa > best_oa:
        with torch.no_grad():
            print(f'[in epoch {t+1}, OA got new best: from {best_oa:.4f} to {oa:.4f}. Will save the model]')
            best_oa = oa
            # torch.save(model, os.path.join(save_dir, f'model_floor{num_floor}_best.pth'))
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_floor{num_floor}_best.pth'))
    
print("Done!")

