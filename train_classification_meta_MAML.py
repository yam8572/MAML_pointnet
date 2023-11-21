"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def cloned_state_dict(model: nn.Module):
    cloned_state_dict = {
        key: val.clone()
        for key, val in model.state_dict().items()
    }
    return cloned_state_dict


from models.pointnet2_cls_msg import get_model as pointnet2_model
from collections import OrderedDict
def maml_single_task_training(model: pointnet2_model, loss_fn,
                              points: torch.Tensor, labels: torch.Tensor,
                              task_lr, device='cuda'):
    # inner loop is set to 1 so you will not see this code
    model.train()
    pred, trans_feat = model(points)    
    loss = loss_fn(pred, labels.long(), trans_feat)
    model.zero_grad(False)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    adapted_state_dict = cloned_state_dict(model)

    for (key, val), grad in zip(model.named_parameters(), grads):
        pp = val - task_lr * grad
        adapted_state_dict[key] = pp

    return adapted_state_dict, loss


def maml_points_do_staff(points):
    points = points.data.numpy()
    points = provider.random_point_dropout(points)
    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
    points = torch.Tensor(points)
    points = points.transpose(2, 1)
    points = points.cuda()
    return points

def maml_main(args):
    """
    參考影片內容：
    https://youtu.be/xoastiYx9JU?si=XPB5xYk82t5p4hb0
    以下內容可能有誤
    將 ModelNet40 的 training set 分成 (10, 10, 10, 10) 組類別，命名為 Task1, Task2, Task3, Task4
    MAML 訓練順序：
    Task 1 Data 0 --> 取得 T1 的 g1 (gradient or loss)
    Task 2 Data 0 --> 取得 T2 的 g2 (gradient or loss)
    Task 3 Data 0 --> 取得 T3 的 g3 (gradient or loss)
    將 g1, g2 and g3 以某算式結合後
    再度做一次 gradient 計算 -->
    取 Data 1, 2, 3... 並重複以上步驟
    
    最終取得某個「學習如何學習」的神經網路。

    為了要驗證是否在未知任務上能否比較快收斂，在 Task 4 上 fine tuning
    fine tuning 後在 ModelNet40 的 testing set 對應 Task 4 做測試
    1. 查看收斂的速度
    2. 微調後的 Model 跟原本 PointNet2 直接訓練 Task 4 相比
    """
    data_root = '/home/g111056119/Documents/7111056119/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled'
    from data_utils.MAML_ModelNetDataLoader40 import split_four_categoris
    datasets = split_four_categoris(data_root, args)
    # dataset_finetune = datasets[3]
    datasets = datasets[0:3]
    task_data_loader = [DataLoader(x, batch_size=4, shuffle=True, num_workers=10, drop_last=True) for x in datasets]
    
    from models.pointnet2_cls_msg import get_model as pointnet2_model
    from models.pointnet2_cls_msg import get_loss as pointnet2_loss_fn
    model = pointnet2_model(10, False).to("cuda")
    criterion = pointnet2_loss_fn()
    meta_lr = 1e-3
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    start_epoch = 0
    loop_count = 1000 // 10
    for epoch in range(start_epoch, 100):
        task_iter = [iter(x) for x in task_data_loader]
        task_lr = 1e-1
        for lc in range(loop_count):
            task_model_dict = []
            task_model_loss = []
            task_points = []
            task_labels = []
            for (i, ti) in enumerate(task_iter):
                try:
                    points, labels = next(ti)
                except StopIteration:
                    task_iter[i] = iter(task_data_loader[i])
                    points, labels = next(task_iter[i])
                points = maml_points_do_staff(points)
                labels = labels.cuda()
                task_points.append(points)
                task_labels.append(labels)

                task_state_dict, task_loss = maml_single_task_training(model, criterion, points, labels, task_lr)
                task_model_dict.append(task_state_dict)
                task_model_loss.append(task_loss)

            # for i in range(len(task_labels)):
            #     # Just a bunch of weird code
            #     tmp_model = pointnet2_model(10, model.normal_channel).to("cuda")
            #     tmp_model.load_state_dict(task_model_dict[i])
            #     pred, trans_feat = tmp_model(task_points[i])
            #     loss = criterion(pred, task_labels[i].long(), trans_feat)
            #     task_model_loss.append(loss)
                
            meta_loss = torch.stack(task_model_loss).mean()
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            loss_cpu = meta_loss.item()
            print(epoch, lc, "Meta Loss:", loss_cpu)
            pass
    pass
    torch.save(model.state_dict(), "good_model.pth")

if __name__ == '__main__':
    args = parse_args()
    # main(args)
    maml_main(args)
