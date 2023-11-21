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
from torch.utils.data import DataLoader,random_split

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

def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval() # 測試時不啟用 BatchNormalization 和 Dropout
    class_acc = np.zeros((num_class, 3)) # (40,3)

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        # if not args.use_cpu:
        #     points, target = points.cuda(), target.cuda()

        # points = points.transpose(2, 1)
        target = target.cuda()
        # 更改成和 train 一樣的點預處理
        points = maml_points_do_staff(points)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()
        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # 個別類別 accuracy
    class_acc = np.mean(class_acc[:, 2])
    # 不分類別 accuracy
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


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

from datetime import datetime
import timeit
def time():
    # Get the current time
    current_time = datetime.now()
    # Format the time as a string
    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    return current_time

def maml_points_do_staff(points):
    points = points.data.numpy()
    points = provider.random_point_dropout(points)
    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
    points = torch.Tensor(points)
    points = points.transpose(2, 1)
    points = points.cuda()
    return points

def finetune_train(model: pointnet2_model, points, label):
    pass

def finetune_train(args,train_dataloader):
    
    '''MODEL LOADING'''
    # Load good model 10 >> 10類 False >> normal_channel=args.use_normals
    model = pointnet2_model(num_class=10, normal_channel=False).to("cuda")
    model_path = '/home/g111056119/Documents/7111056119/Pointnet_Pointnet2_pytorch/good_model.pth'
    model.load_state_dict(torch.load(model_path))
    from models.pointnet2_cls_msg import get_loss as pointnet2_loss_fn
    criterion = pointnet2_loss_fn()
    model.apply(inplace_relu)
    # print(model.parameters())

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    if not args.use_cpu:
        model = model.cuda()
        criterion = criterion.cuda()

    """ TRAIN START """
    start_time=time()
    print(f"start training time={start_time}")
    for epoch in range(1000):
        # print('Epoch (%d/%s):' % (epoch + 1, 100))
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, 1000))

        mean_correct = []
        model = model.train()
        scheduler.step()
        for (points, target) in tqdm(train_dataloader):
            optimizer.zero_grad()
            # 點雲數據預處理
            points = maml_points_do_staff(points)
            points, target = points.cuda(), target.cuda() # target shape B

            pred, trans_feat = model(points)    
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0])) # 分母為B
            loss.backward() # 反向傳播 (梯度計算)
            optimizer.step() # 更新權重
            global_step += 1
            pass
        train_instance_acc = np.mean(mean_correct)
        print('Train Instance Accuracy: %f' % train_instance_acc)
        global_epoch += 1
        continue
        with torch.no_grad():
            instance_acc, class_acc = test(model.eval(), test_dataloader, num_class=10)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            print('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                savepath = 'finetune_good_model.pth'
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
        pass
    # torch.save(model.state_dict(), "finetune_good_model.pth")
    end_time=time()
    print(f"end training time={end_time}")
    pass

    # from test_classification import test as calaccuracy
    # '''test query evaluate adapt to new task haven't seen 評估finetune完後 適應到新類別結果'''
    # '''FINETUNE MODEL LOADING'''
    # finetine_model = pointnet2_model(num_class=10, normal_channel=False).to("cuda")
    # finetine_model_path = '/home/g111056119/Documents/7111056119/Pointnet_Pointnet2_pytorch/finetune_good_model_old.pth'
    # finetine_model.load_state_dict(torch.load(finetine_model_path))
    # print("評估finetune完後 適應到新類別結果")
    # with torch.no_grad():
    #     instance_acc, class_acc = test(finetine_model.eval(), test_dataloader, vote_num=3, num_class=10)
    #     print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

def test_eval(args,test_dataloader):
    from test_classification import test as calaccuracy
    '''test query evaluate adapt to new task haven't seen 評估finetune完後 適應到新類別結果'''
    '''FINETUNE MODEL LOADING'''
    finetine_model = pointnet2_model(num_class=10, normal_channel=False).to("cuda")
    finetine_model_path = '/home/g111056119/Documents/7111056119/Pointnet_Pointnet2_pytorch/log/finetune_good_model_old.pth'
    finetine_model.load_state_dict(torch.load(finetine_model_path))
    print("評估finetune完後 適應到新類別結果")
    with torch.no_grad():
        instance_acc, class_acc = test(finetine_model.eval(), test_dataloader, vote_num=3, num_class=10)
        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

def LoadData(args,train_size:float):
    '''DATA LOADING'''
    data_root = '/home/g111056119/Documents/7111056119/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled'
    from data_utils.MAML_ModelNetDataLoader40 import split_four_categoris
    datasets = split_four_categoris(data_root, args)
    # test set
    dataset_finetune = datasets[3]
    
    # Define the sizes for the training and testing sets
    support_set_size = int(train_size* len(dataset_finetune))  # 50% for training
    query_set_size = len(dataset_finetune) - support_set_size   # 50% for testing

    # test database 再分 train(support set) >> fine tune & test(query set) >> cal accuracy
    support_dataset, query_dataset = random_split(dataset_finetune, [support_set_size, query_set_size])

    # datasets = datasets[0:3]
    train_dataloader = DataLoader(support_dataset,batch_size=2,drop_last=True)
    test_dataloader = DataLoader(query_dataset, batch_size=2)

    return train_dataloader,test_dataloader


if __name__ == '__main__':
    args = parse_args()
    train_dataloader,test_dataloader = LoadData(args,train_size=0.5)
    finetune_train(args,train_dataloader)
    # 使用 timeit 測量代碼執行一次時間 number=1
    # execution_time = timeit.timeit(finetune_train(args), number=1)
    # print(f"訓練時間: {execution_time} 秒")
