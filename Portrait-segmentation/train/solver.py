# -*- coding: utf-8 -*-
import sys


from deeplab import DeepLab
from mypath import Path
from metrics import Evaluator
from lr_scheduler import LR_Scheduler
from MyDataset import VOCSegmentation
from calculate_weights import calculate_weigths_labels
from SegmentationLosses import SegmentationLosses

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler


"""
1、定义参数

"""

base_dir="/humanparsing/"
model_save="/checkpoint/"

lr=0.001
n_classes=18
in_channels=3
momentum = 0.9
weight_decay = 0.0005
nesterov = False
use_balanced_weights = False



dataset = "humanparsing"
loss_type="ce"   #choices=['ce', 'focal']

#lr_scheduler = "poly"    #choices=['poly', 'step', 'cos']



epochs = 20
batch_size=2
cuda = torch.cuda.is_available()



"""
2、定义数据加载器

"""



parser = argparse.ArgumentParser()
args = parser.parse_args()

args.base_size = 513
args.crop_size = 513

voc_train = VOCSegmentation(args, base_dir=base_dir,mode='train')
dataloader = DataLoader(voc_train, batch_size=batch_size, shuffle=True, num_workers=0)


voc_val = VOCSegmentation(args, base_dir=base_dir,mode='test')
val_dataloader = DataLoader(voc_val, batch_size=batch_size, shuffle=True, num_workers=0)




"""
3、定义模型

"""

model = DeepLab(num_classes=18,backbone="mobilenet",output_stride=8,sync_bn=False,freeze_bn=False)


"""
4、定义优化器

"""

train_params = model.parameters()
optimizer = torch.optim.SGD(train_params,lr=0.003, momentum=0.9,weight_decay=5e-4, nesterov=False)

scheduler = lr_scheduler.StepLR(optimizer, 100, gamma=0.5, last_epoch=-1)




"""
5.1、定义损失函数

"""
#是否使用类平衡权重


#如果有类平衡权重，则直接加载就行了，否则需要计算一下各个类的平衡数据
classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
if os.path.isfile(classes_weights_path):
    weight = np.load(classes_weights_path)
else:
    weight = calculate_weigths_labels(dataset, dataloader, 18)
weight = torch.from_numpy(weight.astype(np.float32))    


"""
5.2、定义损失函数

"""

criterion = SegmentationLosses(weight=weight, cuda=False).build_loss(mode="ce")

"""
6、定义评价指标

"""

evaluator = Evaluator(18)

"""
7、定义学习率调度器

"""

scheduler = LR_Scheduler(lr_scheduler, lr, epochs, len(dataloader))

"""
8、使用预定义模型，并判断是否使用cuda

if os.path.isdir(model_save):
    try:
        checkpoint = torch.load(model_save+'model.t7', map_location='cpu')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found model.t7')
else:
    start_epoch = 0
    print('===> Start from scratch')

if cuda:
    model.cuda()
    cudnn.benchmark = True
"""




#%%
"""

9、训练

"""    


def training(dataloader,epoch,biact_size):
    train_loss = 0
    model.train()
    tbar = tqdm(dataloader)
    num_img_tr = len(dataloader)
    k=0
    for i,sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        if cuda:
            image, target = image.cuda(), target.cuda()
            
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        k=k+1
        if(k%50==0):
            print('train/total_loss_iter', train_loss/k, i + num_img_tr * epoch)
        break
    print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
    print('Loss: ' ,train_loss/num_img_tr)
    
    if (epoch % 2 == 0 and epoch!=0):
        print('===> Saving models...')
        state = {
            'state': model.state_dict(),
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, model_save+'model.t7')


#%%
    
"""
9、模型验证

"""  
def validation(val_dataloader,epoch,batch_size):
    model.eval()
    tbar = tqdm(val_dataloader, desc='\r')
    test_loss = 0.0
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        if cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)
        loss = criterion(output, target)
        test_loss += loss.item()
        tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)
    
    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    
    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, i * batch_size + image.data.shape[0]))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % test_loss)


for epoch in range(epochs):
    training(dataloader,epoch,batch_size)
    validation(val_dataloader,epoch,batch_size)
    scheduler.step()    






