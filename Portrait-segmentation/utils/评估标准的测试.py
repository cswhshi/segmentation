# -*- coding: utf-8 -*-
import sys

from metrics import Evaluator
from UNet import Unet
import torch
import torch.nn as nn



#%%

"""
一、获取用于测试评估标准函数的数据

"""
def get_test_data():
    from MyDataset import VOCSegmentation
    
    from torch.utils.data import DataLoader
    import argparse
    
    base_dir="/humanparsing/"
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    
    voc_train = VOCSegmentation(args, base_dir=base_dir,mode='train')
    dataloader = DataLoader(voc_train, batch_size=2, shuffle=True, num_workers=0)
        
    for ii, sample in enumerate(dataloader):
        logit = sample["image"]
        target = sample["label"]
        
        print(sample["image"].size())#torch.Size([5, 3, 513, 513])
        print(sample["label"].size())#torch.Size([5, 513, 513])
        break
    return logit,target




images,target = get_test_data()

model = Unet(in_channels=3,n_classes=18)
model.eval()
logit = model(images)
print(target.shape)
print(logit.shape)

#torch.Size([2, 513, 513])
#torch.Size([2, 18, 513, 513])


#%%


import numpy as np
pred = logit.data.cpu().numpy()
target = target.cpu().numpy()
pred = np.argmax(pred, axis=1)

print(target.shape)   #(2, 513, 513)
print(pred.shape)    #(2, 513, 513)

gt_image, pre_image = target, target  
evaluator = Evaluator(18)
evaluator.add_batch(gt_image, pre_image.astype('int'))

Acc = evaluator.Pixel_Accuracy()
Acc_class = evaluator.Pixel_Accuracy_Class()
mIoU = evaluator.Mean_Intersection_over_Union()
FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))








