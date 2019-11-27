# -*- coding: utf-8 -*-

from UNet import Unet
import torch
import torch.nn as nn



#%%
"""
一、获取用于测试损失函数的数据

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
print(logit.shape)


    
#images.Size([2, 3, 513, 513])
#target：torch.Size([2, 513, 513])
#logit：torch.Size([2, 18, 513, 513])
#%%
"""
测试交叉熵损失函数

#images：torch.Size([5, 3, 513, 513])
#target：torch.Size([5, 513, 513])
#logit：torch.Size([5, 18, 513, 513])
结论：由此可见，分割任务中多个类别的交叉上输入形式是这样的:
    images：的维度要构造成（batch_size,channel,h,w）
    target：的维度要构造成（batch_size,h,w）,也就是说里面每个类别用一个离散变量表示
    logit：的维度要构造成（batch_size,classes_num,h,w）
"""

n, c, h, w = logit.size() #(2, 18, 513, 513)
criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255,size_average=True)
loss = criterion(logit, target.long())
print(loss)
batch_average=loss/n
print(batch_average)

"""
输出：
tensor(2.8247, grad_fn=<NllLoss2DBackward>)
tensor(1.4124, grad_fn=<DivBackward0>)

"""
#%%

"""
测试focal_loss损失函数

"""
cuda=False
alpha=0.5
gamma=2
n, c, h, w = logit.size()
criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255,size_average=True)
if cuda:
    criterion = criterion.cuda()
    
logpt = -criterion(logit, target.long())

pt = torch.exp(logpt)
if alpha is not None:
    logpt *= alpha
loss = -((1 - pt) ** gamma) * logpt

batch_average = loss / n
print(loss)

print(batch_average)
"""
输出为：
tensor(1.2498, grad_fn=<MulBackward0>)
tensor(0.6249, grad_fn=<DivBackward0>)

"""

#%%








