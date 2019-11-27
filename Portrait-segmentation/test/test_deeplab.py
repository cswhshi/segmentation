# -*- coding: utf-8 -*-
import sys

#from UNet import Unet
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


import custom_transforms as tr
from torchvision.transforms import transforms

"""
1、定义参数

"""

base_dir="/humanparsing/"
model_path="/checkpoint/"

dataset = "humanparsing"
cuda = torch.cuda.is_available()


category=['background   0    背景',
        'hat            1    帽子',
        'hair           2    头发',
        'sunglass       3    太阳镜',
        'upper-clothes  4    上衣',
        'skirt          5    裙子',
        'pants          6    裤子',
        'dress          7    礼服',
        'belt           8    皮带',
        'left-shoe      9    左鞋',
        'right-shoe     10   右鞋',
        'face           11   脸蛋',
        'left-leg       12   左腿',
        'right-leg      13   右腿',
        'left-arm       14   左臂',
        'right-arm      15   右臂',
        'bag            16   袋子',
        'scarf          17   围巾']

"""
2、定义数据加载器

"""



parser = argparse.ArgumentParser()
args = parser.parse_args()

args.base_size = 513
args.crop_size = 513

voc_train = VOCSegmentation(args, base_dir=base_dir,mode='train')
dataloader = DataLoader(voc_train, batch_size=2, shuffle=True, num_workers=0)


voc_val = VOCSegmentation(args, base_dir=base_dir,mode='test')
val_dataloader = DataLoader(voc_val, batch_size=2, shuffle=True, num_workers=0)



"""
3、定义模型

"""
#model = Unet(in_channels=3,n_classes=18)
model = DeepLab(num_classes=18,backbone="resnet",output_stride=8,sync_bn=False,freeze_bn=False)


"""
8、使用预定义模型，并判断是否使用cuda
"""


if os.path.isdir(model_path):
    try:
        checkpoint = torch.load(model_path+'model_deeplab.t7', map_location='cpu')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found model_deeplab.t7')
else:
    start_epoch = 0
    print('===> Start from scratch')

if cuda:
    model.cuda()
    cudnn.benchmark = True


#%%
parser = argparse.ArgumentParser()
args = parser.parse_args()

args.base_size = 513
args.crop_size = 513 

composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),#随机水平翻转
            tr.RandomScaleCrop(base_size=args.base_size, crop_size=args.crop_size),#随机尺寸裁剪
            tr.RandomGaussianBlur(),#随机高斯模糊
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),#归一化
            tr.ToTensor()])


    
#%%    
import matplotlib.pyplot as plt
import numpy as np
# dataset.utils import decode_segmap


tbar = tqdm(dataloader)
num_img_tr = len(dataloader)
for epoch in range(0, 10):
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        print(image.shape,target.shape)
        
        """
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        """
        #scheduler(optimizer, i, epoch, best_pred)
        
        #optimizer.zero_grad()
        target_0=torch.squeeze(target[0]).detach().numpy()
        plt.imshow(target_0)
        plt.show()
        output = model(image)
        print(output.shape)
        print("==========================")
        break
    break
    


#%%
    
def clean_a(image):
    # clean the pixels which < 0 or >255
    #image[image > 255] = 255
    image[(image < 2) & (image>-2)] = 255
    image[(image >= 2) & (image<=-2)]=0
    
    image = image.astype(np.uint8)
    return image



def clean_res(image):
    # clean the pixels which < 0 or >255
    #image[image > 255] = 255
    image[image < 200] = 0
    image[image >= 200]=1
    
    image = image.astype(np.uint8)
    return image
target_0=torch.squeeze(target[1]).detach().numpy()

aa=output

plt.imshow(target_0)
plt.show()

pre_y=torch.squeeze(aa[1]).detach().numpy()       

a=pre_y[0]

plt.imshow(a)
plt.show()
a_res=clean_a(a)
res=clean_res(a_res)
plt.imshow(res)
plt.show()   



img=image.numpy()

img_tmp = np.transpose(img[1], axes=[1, 2, 0])
img_tmp *= (0.229, 0.224, 0.225)
img_tmp += (0.485, 0.456, 0.406)
img_tmp *= 255.0
img_tmp = img_tmp.astype(np.uint8)
plt.imshow(img_tmp)
plt.show()

#%%
plt.imshow(res)
plt.show()  

plt.imshow(img_tmp)
plt.show()


plt.imshow(res)
plt.imshow(img_tmp,alpha=0.5)
 
plt.show()


img_tmp[:,:,0]=img_tmp[:,:,0]*res
img_tmp[:,:,1]=img_tmp[:,:,1]*res
img_tmp[:,:,2]=img_tmp[:,:,2]*res

plt.imshow(img_tmp)
plt.show()



















