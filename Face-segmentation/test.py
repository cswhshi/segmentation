# -*- coding: utf-8 -*-

import torch


from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
#from unet import Unet
from mydataset import MyDataset
from torch.utils import data
from model import Unet


image_path="/dataset/images/"
label_path="/dataset/labels/"
val_data="/dataset/val.txt"


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])


model = Unet(3,1)
"""
加载模型
"""
try:
    checkpoint = torch.load('/weights/weights.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    #start_epoch = checkpoint['epoch']
    #print("start_epoch:",start_epoch)
    print('===> Load last checkpoint data')
except FileNotFoundError:
    print('Can\'t found weight.pth')

cuda=torch.cuda.is_available()
if(cuda):
    model.cuda()


batch_size = 1
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())



dataset = MyDataset(image_path,label_path,val_data,transform=x_transforms, target_transform=y_transforms)
data_loader = data.DataLoader(dataset,  batch_size=1, shuffle=True, num_workers=0)




def clean(img):
    img[img <0 ] = 0
    img[img >0 ] = 1
    return img


import matplotlib.pyplot as plt
model.eval()


i=0
with torch.no_grad():
    for x, target in data_loader:
        y=model(x)
        img_y=torch.squeeze(y).numpy()
        img_y=clean(img_y)
        print("预测的掩码：")
        plt.imshow(img_y)
        plt.pause(0.01)
        plt.show()
        print("=====================")
        print("真实的掩码：")
        target=torch.squeeze(target).numpy()
        plt.imshow(target)
        plt.pause(0.01)
        plt.show()
        print("===============================================================================")
        #break
        i=i+1
        if(i==4):
            break
    
    
    













