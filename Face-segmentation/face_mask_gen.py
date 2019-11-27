# -*- coding: utf-8 -*-
import torch

from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
#from unet import Unet
from mydataset import MyDataset
from torch.utils import data
import cv2
from model import Unet


import numpy as np

from PIL import Image  
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
loader = transforms.Compose([transforms.ToTensor()])
model = Unet(3,1)
"""
加载模型
"""
try:
    checkpoint = torch.load('/weights/face_weights.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    #start_epoch = checkpoint['epoch']
    #print("start_epoch:",start_epoch)
    print('===> Load last checkpoint data')
except FileNotFoundError:
    print('Can\'t found weight.pth')

cuda=torch.cuda.is_available()
if(cuda):
    model.cuda()
    
    
def clean(img):
    img[img <0 ] = 0
    img[img >0 ] = 1
    return img



model.eval()

path="1.jpg"

img = cv2.imread(path)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


#%%
image = Image.fromarray(img)
image = loader(image).unsqueeze(0)

y=model(image)

img_y=torch.squeeze(y).detach().numpy()
img_y=clean(img_y)
print("预测的掩码：")
plt.imshow(img_y)
plt.pause(0.01)
plt.show()



img[:,:,0]=img[:,:,0]*img_y
img[:,:,1]=img[:,:,1]*img_y
img[:,:,2]=img[:,:,2]*img_y

output_im=img
output_im=output_im.astype(dtype='uint8')
plt.imshow(output_im)
plt.pause(0.01)
plt.show()



#%%
import os

def mask_do_image(path):
    img = cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    image = Image.fromarray(img)
    image = loader(image).unsqueeze(0)
    
    y=model(image)
    img_y=torch.squeeze(y).detach().numpy()
    img_y=clean(img_y)
    img[:,:,0]=img[:,:,0]*img_y
    img[:,:,1]=img[:,:,1]*img_y
    img[:,:,2]=img[:,:,2]*img_y
    
    output_im=img
    output_im=output_im.astype(dtype='uint8')
    return output_im
    #plt.imshow(output_im)
    #plt.pause(0.01)
    #plt.show()







def run(path,save_path):
    images=os.listdir(path)
    for name in images:
        print(path+name)
        output_im=mask_do_image(path+name)
        cv2.imwrite(save_path+name, output_im)


