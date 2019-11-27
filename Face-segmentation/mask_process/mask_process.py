# -*- coding: utf-8 -*-

import sys
import os
from PIL import Image
#import config as cfg
import matplotlib.pyplot as plt
import numpy

#获取对应图片的mask路径列表
label_path="./experiment/dataset/labels/232194_1/"

labels_path_list = os.listdir(label_path)
def clean(img):
    img[img <0 ] = 0
    img[img >0 ] = 1
    return img
i=1;
for name in labels_path_list:
    if("00" in name[-6:-1]):
        label_image = Image.open(label_path+name)
        label_image = numpy.array(label_image)          # array is a numpy array
        res=label_image
        #plt.imshow(res)
        #plt.show()
        #break
    else:
        label_image = Image.open(label_path+name)
        label_image = numpy.array(label_image) 
        #label_image=clean(label_image)
        plt.imshow(label_image)
        plt.show()
        break
        label_image=label_image+i*2
        res+=label_image
        i=i+1
        break
    
    
#plt.imshow(res)
#plt.show()
      








i=0
for name in labels_path_list:
    print(name[-6:-1])
    if("00" in name[-6:-1] or "10" in name[-6:-1]):
        pass
    elif("01" in name[-6:-1]):
        label_image = Image.open(label_path+name)
        label_image = numpy.array(label_image)          # array is a numpy array 
        res=label_image
    else:
        label_image = Image.open(label_path+name)
        label_image = numpy.array(label_image) 
        res+=label_image
        
plt.imshow(res)
plt.show()
    