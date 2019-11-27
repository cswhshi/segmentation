# -*- coding: utf-8 -*-

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import custom_transforms as tr
from seg_map_utils import decode_segmap


base_dir="/humanparsing/"


#%%
"""
#图片所在目录
image_dir = os.path.join(base_dir, 'JPEGImages')
print(image_dir)

#掩码所在目录
cat_dir = os.path.join(base_dir, 'SegmentationClassAug')
print(cat_dir)

im_ids = []
images = []
categories = []
mode="train"

with open(os.path.join(base_dir,mode+'.txt'),'r') as f:
    lines=f.readlines()
    for ii,line in enumerate(lines):
        image=os.path.join(image_dir,line.split(".jpg")[0]+".jpg")
        print(image)
        cat=os.path.join(cat_dir,line.split(".jpg")[0]+".png")
        print(cat)
        assert os.path.isfile(image)
        assert os.path.isfile(cat)
        im_ids.append(line)
        images.append(image)
        categories.append(cat)
        break
print("===========")
print(im_ids)
print(images)
print(categories)
assert (len(images) == len(categories))
# Display stats
print('Number of images in {}: {:d}'.format(mode, len(images)))


img=Image.open(images[0]).convert("RGB")
img
target=Image.open(categories[0])
target
"""

#%%

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 18   #18个类别
    
    def __init__(self,args,base_dir,mode="train"):
        super().__init__()
        self._base_dir=base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')             #图片所在目录
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')     #类别所在目录
        
        #self.im_ids = []
        self.images = []    #图片路径列表
        self.categories = []#图片mask路径列表
        self.mode=mode   #当前的数据集模式，如果是train，表示是训练集，选择train.txt去解析并读取图片名称，然会保存到images和categories列表
        self.args=args
        
        with open(os.path.join(self._base_dir, self.mode + '.txt'), "r") as f:#读取训练集train.txt
            lines = f.readlines()
            
            for ii, line in enumerate(lines):#解析train.txt的每一个图片名称，并加上路径，分别保存到图片路径列表以及图片mask路径列表。\
                _image = os.path.join(self._image_dir, line.split(".jpg")[0] + ".jpg")    
                _cat = os.path.join(self._cat_dir, line.split(".jpg")[0] + ".png")
                #print(_image)
                #print(_cat)
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                #self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
            
        assert (len(self.images) == len(self.categories))
        
        # Display stats
        print('Number of images in {}: {:d}'.format(self.mode, len(self.images)))#显示当前选择的模式下的数据集有多少张
        
    def _make_img_gt_point_pair(self, index):
        """
        此函数的作用是根据传入的index索引，然后根据索引去获取对应图片以及mask的路径，然后用Image读取图片并返回
        """
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        return _img, _target
    
    def __getitem__(self, index):
        """
        返回与指定键想关联的值
        """
        _img, _target = self._make_img_gt_point_pair(index)  #获取图片以及mask矩阵
        sample = {'image': _img, 'label': _target} #key-value的形式将img和target重新组织

        
        if self.mode == "train":#进行数据增强,并返回处理只会的sample
            return self.transform_tr(sample)
        elif self.mode == 'test':#进行数据增强,并返回处理只会的sample
            return self.transform_val(sample)
    
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),#随机水平翻转
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),#随机尺寸裁剪
            tr.RandomGaussianBlur(),#随机高斯模糊
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),#归一化
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
    
    
    def __len__(self):
        return len(self.images)
    def __str__(self):
        return 'VOC2012(split=' + str(self.mode) + ')'


#%%
        
"""

结论：
    image的维度为：torch.Size([5, 3, 513, 513])
    target的维度为：(5, 513, 513)
    
"""   
if __name__ == '__main__':    
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 600
    args.crop_size = 400
    
    voc_train = VOCSegmentation(args, base_dir=base_dir,mode='train')
    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)
    
    for ii, sample in enumerate(dataloader):
        print(ii)
        print(sample["image"].size())#torch.Size([5, 3, 513, 513])
        print(sample["image"].size()[0])  #5,表示有5个图片
        print(sample["label"].size())#torch.Size([5, 513, 513])
        print("=========")
        print("循环一个batch的所有图片：")
        img=sample['image'].numpy()
        gt=sample['label'].numpy()
        print(img.shape)                #(5, 3, 513, 513)
        print(gt.shape)                 ###(5, 513, 513)
        for jj in range(sample["image"].size()[0]):
            print("循环显示图片：")
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.imshow(img_tmp)
            plt.show()
            print("循环显示mask:")
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')#赋给不同的颜色
            plt.imshow(segmap)
            plt.show()
            break
        break
    
    


