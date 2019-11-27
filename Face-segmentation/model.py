# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Conv_BN_Relu,DownSample,UpSample,Output_Layer_Conv

class Unet(nn.Module):
    def __init__(self,in_channels,n_classes,bilinear=True):
        super(Unet,self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.conv1=Conv_BN_Relu(self.in_channels,64)
      
        self.down1=DownSample(in_channels=64,out_channels=128)
        self.down2=DownSample(in_channels=128,out_channels=256)
        self.down3=DownSample(in_channels=256,out_channels=512)
        self.down4=DownSample(in_channels=512,out_channels=512)
        
        self.up1=UpSample(in_channels=1024,out_channels=256)
        self.up2=UpSample(in_channels=512,out_channels=128)
        self.up3=UpSample(in_channels=256,out_channels=64)
        self.up4=UpSample(in_channels=128,out_channels=64)
        
        self.output=Output_Layer_Conv(64,self.n_classes)
        
        
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        
        x=self.up1(x5,x4)
        x=self.up2(x,x3)
        x=self.up3(x,x2)
        x=self.up4(x,x1)        
        logits=self.output(x)
        
        return logits
        #return F.log_softmax(logits, dim=1)
    
        """
        if self.n_classes > 1:
            return F.log_softmax(logits, dim=1)
        else:
            return torch.sigmoid(logits)
        
        """










