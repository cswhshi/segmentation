# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
convolution => [BN] => ReLU 

"""
class Conv_BN_Relu(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv_BN_Relu, self).__init__()
        
        self.conv_bn_relu=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels,kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
    def forward(self,x):
        return self.conv_bn_relu(x)

"""

Down sampling

"""
class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownSample,self).__init__()
        self.maxpool_conv=nn.Sequential(
                nn.MaxPool2d(2),
                Conv_BN_Relu(in_channels,out_channels))
    def forward(self,x):
        return self.maxpool_conv(x)
"""

Up sampling

"""

  
class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        super(UpSample,self).__init__()
        if(bilinear):
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv=Conv_BN_Relu(in_channels,out_channels)
    def forward(self,x1,x2):
        x1=self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Output_Layer_Conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Output_Layer_Conv,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
    
