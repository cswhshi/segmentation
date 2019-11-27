# -*- coding: utf-8 -*-
import sys
sys.path.append("F:\\lingji2019\\Seg\\Person_Seg\\model\\")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d#同步不同GPU的bn    #https://www.zhihu.com/question/59321480/answer/198077371

class Decoder(nn.Module):
    def __init__(self,num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()
        


    def forward(self, x, low_level_feat):
        #为了防止encoder得到的高级特征被弱化，先采用1x1卷积对低级特征进行降维（paper中输出维度为48）
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        #插值方式进行上采用
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        #横向连接，组合底层和顶层特征
        x = torch.cat((x, low_level_feat), dim=1)
        
        #两个特征concat后，再采用3x3卷积进一步融合特征，最后再双线性插值得到与原始图片相同大小的分割预测。
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
        




