# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        """
        1、nn.CrossEntropyLoss()的相关参数
        """
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        
        """
        2、是否对batch求平均损失
        """
        self.batch_average = batch_average
        
        """
        3、是否使用GPU进行训练
        """
        self.cuda = cuda
    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss


    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
    
    def build_loss(self,mode='ce'):
        """
        函数的作用是选择那种类型的损失函数：
        目前有两个选择，1个是CrossEntropyLoss，另外一个是FocalLoss
        分别对应的输入是mode == 'ce' 以及 mode == 'focal'
        """
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

if __name__ == "__main__":
    loss = SegmentationLosses()
    a = torch.rand(1, 3, 7, 7)#.cuda()
    b = torch.rand(1, 7, 7)#.cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    
    

            