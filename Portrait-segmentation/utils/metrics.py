import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # return all class overall pixel accuracy
		"""
		1、像素准确率(PA)
		像素准确率是所有分类正确的像素数占像素总数的比例。利用混淆矩阵计算则为（对角线元素之和除以矩阵所有元素之和）：
		"""
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        #返回平均精准率
        #注意，np.nanmean会忽略为空的类别
		"""
		2、平均像素准确率(MPA)
			平均像素准确率是分别计算每个类别分类正确的像素数占所有预测为该类别像素数的比例，即精确率，然后累加求平均。
		利用混淆矩阵计算公式为(每一类的精确率Pi都等于对角线上的TP除以对应类别的像素数) ：
		"""
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        """
		3、平均交并比：
		平均交并比是对每一类预测的结果和真实值的交集与并集的比值求和平均的结果
		
		"""
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
		"""
		4、 频权交并比(FWloU)
		　频权交并比是根据每一类出现的频率设置权重，权重乘以每一类的IoU并进行求和，
		利用混淆矩阵计算：每个类别的真实数目为TP+FN，总数为TP+FP+TN+FN，其中每一类的权重和其IoU的乘积计算公式如下，在将所有类别的求和即可：
		"""
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)#计算各个类别的数量矩阵
        
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':
    imgPredict = np.array([0, 0, 0, 0, 0, 0])
    imgLabel = np.array([0, 0, 1, 1, 2, 2])
    metric = Evaluator(3)
    
    metric.add_batch(imgPredict, imgLabel)
    
    acc = metric.Pixel_Accuracy()
    Acc_class=metric.Pixel_Accuracy_Class()
    mIoU = metric.Mean_Intersection_over_Union()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print(acc,Acc_class,mIoU,FWIoU)

np.nanmean([1,2,3,4,5,6,np.nan])