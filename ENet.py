# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

"""
初始阶段模块
"""
class InitialBlock(nn.Module):
    def __init__(self,in_channels,out_channels,bias=False,relu=True):
        super(InitialBlock, self).__init__()
        if(relu):
            activation=nn.ReLU
        else:
            activation=nn.PReLU
        #主分支
        self.main_branch=nn.Conv2d(in_channels,out_channels-3,kernel_size=3,stride=2,padding=1,bias=bias)
        #分支
        self.ext_branch=nn.MaxPool2d(3,stride=2,padding=1)
        self.bn=nn.BatchNorm2d(out_channels)
        self.out_relu=activation()
        
    def forward(self,x):
        x1=self.main_branch(x)
        x2=self.ext_branch(x)
        out=torch.cat((x1,x2),1)
        out=self.bn(out)
        
        return self.out_relu(out)

"""
不带下采样的Bottleneck

"""
class Bottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super(Bottleneck, self).__init__()
        """
        检查internal_ratio参数范围
        """
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))
        internal_channels=channels//internal_ratio
        
        if(relu):
            activation=nn.ReLU
        else:
            activation=nn.PReLU
        
        """
        分支上的第一个1x1
        """
        self.ext_conv1=nn.Sequential(
                nn.Conv2d(channels,internal_channels,kernel_size=1,stride=1,bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation())
        """
        是否使用非对称卷积
        """
        if asymmetric:
            self.ext_conv2=nn.Sequential(
                    nn.Conv2d(
                            internal_channels,
                            internal_channels,
                            kernel_size=(kernel_size,1),
                            stride=1,
                            padding=(padding,0),
                            dilation=dilation,
                            bias=bias),
                    nn.BatchNorm2d(internal_channels),
                    activation(),
                    nn.Conv2d(
                            internal_channels,
                            internal_channels,
                            kernel_size=(1,kernel_size),
                            stride=1,
                            padding=(0,padding),
                            dilation=dilation,
                            bias=bias),
                    nn.BatchNorm2d(internal_channels),
                    activation())
        else:
            self.ext_conv2=nn.Sequential(
                    nn.Conv2d(
                            internal_channels,
                            internal_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                            dilation=dilation,
                            bias=bias),
                    nn.BatchNorm2d(internal_channels),
                    activation())
        """
        分支上的最好一个1x1
        """
        self.ext_conv3=nn.Sequential(
                nn.Conv2d(internal_channels,channels,kernel_size=1,stride=1,bias=bias),
                nn.BatchNorm2d(channels),
                activation())
        """
        正则化
        """
        self.ext_regul=nn.Dropout2d(p=dropout_prob)
        """
        非线性激活
        """
        self.out_activation=activation()
        
                
    def forward(self,x):
        main=x
        #print(type(x))
        #print("==========")
        ext=self.ext_conv1(x)
        ext=self.ext_conv2(ext)
        ext=self.ext_conv3(ext)
        ext=self.ext_regul(ext)
        
        out=main+ext
        return self.out_activation(out)
    
                
"""
下采样的Bottleneck

"""       
class DownsamplingBottleneck(nn.Module):      
    def __init__(self,in_channels,out_channels,internal_ratio=4,return_indices=False, dropout_prob=0,bias=False,relu=True):
        super(DownsamplingBottleneck, self).__init__()
        
        self.return_indices=return_indices
        """
        检查internal_ratio参数范围
        """
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(in_channels, internal_ratio))
        internal_channels=in_channels//internal_ratio
        
        if(relu):
            activation=nn.ReLU
        else:
            activation=nn.PReLU

        """
        主分支里面进行最大池化,并返回索引
        """
        self.main_max1 = nn.MaxPool2d(2,stride=2,return_indices=return_indices)

        """
        从分支2x2卷积步长为2进行下采样
        """
        self.ext_conv1=nn.Sequential(
                nn.Conv2d(in_channels,internal_channels,kernel_size=2,stride=2,bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation())
        
        self.ext_conv2=nn.Sequential(
                nn.Conv2d(internal_channels,internal_channels,kernel_size=3,stride=1,padding=1,bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation())
        
        self.ext_conv3=nn.Sequential(
                nn.Conv2d(internal_channels,out_channels,kernel_size=1,stride=1,bias=bias),
                nn.BatchNorm2d(out_channels),
                activation())
        self.ext_regul=nn.Dropout2d(p=dropout_prob)
        self.out_activation=activation()
        
    def forward(self,x):
        if(self.return_indices):
            main,max_indices=self.main_max1(x)
        else:
            main=self.main_max1(x)
            
        ext=self.ext_conv1(x)
        ext=self.ext_conv2(ext)
        ext=self.ext_conv3(ext)
        ext=self.ext_regul(ext)
        
        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)
        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)
        # Add main and extension branches
        out = main + ext
        return self.out_activation(out), max_indices

"""
上采样的Bottleneck
"""
class UpsamplingBottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,internal_ratio=4,dropout_prob=0,bias=False,relu=True):
        super(UpsamplingBottleneck, self).__init__()
        
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))
        
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)
        
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())


        """
        Transposed convolution
        """
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()
        
        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        
        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()
        
        
    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)

class ENet(nn.Module):
    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super(ENet, self).__init__()
        ##256x256
        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)
        
        # Stage 1 - Encoder
        ##128x128
        self.downsample1_0 = DownsamplingBottleneck(16,64,return_indices=True,dropout_prob=0.01,relu=encoder_relu)
        
        #4个不带下采样的Bottleneck
        self.regular1_1 = Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        
        
        
        # Stage 2 - Encoder
        ##64x64
        self.downsample2_0 = DownsamplingBottleneck(64,128,return_indices=True,dropout_prob=0.1,relu=encoder_relu)
        self.regular2_1 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = Bottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = Bottleneck(128,kernel_size=5,padding=2, asymmetric=True,dropout_prob=0.1,relu=encoder_relu)
        
        
        self.dilated2_4 = Bottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = Bottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        
        self.asymmetric2_7 = Bottleneck(128,kernel_size=5,asymmetric=True,padding=2,dropout_prob=0.1,relu=encoder_relu)
        self.dilated2_8 = Bottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        
        # Stage 3 - Encoder
        ##64x64
        self.regular3_0 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = Bottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = Bottleneck(128,kernel_size=5,padding=2,asymmetric=True,dropout_prob=0.1,relu=encoder_relu)
        
        self.dilated3_3 = Bottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = Bottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = Bottleneck(128,kernel_size=5,asymmetric=True,padding=2,dropout_prob=0.1,relu=encoder_relu)
        self.dilated3_7 = Bottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        
        
        # Stage 4 - Decoder
        ##128x128
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        
        self.regular4_1 = Bottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = Bottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        ##256x256
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = Bottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        
        ##512x512
        self.transposed_conv = nn.ConvTranspose2d(16,num_classes,kernel_size=3,stride=2,padding=1,bias=False)
        
    def forward(self, x):
        # Initial block
        ##512x512
        input_size = x.size()
        
        ##256x256
        x = self.initial_block(x)

        # Stage 1 - Encoder
        ##128x128
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        ##64x64
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        ##64x64
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        ##128x128
        #传入最大池化的索引max_indices2_0
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        ##256x256
        #传入最大池化的索引max_indices1_0
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        
        ##512x512
        print(x.shape)
        print("===========")
        x = self.transposed_conv(x, output_size=input_size)
        return x

if __name__ == "__main__":
    model = ENet(num_classes=8, encoder_relu=False, decoder_relu=True)
    #model.eval()
    inputs = torch.randn(1, 3, 512, 512)
    output = model(inputs)
    print(output.size())













