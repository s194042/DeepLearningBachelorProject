import torch
from torch import nn
import numpy as np
from torchsummary import summary


class Down_block(nn.Module): 
    def __init__(
        self,
        channels):
        super(Down_block, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False)

        self.conv_1x1_2 = nn.Conv2d(channels*3,channels*2, kernel_size=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.activation = nn.ELU()   

    def forward(self, x):
        x_3x3 = self.conv_3x3(x) + x
        x_5x5 = self.conv_5x5(x) + x
        x_1x1 = self.conv_1x1_1(x) + x

        x = torch.cat([x_3x3, x_5x5, x_1x1],1)
        x = self.activation(self.conv_1x1_2(self.activation(x)))
        x = self.pool(x)
        return x
    


###############################################################################################################################################################


class Down_with_depthwise_Seperable(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(
        self,
        channels):
        super(Down_with_depthwise_Seperable, self).__init__()
        self.conv_1x1_0 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv_5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)

        self.conv_1x1_2 = nn.Conv2d(channels*3,channels*2, kernel_size=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.activation = nn.ELU()  

    def forward(self, x):
        x_3x3 = self.conv_3x3(self.conv_1x1_0(x)) + x
        x_5x5 = self.conv_5x5(self.conv_1x1_0(x)) + x
        x_1x1 = self.conv_1x1_1(self.conv_1x1_0(x)) + x

        x = torch.cat([x_3x3, x_5x5, x_1x1],1)
        x = self.activation(self.conv_1x1_2(self.activation(x)))
        x = self.pool(x)
        return x


    


###############################################################################################################################################################


class Down_with_slim_kernels(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(
        self,
        channels):
        super(Down_with_slim_kernels, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)
        self.conv_1x3 = nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1), bias=False)
        self.conv_3x1 = nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0), bias=False)

        self.conv_1x5 = nn.Conv2d(channels, channels, kernel_size=(1,5), padding=(0,2), bias=False)
        self.conv_5x1 = nn.Conv2d(channels, channels, kernel_size=(5,1), padding=(2,0), bias=False)

        self.conv_1x1_2 = nn.Conv2d(channels*3,channels*2, kernel_size=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.activation = nn.ELU()  

    def forward(self, x):
        x_3x3 = self.conv_3x1(self.conv_1x3(x)) + x
        x_5x5 = self.conv_5x1(self.conv_1x5(x)) + x
        x_1x1 = self.conv_1x1_1(x) + x

        x = torch.cat([x_3x3, x_5x5, x_1x1],1)
        x = self.activation(self.conv_1x1_2(self.activation(x)))
        x = self.pool(x)
        return x





###############################################################################################################################################################


class Down_final(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(
        self,
        channels):
        super(Down_final, self).__init__()
        self.conv_1x1_0 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)
        self.conv_1x3 = nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1), groups=channels, bias=False)
        self.conv_3x1 = nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0), groups=channels, bias=False)

        self.conv_1x5 = nn.Conv2d(channels, channels, kernel_size=(1,5), padding=(0,2), groups=channels, bias=False)
        self.conv_5x1 = nn.Conv2d(channels, channels, kernel_size=(5,1), padding=(2,0), groups=channels, bias=False)

        self.conv_1x1_2 = nn.Conv2d(channels*3,channels*2, kernel_size=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.activation = nn.ELU()  

    def forward(self, x):
        x_3x3 = self.conv_3x1(self.conv_1x3(self.conv_1x1_0(x))) + x
        x_5x5 = self.conv_5x1(self.conv_1x5(self.conv_1x1_0(x))) + x
        x_1x1 = self.conv_1x1_1(x) + x

        x = torch.cat([x_3x3, x_5x5, x_1x1],1)
        x = self.activation(self.conv_1x1_2(self.activation(x)))
        x = self.pool(x)
        return x


###############################################################################################################################################################


class Down_final2(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(
        self,
        channels):
        super(Down_final2, self).__init__()
        self.conv_1x1_0 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)

        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1), groups=channels, bias=False)

        self.conv_1x5 = nn.Conv2d(channels, channels, kernel_size=(1,5), padding=(0,2), groups=channels, bias=False)
        self.conv_5x1 = nn.Conv2d(channels, channels, kernel_size=(5,1), padding=(2,0), groups=channels, bias=False)

        self.conv_1x1_2 = nn.Conv2d(channels*3,channels*2, kernel_size=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.activation = nn.ELU()  

    def forward(self, x):
        x_3x3 = self.conv_3x3(self.conv_1x1_0(x)) + x
        x_5x5 = self.conv_5x1(self.conv_1x5(self.conv_1x1_0(x))) + x
        x_1x1 = self.conv_1x1_1(x) + x

        x = torch.cat([x_3x3, x_5x5, x_1x1],1)
        x = self.activation(self.conv_1x1_2(self.activation(x)))
        x = self.pool(x)
        return x

class Loss(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(self):
        super(Loss, self).__init__()
        self.dscds_1 = Down_block(6)
        self.dscds_2 = Down_block(6*2)
        self.dscds_3 = Down_block(6*4)
        self.dscds_4 = Down_block(6*8)
        self.dscds_5 = Down_block(6*16)
        self.dscds_6 = Down_block(6*32)
        self.dscds_7 = Down_block(6*64)
        self.dscds_8 = Down_block(6*128)
        
        self.activation = nn.ELU()   

        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.3)

        self.fc_1 = nn.Linear(3072, 256)
        self.fc_2 = nn.Linear(256, 8)
        self.fc_3 = nn.Linear(8, 1)
        

    def forward(self, x):
        x = self.dscds_1(x)
        x = self.dscds_2(x)
        x = self.dscds_3(x)
        x = self.dscds_4(x)
        x = self.dscds_5(x)
        x = self.dscds_6(x)
        x = self.dscds_7(x)
        x = self.dscds_8(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.activation(self.fc_1(x))
        x = self.activation(self.fc_2(x))
        x = self.fc_3(x)
        return self.sigmoid(x)