import torch
from torch import nn
import numpy as np
from torchsummary import summary


class Down_block(nn.Module): 
    def __init__(
        self,
        channels):
        super(Down_block, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.conv_1x1_2 = nn.Conv2d(channels*2,channels*2, kernel_size=1, padding=0)
        self.conv_3x3_stride = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1, stride=2)

        self.activation = nn.ELU()   

    def forward(self, x):
        x_3x3 = self.conv_3x3(x) + x
        x_1x1 = self.conv_1x1_1(x) + x

        x = torch.cat([x_3x3, x_1x1],1)
        x = self.activation(self.conv_1x1_2(self.activation(x)))
        x = self.activation(self.conv_3x3_stride(x))
        return x
    
class Stay_block(nn.Module): 
    def __init__(
        self,
        channels):
        super(Stay_block, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.conv_1x1_2 = nn.Conv2d(channels*2,channels*1, kernel_size=1, padding=0)

        self.activation = nn.ELU()   

    def forward(self, x):
        x_3x3 = self.conv_3x3(x) + x
        x_1x1 = self.conv_1x1_1(x) + x

        x = torch.cat([x_3x3, x_1x1],1)
        x = self.activation(self.conv_1x1_2(self.activation(x)))
        return x
    
class Chan_down(nn.Module): 
    def __init__(
        self,
        channels):
        super(Chan_down, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_1x1_2 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0)

        self.activation = nn.ELU()   

    def forward(self, x):
        org = x
        x = self.activation(self.conv_1x1_1(x)) + org
        x = self.activation(self.conv_3x3(x)) + org
        x = self.activation(self.conv_1x1_2(x))
        return x
    

###############################################################################################################################################################


class Down_with_depthwise_Seperable(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(
        self,
        channels):
        super(Down_with_depthwise_Seperable, self).__init__()
        self.conv_1x1_0 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)

        self.conv_1x1_2 = nn.Conv2d(channels*3,channels*2, kernel_size=1, padding=0)
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
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.conv_1x3 = nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1))
        self.conv_3x1 = nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0))

        self.conv_1x5 = nn.Conv2d(channels, channels, kernel_size=(1,5), padding=(0,2))
        self.conv_5x1 = nn.Conv2d(channels, channels, kernel_size=(5,1), padding=(2,0))

        self.conv_1x1_2 = nn.Conv2d(channels*3,channels*2, kernel_size=1, padding=0)
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
        self.conv_1x1_0 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)
        self.conv_1x3 = nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1), groups=channels)
        self.conv_3x1 = nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0), groups=channels)

        self.conv_1x5 = nn.Conv2d(channels, channels, kernel_size=(1,5), padding=(0,2), groups=channels)
        self.conv_5x1 = nn.Conv2d(channels, channels, kernel_size=(5,1), padding=(2,0), groups=channels)

        self.conv_1x1_2 = nn.Conv2d(channels*3,channels*2, kernel_size=1, padding=0)
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
        self.conv_1x1_0 = nn.Conv2d(channels,channels, kernel_size=1, padding=0,)
        self.conv_1x1_1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)

        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1), groups=channels)

        self.conv_1x5 = nn.Conv2d(channels, channels, kernel_size=(1,5), padding=(0,2), groups=channels)
        self.conv_5x1 = nn.Conv2d(channels, channels, kernel_size=(5,1), padding=(2,0), groups=channels)

        self.conv_1x1_2 = nn.Conv2d(channels*3,channels*2, kernel_size=1, padding=0)
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
        self.org1 = Stay_block(6)
        self.org2 = Stay_block(6)
        self.org3 = Stay_block(6)
        self.org4 = Stay_block(6)
        self.dscds_1 = Down_block(6)
        self.dscds_2 = Down_block(6*2)
        self.dscds_3 = Down_block(6*4)
        self.dscds_4 = Down_block(6*8)
        self.dscds_5 = Down_block(6*16)
        self.dscds_6 = Down_block(6*32)
        self.dscds_7 = Down_block(6*64)
        self.org5 = Stay_block(6*128)
        self.org6 = Stay_block(6*128)
        self.org7 = Stay_block(6*128)
        self.org8 = Stay_block(6*128)
        self.down1 = Chan_down(6*128)
        self.down2 = Chan_down(6*64)
        self.org9 = Stay_block(6*32)
        self.org10 = Stay_block(6*32)

        self.activation = nn.ELU()   

        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.3)

        self.fc_1 = nn.Linear(4608, 1024)
        self.fc_2 = nn.Linear(1024, 256)
        self.fc_3 = nn.Linear(256, 1)

        self.initialize_weights()
        

    def forward(self, x):
        x = self.org1(x) + x
        x = self.org2(x) + x
        x = self.org3(x) + x
        x = self.org4(x) + x
        x = self.dscds_1(x)
        x = self.dscds_2(x)
        x = self.dscds_3(x)
        x = self.dscds_4(x)
        x = self.dscds_5(x)
        x = self.dscds_6(x)
        x = self.dscds_7(x)
        x = self.org5(x) + x
        x = self.org6(x) + x
        x = self.org7(x) + x
        x = self.org8(x) + x
        x = self.down1(x)
        x = self.down2(x)
        x = self.org9(x) + x
        x = self.org10(x) + x
        
        x = x.reshape(x.shape[0], -1)
        #x = self.dropout(x)
        x = self.activation(self.fc_1(x))
        x = self.sigmoid(self.fc_2(x))
        x = self.fc_3(x)
        return self.sigmoid(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)










###############################################################################################################################################################


class Same3(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(
        self,
        channels):
        super(Same3, self).__init__()
        self.conv1 = nn.Conv2d(channels,channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels,channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels,channels, kernel_size=3, padding=1)

        self.activation = nn.ELU()  

    def forward(self, x):
        org = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x)) + org
        x = self.activation(self.conv3(x))
        return x
    

class Same5(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(
        self,
        channels):
        super(Same5, self).__init__()
        self.conv1 = nn.Conv2d(channels,channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(channels,channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(channels,channels, kernel_size=5, padding=2)

        self.activation = nn.ELU()  

    def forward(self, x):
        org = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x)) + org
        x = self.activation(self.conv3(x))
        return x
    
class Stride(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(
        self,
        channels):
        super(Stride, self).__init__()
        self.conv1 = nn.Conv2d(channels,channels*2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(channels*2,channels*2, kernel_size=3, padding=1)

        self.activation = nn.ELU()  

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x)) + x
        return x
    
class Chan_down(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(
        self,
        channels):
        super(Chan_down, self).__init__()
        self.conv1 = nn.Conv2d(channels,channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels,channels//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels//2,channels//2, kernel_size=3, padding=1)

        self.activation = nn.ELU()  

    def forward(self, x):
        #x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x)) + x
        return x

class Loss(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(self):
        super(Loss, self).__init__()
        self._1 = Same3(6)
        self._2 = Same3(6)
        self._2_1 = Same3(6)
        self._2_2 = Same3(6)
        self._3 = Stride(6)
        self._4 = Same3(6*2)
        self._5 = Same3(6*2)
        self._6 = Stride(6*2)
        self._7 = Stride(6*4)
        self._8 = Same3(6*8)
        self._9 = Same3(6*8)
        self._10 = Stride(6*8)
        self._10_1 = Same5(6*16)
        self._10_2 = Same5(6*16)
        self._10_3 = Same5(6*16)
        self._10_4 = Same5(6*16)
        self._11 = Stride(6*16)
        self._12 = Same3(6*32)
        self._13 = Same3(6*32)
        self._14 = Stride(6*32)
        self._15 = Stride(6*64)

        self._16 = Same3(6*128)
        self._17 = Same3(6*128)
        self._18 = Chan_down(6*128)
        self._19 = Same3(6*64)
        self._20 = Chan_down(6*64)
        self._21 = Same3(6*32)

        self.activation = nn.ELU()   

        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.3)

        self.fc_1 = nn.Linear(4608, 1024)
        self.fc_2 = nn.Linear(1024, 256)
        self.fc_3 = nn.Linear(256, 4)
        self.fc_3_5 = nn.Linear(256, 8)
        self.fc_4 = nn.Linear(12, 6)
        self.fc_5 = nn.Linear(6, 1)

        #self.initialize_weights()
        

    def forward(self, x):
        org = x
        x = self._1(x) + x
        x = self._2(x) + org
        x = self._2_1(x) + x
        x = self._2_2(x) + org
        x = self._3(x)
        x = self._4(x)
        x = self._5(x) + x
        x = self._6(x)
        x = self._7(x)
        x = self._8(x)
        x = self._9(x)
        x = self._10(x)
        org = x
        x = self._10_1(x) + x
        x = self._10_2(x) + x
        x = self._10_3(x) + x
        x = self._10_4(x) + org
        x = self._11(x)
        x = self._12(x)
        x = self._13(x) + x
        x = self._14(x)
        x = self._15(x)
        x = self._16(x)
        x = self._17(x) + x
        x = self._18(x)
        x = self._19(x)
        x = self._20(x)
        #x = self._21(x)
        
        x = x.reshape(x.shape[0], -1)
        #x = self.dropout(x)
        x = self.activation(self.fc_1(x))
        x = self.fc_2(x)
        x_1 = self.activation(self.fc_3(self.activation(x)))
        x_2 = self.sigmoid(self.fc_3_5(self.sigmoid(x)))
        x = torch.concat((x_1, x_2), dim=-1)
        x = self.fc_4(x)
        x = self.fc_5(x)
        return self.sigmoid(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)