import torch
from torch import nn
import numpy as np
from torchsummary import summary


slim = True
seperable = True


class Stem(nn.Module): # Based very loosly on inception v4
    def __init__(
        self,
        channels):
        super(Stem, self).__init__()
        self.skip_1x1_1 = nn.Conv2d(channels,64, kernel_size=1, padding=0)
        self.skip_1 = nn.Conv2d(64 if seperable else channels,64, kernel_size=7, padding=2, stride=4, groups=64 if seperable else 1)
        self.skip_slim_7x1_1 = nn.Conv2d(64 if seperable else channels,64, kernel_size=(7,1), padding=(3,0), stride=(4,1), groups=64 if seperable else 1)
        self.skip_slim_1x7_1 = nn.Conv2d(64,64, kernel_size=(1,7), padding=(0,3), stride=(1,4), groups=64 if seperable else 1)
        

        self.base_1x1 = nn.Conv2d(channels,32, kernel_size=1, padding=0)
        self.base_3x3 = nn.Conv2d(32 if seperable else channels,32, kernel_size=3, padding=1, stride=2, groups=32 if seperable else 1)
        self.base_slim_3x1 = nn.Conv2d(32 if seperable else channels,32, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=32 if seperable else 1)
        self.base_slim_1x3 = nn.Conv2d(32,32, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=32 if seperable else 1)
        

        self.A_comv_1x1_1 = nn.Conv2d(32,32, kernel_size=1, padding=0)
        self.A_conv_3x3_1 = nn.Conv2d(32,32, kernel_size=3, padding=1, groups=32 if seperable else 1)
        self.A_conv_slim_3x1_1 = nn.Conv2d(32,32, kernel_size=(3,1), padding=(1,0), groups=32 if seperable else 1)
        self.A_conv_slim_1x3_1 = nn.Conv2d(32,32, kernel_size=(1,3), padding=(0,1), groups=32 if seperable else 1)
        

        self.A_comv_1x1_2 = nn.Conv2d(32,32, kernel_size=1, padding=0)
        self.A_conv_3x3_2 = nn.Conv2d(32,32, kernel_size=3, padding=1, stride=2, groups=32 if seperable else 1)
        self.A_conv_slim_3x1_2 = nn.Conv2d(32,32, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=32 if seperable else 1)
        self.A_conv_slim_1x3_2 = nn.Conv2d(32,32, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=32 if seperable else 1)
        

        self.B_conv_1x1_1 = nn.Conv2d(32,32, kernel_size=1, padding=0)
        self.B_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        

        self.C_comv_1x1_1 = nn.Conv2d(64,64, kernel_size=1, padding=0)
        self.C_conv_3x3_1 = nn.Conv2d(64,64, kernel_size=3, padding=1, groups=64 if seperable else 1)
        self.C_conv_slim_3x1_1 = nn.Conv2d(64,64, kernel_size=(3,1), padding=(1,0), groups=64 if seperable else 1)
        self.C_conv_slim_1x3_1 = nn.Conv2d(64,64, kernel_size=(1,3), padding=(0,1), groups=64 if seperable else 1)
        
        self.C_comv_1x1_2 = nn.Conv2d(64,64, kernel_size=1, padding=0)
        self.C_conv_3x3_2 = nn.Conv2d(64,64, kernel_size=3, padding=1, stride=2, groups=64 if seperable else 1)
        self.C_conv_slim_3x1_2 = nn.Conv2d(64,64, kernel_size=(3,1), stride=(2,1), padding=(1,0), groups=64 if seperable else 1)
        self.C_conv_slim_1x3_2 = nn.Conv2d(64,64, kernel_size=(1,3), stride=(1,2), padding=(0,1), groups=64 if seperable else 1)
        

        self.D_conv_1x1_1 = nn.Conv2d(64,64, kernel_size=1, padding=0)
        self.D_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        

        self.skip_1x1_2 = nn.Conv2d(64,128, kernel_size=1, padding=0)
        self.skip_2 = nn.Conv2d(128 if seperable else 64,128, kernel_size=3, padding=1, stride=2, groups=128 if seperable else 1)
        self.skip_slim_3x1_2 = nn.Conv2d(128 if seperable else 64,128, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=128 if seperable else 1)
        self.skip_slim_1x3_2 = nn.Conv2d(128,128, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=128 if seperable else 1)
        


        self.activation = nn.ELU()  

    def forward(self, x):
        org = x
        # base
        if seperable:
            x = self.base_1x1(x)
        if slim:
            x = self.base_slim_3x1(x)
            x = self.base_slim_1x3(x)
        else:
            x = self.base_3x3(x)
        
        x = self.activation(x)

        # A branch
        A = x
        if seperable:
            A = self.A_comv_1x1_1(A)
        if slim:
            A = self.A_conv_slim_3x1_1(A)
            A = self.A_conv_slim_1x3_1(A)
        else:
            A = self.A_conv_3x3_1(A)
        A = self.activation(A)

        if seperable:
            A = self.A_comv_1x1_2(A)
        if slim:
            A = self.A_conv_slim_3x1_2(A)
            A = self.A_conv_slim_1x3_2(A)
        else:
            A = self.A_conv_3x3_2(A)
        
        A = self.activation(A)

        # B branch
        B = self.B_conv_1x1_1(x)
        B = self.B_pool(B)
        B = self.activation(B)

        # Skip1
        if seperable:
            org = self.skip_1x1_1(org)
        if slim:
            skip1 = self.skip_slim_7x1_1(org)
            skip1 = self.skip_slim_1x7_1(skip1)
        else:
            skip1 = self.skip_1(org)
        

        x = torch.concat((A, B), dim=1) + skip1



        org = x

        # C branch
        C = x
        if seperable:
            C = self.C_comv_1x1_1(C)
        if slim:
            C = self.C_conv_slim_3x1_1(C)
            C = self.C_conv_slim_1x3_1(C)
        else:
            C = self.C_conv_3x3_1(C)
        C = self.activation(C)

        if seperable:
            C = self.C_comv_1x1_2(C)
        if slim:
            C = self.C_conv_slim_3x1_2(C)
            C = self.C_conv_slim_1x3_2(C)
        else:
            C = self.C_conv_3x3_2(C)
        C = self.activation(C)

        # B branch
        D = self.D_conv_1x1_1(x)
        D = self.D_pool(D)
        D = self.activation(D)

        # Skip1
        if seperable:
            org = self.skip_1x1_2(org)
        if slim:
            skip2 = self.skip_slim_3x1_2(org)
            skip2 = self.skip_slim_1x3_2(skip2)
        else:
            skip2 = self.skip_2(org)
        
        x = torch.concat((C, D), dim=1) + skip2

        return x
        





class InceptionA(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels):
        super(InceptionA, self).__init__()
        self.A_1x1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0)
        self.A_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.B_1x1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0)


        self.C_1x1_1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0)
        self.C_3x3 = nn.Conv2d(channels//4, channels//4, kernel_size=3, stride=1, padding=1, groups=channels//4 if seperable else 1)
        self.C_3x1 = nn.Conv2d(channels//4, channels//4, kernel_size=(3,1), stride=1, padding=(1,0), groups=channels//4 if seperable else 1)
        self.C_1x3 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,3), stride=1, padding=(0,1), groups=channels//4 if seperable else 1)
        self.C_1x1_2 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0)


        self.D_1x1_1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0)
        self.D_5x5 = nn.Conv2d(channels//4, channels//4, kernel_size=5, stride=1, padding=2, groups=channels//4 if seperable else 1)
        self.D_5x1 = nn.Conv2d(channels//4, channels//4, kernel_size=(5,1), stride=1, padding=(2,0), groups=channels//4 if seperable else 1)
        self.D_1x5 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,5), stride=1, padding=(0,2), groups=channels//4 if seperable else 1)
        self.D_1x1_2 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0)


        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch

        A = self.A_1x1(x)
        A = self.A_pool(A)
        A = self.activation(A)

        # B branch
        B = self.B_1x1(x)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if slim:
            C = self.C_3x1(C)
            C = self.C_1x3(C)
        else:
            C = self.C_3x3(C)
        if seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)


        # D branch
        D = self.D_1x1_1(x)
        D = self.activation(D)
        if slim:
            D = self.D_5x1(D)
            D = self.D_1x5(D)
        else:
            D = self.D_5x5(D)
        if seperable:
            D = self.D_1x1_2(D)
        D = self.activation(D)

        x = torch.concat((A, B, C, D), dim=1) + org

        return x
    


class ReductionA(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels):
        super(ReductionA, self).__init__()
        self.A_1x1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0)
        self.A_pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.B_3x3 = nn.Conv2d(channels,channels, kernel_size=3, padding=1, stride=2, groups=channels if seperable else 1)
        self.B_3x1 = nn.Conv2d(channels,channels, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels if seperable else 1)
        self.B_1x3 = nn.Conv2d(channels,channels, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels if seperable else 1)
        self.B_1x1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0)

        self.C_1x1_1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0)
        self.C_5x5 = nn.Conv2d(channels//2,channels//2, kernel_size=5, padding=2, stride=2, groups=channels//2 if seperable else 1)
        self.C_5x1 = nn.Conv2d(channels//2,channels//2, kernel_size=(5,1), padding=(2,0), stride=(2,1), groups=channels//2 if seperable else 1)
        self.C_1x5 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,5), padding=(0,2), stride=(1,2), groups=channels//2 if seperable else 1)
        self.C_1x1_2 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0)


        self.skip = nn.Conv2d(channels,channels*2, kernel_size=3, padding=1, stride=2, groups=channels if seperable else 1)
        self.skip_slim_3x1 = nn.Conv2d(channels,channels*2, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels if seperable else 1)
        self.skip_slim_1x3 = nn.Conv2d(channels*2,channels*2, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels*2 if seperable else 1)
        self.skip_1x1 = nn.Conv2d(channels*2,channels*2, kernel_size=1, padding=0)



        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = self.A_1x1(x)
        A = self.A_pool(A)
        A = self.activation(A)

        # B branch
        if slim:
            B = self.B_3x1(x)
            B = self.B_1x3(B)
        else:
            B = self.B_3x3(x)
        if seperable:
            B = self.B_1x1(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if slim:
            C = self.C_5x1(C)
            C = self.C_1x5(C)
        else:
            C = self.C_5x5(C)
        if seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)

        # skip
        if slim:
            skip = self.skip_slim_3x1(x)
            skip = self.skip_slim_1x3(skip)
        else:
            skip = self.skip(x)
        if seperable:
            skip = self.skip_1x1(skip)


        x = torch.concat((A, B, C), dim=1) + skip

        return x




class InceptionB(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels):
        super(InceptionB, self).__init__()
        self.A_1x1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0)
        self.A_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.B_1x1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0)


        self.C_1x1_1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0)
        self.C_7x7 = nn.Conv2d(channels//4, channels//4, kernel_size=7, stride=1, padding=3, groups=channels//4 if seperable else 1)
        self.C_7x1 = nn.Conv2d(channels//4, channels//4, kernel_size=(7,1), stride=1, padding=(3,0), groups=channels//4 if seperable else 1)
        self.C_1x7 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,7), stride=1, padding=(0,3), groups=channels//4 if seperable else 1)
        self.C_1x1_2 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0)


        self.D_1x1_1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0)

        self.D_7x7_1 = nn.Conv2d(channels//4, channels//4, kernel_size=7, stride=1, padding=3, groups=channels//4 if seperable else 1)
        self.D_7x1_1 = nn.Conv2d(channels//4, channels//4, kernel_size=(7,1), stride=1, padding=(3,0), groups=channels//4 if seperable else 1)
        self.D_1x7_1 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,7), stride=1, padding=(0,3), groups=channels//4 if seperable else 1)
        self.D_1x1_2 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0)

        self.D_7x7_2 = nn.Conv2d(channels//4, channels//4, kernel_size=7, stride=1, padding=3, groups=channels//4 if seperable else 1)
        self.D_7x1_2 = nn.Conv2d(channels//4, channels//4, kernel_size=(7,1), stride=1, padding=(3,0), groups=channels//4 if seperable else 1)
        self.D_1x7_2 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,7), stride=1, padding=(0,3), groups=channels//4 if seperable else 1)
        self.D_1x1_3 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0)


        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch

        A = self.A_1x1(x)
        A = self.A_pool(A)
        A = self.activation(A)

        # B branch
        B = self.B_1x1(x)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if slim:
            C = self.C_7x1(C)
            C = self.C_1x7(C)
        else:
            C = self.C_7x7(C)
        if seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)


        # D branch
        D = self.D_1x1_1(x)
        D = self.activation(D)
        if slim:
            D = self.D_7x1_1(D)
            D = self.D_1x7_1(D)
        else:
            D = self.D_7x7_1(D)
        if seperable:
            D = self.D_1x1_2(D)
        D = self.activation(D)

        if slim:
            D = self.D_7x1_2(D)
            D = self.D_1x7_2(D)
        else:
            D = self.D_7x7_2(D)
        if seperable:
            D = self.D_1x1_3(D)
        D = self.activation(D)


        x = torch.concat((A, B, C, D), dim=1) + org

        return x





class ReductionB(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels):
        super(ReductionB, self).__init__()
        self.A_pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.B_1x1_1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0)
        self.B_3x3 = nn.Conv2d(channels//2,channels//2, kernel_size=3, padding=1, stride=2, groups=channels//2 if seperable else 1)
        self.B_3x1 = nn.Conv2d(channels//2,channels//2, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels//2 if seperable else 1)
        self.B_1x3 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels//2 if seperable else 1)
        self.B_1x1_2 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0)


        self.C_1x1_1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0)
        self.C_7x7 = nn.Conv2d(channels//2,channels//2, kernel_size=7, padding=3, groups=channels//2 if seperable else 1)
        self.C_7x1 = nn.Conv2d(channels//2,channels//2, kernel_size=(7,1), padding=(3,0), groups=channels//2 if seperable else 1)
        self.C_1x7 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,7), padding=(0,3), groups=channels//2 if seperable else 1)
        self.C_1x1_2 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0)

        self.C_3x3 = nn.Conv2d(channels//2,channels//2, kernel_size=3, stride=2, padding=1, groups=channels//2 if seperable else 1)
        self.C_3x1 = nn.Conv2d(channels//2, channels//2, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels//2 if seperable else 1)
        self.C_1x3 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels//2 if seperable else 1)
        self.C_1x1_3 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0)


        self.skip = nn.Conv2d(channels,channels*2, kernel_size=3, padding=1, stride=2, groups=channels if seperable else 1)
        self.skip_slim_3x1 = nn.Conv2d(channels,channels*2, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels if seperable else 1)
        self.skip_slim_1x3 = nn.Conv2d(channels*2,channels*2, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels*2 if seperable else 1)
        self.skip_1x1 = nn.Conv2d(channels*2,channels*2, kernel_size=1, padding=0)



        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = self.A_pool(x)
        A = self.activation(A)

        # B branch
        B = self.B_1x1_1(x)
        B = self.activation(B)
        if slim:
            B = self.B_3x1(B)
            B = self.B_1x3(B)
        else:
            B = self.B_3x3(B)
        if seperable:
            B = self.B_1x1_2(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if slim:
            C = self.C_7x1(C)
            C = self.C_1x7(C)
        else:
            C = self.C_7x7(C)
        if seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)
        
        if slim:
            C = self.C_3x1(C)
            C = self.C_1x3(C)
        else:
            C = self.C_3x3(C)
        if seperable:
            C = self.C_1x1_3(C)
        C = self.activation(C)

        # skip
        if slim:
            skip = self.skip_slim_3x1(x)
            skip = self.skip_slim_1x3(skip)
        else:
            skip = self.skip(x)
        if seperable:
            skip = self.skip_1x1(skip)

        x = torch.concat((A, B, C), dim=1) + skip

        return x
    






class Loss(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(self):
        super(Loss, self).__init__()
        self._0 = Stem(6)
        self._1 = InceptionA(128)
        self._2 = InceptionA(128)
        self._3 = InceptionA(128)
        self._4 = InceptionA(128)
        self._4_1 = InceptionA(128)
        self._4_2 = InceptionA(128)

        self._5 = ReductionA(128)

        self._6 = InceptionA(256)
        self._7 = InceptionA(256)

        self._8 = ReductionA(256)

        self._9 = InceptionB(512)
        self._10 = InceptionB(512)
        self._11 = InceptionB(512)
        self._12 = InceptionB(512)

        self._13 = ReductionB(512)

        self._14 = InceptionB(1024)
        self._15 = InceptionB(1024)
        self._16 = InceptionB(1024)
        self._17 = InceptionB(1024)

        self._18 = ReductionB(1024)

        self._19 = InceptionB(2048)

        self.pool = nn.AvgPool2d(kernel_size=(4,6))

        self.activation = nn.ELU()   

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.dropout = nn.Dropout(p=0.2)

        self.fc_1 = nn.Linear(2048, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 24)
        self.fc_3_5 = nn.Linear(64, 8)
        self.fc_4 = nn.Linear(32, 1)

        #self.initialize_weights()
        

    def forward(self, x):
        x = self._0(x)
        x = self._1(x)
        x = self._2(x)
        x = self._3(x)
        x = self._4(x)
        x = self._4_1(x)
        x = self._4_2(x)
        x = self._5(x)
        x = self._6(x)
        x = self._7(x)
        x = self._8(x)
        x = self._9(x)
        x = self._10(x)
        x = self._11(x)
        x = self._12(x)
        x = self._13(x)
        x = self._14(x)
        x = self._15(x)
        x = self._16(x)
        x = self._17(x)
        x = self._18(x)
        x = self._19(x) 
        x = self.pool(x)


        
        x = x.reshape(x.shape[0], -1)
        #x = self.dropout(x)
        x = self.activation(self.fc_1(x))
        x = self.activation(self.fc_2(x))
        x_1 = self.activation(self.fc_3(x))
        x_2 = self.activation(self.fc_3_5(x))
        x = torch.concat((x_1, x_2), dim=-1)
        x = self.fc_4(x)
        return self.sigmoid(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight)