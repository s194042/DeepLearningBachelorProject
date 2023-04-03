import torch
from torch import nn
import numpy as np
from torchsummary import summary





class Stem(nn.Module): # Based very loosly on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(Stem, self).__init__()
        self.slim = slim
        self.seperable = seperable

        self.base_1x1 = nn.Conv2d(channels,8, kernel_size=1, padding=0, bias=False)
        self.base_3x3 = nn.Conv2d(8 if self.seperable else channels,8, kernel_size=3, padding=1, stride=1, groups=8 if self.seperable else 1, bias=False)
        self.base_slim_3x1 = nn.Conv2d(8 if self.seperable else channels,8, kernel_size=(3,1), padding=(1,0), stride=1, groups=8 if self.seperable else 1, bias=False)
        self.base_slim_1x3 = nn.Conv2d(8,8, kernel_size=(1,3), padding=(0,1), stride=1, groups=8 if self.seperable else 1, bias=False)
        

        self.A_comv_1x1_1 = nn.Conv2d(8,8, kernel_size=1, padding=0, bias=False)
        self.A_conv_3x3_1 = nn.Conv2d(8,8, kernel_size=3, padding=1, groups=8 if self.seperable else 1, bias=False)
        self.A_conv_slim_3x1_1 = nn.Conv2d(8,8, kernel_size=(3,1), padding=(1,0), groups=8 if self.seperable else 1, bias=False)
        self.A_conv_slim_1x3_1 = nn.Conv2d(8,8, kernel_size=(1,3), padding=(0,1), groups=8 if self.seperable else 1, bias=False)
        

        self.A_comv_1x1_2 = nn.Conv2d(8,8, kernel_size=1, padding=0, bias=False)
        self.A_conv_3x3_2 = nn.Conv2d(8,8, kernel_size=3, padding=1, stride=1, groups=8 if self.seperable else 1, bias=False)
        self.A_conv_slim_3x1_2 = nn.Conv2d(8,8, kernel_size=(3,1), padding=(1,0), stride=1, groups=8 if self.seperable else 1, bias=False)
        self.A_conv_slim_1x3_2 = nn.Conv2d(8,8, kernel_size=(1,3), padding=(0,1), stride=1, groups=8 if self.seperable else 1, bias=False)
        

        self.B_conv_1x1_1 = nn.Conv2d(8,8, kernel_size=1, padding=0, bias=False)
        self.B_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


        self.skip = nn.Conv2d(channels,16, kernel_size=1, padding=0, bias=False)

        self.activation = nn.ELU()  

    def forward(self, x):
        org = x
        # base
        if self.seperable:
            x = self.base_1x1(x)
        if self.slim:
            x = self.base_slim_3x1(x)
            x = self.base_slim_1x3(x)
        else:
            x = self.base_3x3(x)
        
        x = self.activation(x)

        # A branch
        A = x
        if self.seperable:
            A = self.A_comv_1x1_1(A)
        if self.slim:
            A = self.A_conv_slim_3x1_1(A)
            A = self.A_conv_slim_1x3_1(A)
        else:
            A = self.A_conv_3x3_1(A)
        A = self.activation(A)

        if self.seperable:
            A = self.A_comv_1x1_2(A)
        if self.slim:
            A = self.A_conv_slim_3x1_2(A)
            A = self.A_conv_slim_1x3_2(A)
        else:
            A = self.A_conv_3x3_2(A)
        
        A = self.activation(A)

        # B branch
        B = self.B_conv_1x1_1(x)
        B = self.B_pool(B)
        B = self.activation(B)

        

        x = torch.concat((A, B), dim=1) + self.skip(org)


        return x
        

class DeStem(nn.Module): # Based very loosly on inception v4
    def __init__(
        self,
        slim: bool = False,
        seperable: bool = False):
        super(DeStem, self).__init__()
        self.slim = slim
        self.seperable = seperable
        

        self.A_comv_1x1_1 = nn.Conv2d(16,4, kernel_size=1, padding=0, bias=False)
        self.A_conv_3x3_1 = nn.Conv2d(4 if seperable else 16,4, kernel_size=3, padding=1, groups=4 if self.seperable else 1, bias=False)
        self.A_conv_slim_3x1_1 = nn.Conv2d(4 if seperable else 16,4, kernel_size=(3,1), padding=(1,0), groups=4 if self.seperable else 1, bias=False)
        self.A_conv_slim_1x3_1 = nn.Conv2d(4,4, kernel_size=(1,3), padding=(0,1), groups=4 if self.seperable else 1, bias=False)
        

        self.A_comv_1x1_2 = nn.Conv2d(4,4, kernel_size=1, padding=0, bias=False)
        self.A_conv_3x3_2 = nn.Conv2d(4,4, kernel_size=3, padding=1, stride=1, groups=4 if self.seperable else 1, bias=False)
        self.A_conv_slim_3x1_2 = nn.Conv2d(4,4, kernel_size=(3,1), padding=(1,0), stride=1, groups=4 if self.seperable else 1, bias=False)
        self.A_conv_slim_1x3_2 = nn.Conv2d(4,4, kernel_size=(1,3), padding=(0,1), stride=1, groups=4 if self.seperable else 1, bias=False)
        

        self.B_conv_1x1_1 = nn.Conv2d(16,4, kernel_size=1, padding=0, bias=False)
        self.B_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


        self.skip_1 = nn.Conv2d(16,8, kernel_size=1, padding=0, bias=False)



        self.C_comv_1x1_1 = nn.Conv2d(8,2, kernel_size=1, padding=0, bias=False)
        self.C_conv_3x3_1 = nn.Conv2d(2 if seperable else 8,2, kernel_size=3, padding=1, groups=2 if self.seperable else 1, bias=False)
        self.C_conv_slim_3x1_1 = nn.Conv2d(2 if seperable else 8,2, kernel_size=(3,1), padding=(1,0), groups=2 if self.seperable else 1, bias=False)
        self.C_conv_slim_1x3_1 = nn.Conv2d(2,2, kernel_size=(1,3), padding=(0,1), groups=2 if self.seperable else 1, bias=False)
        

        self.C_comv_1x1_2 = nn.Conv2d(2,2, kernel_size=1, padding=0, bias=False)
        self.C_conv_3x3_2 = nn.Conv2d(2,2, kernel_size=3, padding=1, stride=1, groups=2 if self.seperable else 1, bias=False)
        self.C_conv_slim_3x1_2 = nn.Conv2d(2,2, kernel_size=(3,1), padding=(1,0), stride=1, groups=2 if self.seperable else 1, bias=False)
        self.C_conv_slim_1x3_2 = nn.Conv2d(2,2, kernel_size=(1,3), padding=(0,1), stride=1, groups=2 if self.seperable else 1, bias=False)
        

        self.D_conv_1x1_1 = nn.Conv2d(8,2, kernel_size=1, padding=0, bias=False)
        self.D_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


        self.skip_2 = nn.Conv2d(8,4, kernel_size=1, padding=0, bias=False)





        self.E_comv_1x1_1 = nn.Conv2d(4,3, kernel_size=1, padding=0, bias=False)
        self.E_conv_3x3_1 = nn.Conv2d(3 if seperable else 4,3, kernel_size=3, padding=1, groups=3 if self.seperable else 1, bias=False)
        self.E_conv_slim_3x1_1 = nn.Conv2d(3 if seperable else 4,3, kernel_size=(3,1), padding=(1,0), groups=3 if self.seperable else 1, bias=False)
        self.E_conv_slim_1x3_1 = nn.Conv2d(3,3, kernel_size=(1,3), padding=(0,1), groups=3 if self.seperable else 1, bias=False)
        

        self.E_comv_1x1_2 = nn.Conv2d(3,3, kernel_size=1, padding=0, bias=False)
        self.E_conv_3x3_2 = nn.Conv2d(3,3, kernel_size=3, padding=1, stride=1, groups=3 if self.seperable else 1, bias=False)
        self.E_conv_slim_3x1_2 = nn.Conv2d(3,3, kernel_size=(3,1), padding=(1,0), stride=1, groups=3 if self.seperable else 1, bias=False)
        self.E_conv_slim_1x3_2 = nn.Conv2d(3,3, kernel_size=(1,3), padding=(0,1), stride=1, groups=3 if self.seperable else 1, bias=False)

        self.E_comv_1x1_3 = nn.Conv2d(3,3, kernel_size=1, padding=0, bias=False)
        self.E_conv_3x3_3 = nn.Conv2d(3,3, kernel_size=3, padding=1, stride=1, groups=3 if self.seperable else 1, bias=False)
        self.E_conv_slim_3x1_3 = nn.Conv2d(3,3, kernel_size=(3,1), padding=(1,0), stride=1, groups=3 if self.seperable else 1, bias=False)
        self.E_conv_slim_1x3_3 = nn.Conv2d(3,3, kernel_size=(1,3), padding=(0,1), stride=1, groups=3 if self.seperable else 1, bias=False)
        



        self.skip_3 = nn.Conv2d(4,3, kernel_size=1, padding=0, bias=False)

        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = x
        if self.seperable:
            A = self.A_comv_1x1_1(A)
        if self.slim:
            A = self.A_conv_slim_3x1_1(A)
            A = self.A_conv_slim_1x3_1(A)
        else:
            A = self.A_conv_3x3_1(A)
        A = self.activation(A)

        if self.seperable:
            A = self.A_comv_1x1_2(A)
        if self.slim:
            A = self.A_conv_slim_3x1_2(A)
            A = self.A_conv_slim_1x3_2(A)
        else:
            A = self.A_conv_3x3_2(A)
        
        A = self.activation(A)

        # B branch
        B = self.B_conv_1x1_1(x)
        B = self.B_pool(B)
        B = self.activation(B)

        
        x = torch.concat((A, B), dim=1) + self.skip_1(org)

        org = x

        # C branch
        C = x
        if self.seperable:
            C = self.C_comv_1x1_1(C)
        if self.slim:
            C = self.C_conv_slim_3x1_1(C)
            C = self.C_conv_slim_1x3_1(C)
        else:
            C = self.C_conv_3x3_1(C)
        C = self.activation(C)
        if self.seperable:
            C = self.C_comv_1x1_2(C)
        if self.slim:
            C = self.C_conv_slim_3x1_2(C)
            C = self.C_conv_slim_1x3_2(C)
        else:
            C = self.C_conv_3x3_2(C)
        
        C = self.activation(C)
        # D branch
        D = self.D_conv_1x1_1(x)
        D = self.D_pool(D)
        D = self.activation(D)

        
        x = torch.concat((C, D), dim=1) + self.skip_2(org)

        org = x

        E = x
        if self.seperable:
            E = self.E_comv_1x1_1(E)
        if self.slim:
            E = self.E_conv_slim_3x1_1(E)
            E = self.E_conv_slim_1x3_1(E)
        else:
            E = self.E_conv_3x3_1(E)
        E = self.activation(E)
        if self.seperable:
            E = self.E_comv_1x1_2(E)
        if self.slim:
            E = self.E_conv_slim_3x1_2(E)
            E = self.E_conv_slim_1x3_2(E)
        else:
            E = self.E_conv_3x3_2(E)
        
        E = self.activation(E)
        if self.seperable:
            E = self.E_comv_1x1_3(E)
        if self.slim:
            E = self.E_conv_slim_3x1_3(E)
            E = self.E_conv_slim_1x3_3(E)
        else:
            E = self.E_conv_3x3_3(E)

        E = self.activation(E)


        x = E + self.skip_3(org)

        return x




class SameA(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(SameA, self).__init__()
        self.slim = slim
        self.seperable = seperable


        self.A_1x1_1 = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, bias=False)
        self.A_3x3 = nn.Conv2d(channels//2, channels//2, kernel_size=3, stride=1, padding=1, groups=channels//2 if self.seperable else 1, bias=False)
        self.A_3x1 = nn.Conv2d(channels//2, channels//2, kernel_size=(3,1), stride=1, padding=(1,0), groups=channels//2 if self.seperable else 1, bias=False)
        self.A_1x3 = nn.Conv2d(channels//2, channels//2, kernel_size=(1,3), stride=1, padding=(0,1), groups=channels//2 if self.seperable else 1, bias=False)
        self.A_1x1_2 = nn.Conv2d(channels//2, channels//2, kernel_size=1, padding=0, bias=False)


        self.B_1x1_1 = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, bias=False)
        self.B_5x5 = nn.Conv2d(channels//2, channels//2, kernel_size=5, stride=1, padding=2, groups=channels//2 if self.seperable else 1, bias=False)
        self.B_5x1 = nn.Conv2d(channels//2, channels//2, kernel_size=(5,1), stride=1, padding=(2,0), groups=channels//2 if self.seperable else 1, bias=False)
        self.B_1x5 = nn.Conv2d(channels//2, channels//2, kernel_size=(1,5), stride=1, padding=(0,2), groups=channels//2 if self.seperable else 1, bias=False)
        self.B_1x1_2 = nn.Conv2d(channels//2, channels//2, kernel_size=1, padding=0, bias=False)


        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = self.A_1x1_1(x)
        A = self.activation(A)
        if self.slim:
            A = self.A_3x1(A)
            A = self.A_1x3(A)
        else:
            A = self.A_3x3(A)
        if self.seperable:
            A = self.A_1x1_2(A)
        A = self.activation(A)

        # B branch
        B = self.B_1x1_1(x)
        B = self.activation(B)
        if self.slim:
            B = self.B_5x1(B)
            B = self.B_1x5(B)
        else:
            B = self.B_5x5(B)
        if self.seperable:
            B = self.B_1x1_2(B)
        B = self.activation(B)

        x = torch.concat((A, B), dim=1) + org

        return x
    


class EncoderA(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(EncoderA, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_1x1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0, bias=False)
        self.A_pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.B_3x3 = nn.Conv2d(channels,channels, kernel_size=3, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.B_3x1 = nn.Conv2d(channels,channels, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.B_1x3 = nn.Conv2d(channels,channels, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels if self.seperable else 1, bias=False)
        self.B_1x1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)

        self.C_1x1_1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0, bias=False)
        self.C_5x5 = nn.Conv2d(channels//2,channels//2, kernel_size=5, padding=2, stride=2, groups=channels//2 if self.seperable else 1, bias=False)
        self.C_5x1 = nn.Conv2d(channels//2,channels//2, kernel_size=(5,1), padding=(2,0), stride=(2,1), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x5 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,5), padding=(0,2), stride=(1,2), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0, bias=False)


        self.skip = nn.Conv2d(channels,channels*2, kernel_size=3, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_3x1 = nn.Conv2d(channels,channels*2, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_1x3 = nn.Conv2d(channels*2,channels*2, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels*2 if self.seperable else 1, bias=False)
        self.skip_1x1 = nn.Conv2d(channels*2,channels*2, kernel_size=1, padding=0, bias=False)



        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = self.A_1x1(x)
        A = self.A_pool(A)
        A = self.activation(A)

        # B branch
        if self.slim:
            B = self.B_3x1(x)
            B = self.B_1x3(B)
        else:
            B = self.B_3x3(x)
        if self.seperable:
            B = self.B_1x1(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if self.slim:
            C = self.C_5x1(C)
            C = self.C_1x5(C)
        else:
            C = self.C_5x5(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)

        # skip
        if self.slim:
            skip = self.skip_slim_3x1(x)
            skip = self.skip_slim_1x3(skip)
        else:
            skip = self.skip(x)
        if self.seperable:
            skip = self.skip_1x1(skip)


        x = torch.concat((A, B, C), dim=1) + skip

        return x






class DecoderA(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(DecoderA, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_1x1 = nn.Conv2d(channels,channels//8, kernel_size=1, padding=0, bias=False)
        self.A_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.B_4x4 = nn.ConvTranspose2d(channels,channels//4, kernel_size=4, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.B_4x1 = nn.ConvTranspose2d(channels,channels//4, kernel_size=(4,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.B_1x4 = nn.ConvTranspose2d(channels//4,channels//4, kernel_size=(1,4), padding=(0,1), stride=(1,2), groups=channels//4 if self.seperable else 1, bias=False)
        self.B_1x1 = nn.Conv2d(channels//4,channels//4, kernel_size=1, padding=0, bias=False)

        self.C_1x1_1 = nn.Conv2d(channels,channels//8, kernel_size=1, padding=0, bias=False)
        self.C_6x6 = nn.ConvTranspose2d(channels//8,channels//8, kernel_size=6, padding=2, stride=2, groups=channels//8 if self.seperable else 1, bias=False)
        self.C_6x1 = nn.ConvTranspose2d(channels//8,channels//8, kernel_size=(6,1), padding=(2,0), stride=(2,1), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x6 = nn.ConvTranspose2d(channels//8,channels//8, kernel_size=(1,6), padding=(0,2), stride=(1,2), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//8,channels//8, kernel_size=1, padding=0, bias=False)


        self.skip = nn.ConvTranspose2d(channels,channels//2, kernel_size=4, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_4x1 = nn.ConvTranspose2d(channels,channels//2, kernel_size=(4,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_1x4 = nn.ConvTranspose2d(channels//2,channels//2, kernel_size=(1,4), padding=(0,1), stride=(1,2), groups=channels//2 if self.seperable else 1, bias=False)
        self.skip_1x1 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0, bias=False)

        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = self.A_1x1(x)
        A = self.A_upsample(A)
        A = self.activation(A)

        # B branch
        if self.slim:
            B = self.B_4x1(x)
            B = self.B_1x4(B)
        else:
            B = self.B_4x4(x)
        if self.seperable:
            B = self.B_1x1(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if self.slim:
            C = self.C_6x1(C)
            C = self.C_1x6(C)
        else:
            C = self.C_6x6(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)

        # skip
        if self.slim:
            skip = self.skip_slim_4x1(x)
            skip = self.skip_slim_1x4(skip)
        else:
            skip = self.skip(x)
        if self.seperable:
            skip = self.skip_1x1(skip)

        x = torch.concat((A, B, C), dim=1) + skip

        return x








class SameB(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(SameB, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_1x1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0, bias=False)
        self.A_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.B_1x1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0, bias=False)


        self.C_1x1_1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0, bias=False)
        self.C_3x3 = nn.Conv2d(channels//4, channels//4, kernel_size=3, stride=1, padding=1, groups=channels//4 if self.seperable else 1, bias=False)
        self.C_3x1 = nn.Conv2d(channels//4, channels//4, kernel_size=(3,1), stride=1, padding=(1,0), groups=channels//4 if self.seperable else 1, bias=False)
        self.C_1x3 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,3), stride=1, padding=(0,1), groups=channels//4 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0, bias=False)


        self.D_1x1_1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0, bias=False)
        self.D_7x7 = nn.Conv2d(channels//4, channels//4, kernel_size=7, stride=1, padding=3, groups=channels//4 if self.seperable else 1, bias=False)
        self.D_7x1 = nn.Conv2d(channels//4, channels//4, kernel_size=(7,1), stride=1, padding=(3,0), groups=channels//4 if self.seperable else 1, bias=False)
        self.D_1x7 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,7), stride=1, padding=(0,3), groups=channels//4 if self.seperable else 1, bias=False)
        self.D_1x1_2 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0, bias=False)


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
        if self.slim:
            C = self.C_3x1(C)
            C = self.C_1x3(C)
        else:
            C = self.C_3x3(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)


        # D branch
        D = self.D_1x1_1(x)
        D = self.activation(D)
        if self.slim:
            D = self.D_7x1(D)
            D = self.D_1x7(D)
        else:
            D = self.D_7x7(D)
        if self.seperable:
            D = self.D_1x1_2(D)
        D = self.activation(D)

        x = torch.concat((A, B, C, D), dim=1) + org

        return x
    


class EncoderB(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(EncoderB, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_1x1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0, bias=False)
        self.A_pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.B_3x3 = nn.Conv2d(channels,channels, kernel_size=3, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.B_3x1 = nn.Conv2d(channels,channels, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.B_1x3 = nn.Conv2d(channels,channels, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels if self.seperable else 1, bias=False)
        self.B_1x1 = nn.Conv2d(channels,channels, kernel_size=1, padding=0, bias=False)

        self.C_1x1_1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0, bias=False)
        self.C_7x7 = nn.Conv2d(channels//2,channels//2, kernel_size=7, padding=3, stride=2, groups=channels//2 if self.seperable else 1, bias=False)
        self.C_7x1 = nn.Conv2d(channels//2,channels//2, kernel_size=(7,1), padding=(3,0), stride=(2,1), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x7 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,7), padding=(0,3), stride=(1,2), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0, bias=False)


        self.skip = nn.Conv2d(channels,channels*2, kernel_size=3, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_3x1 = nn.Conv2d(channels,channels*2, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_1x3 = nn.Conv2d(channels*2,channels*2, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels*2 if self.seperable else 1, bias=False)
        self.skip_1x1 = nn.Conv2d(channels*2,channels*2, kernel_size=1, padding=0, bias=False)



        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = self.A_1x1(x)
        A = self.A_pool(A)
        A = self.activation(A)

        # B branch
        if self.slim:
            B = self.B_3x1(x)
            B = self.B_1x3(B)
        else:
            B = self.B_3x3(x)
        if self.seperable:
            B = self.B_1x1(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if self.slim:
            C = self.C_7x1(C)
            C = self.C_1x7(C)
        else:
            C = self.C_7x7(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)

        # skip
        if self.slim:
            skip = self.skip_slim_3x1(x)
            skip = self.skip_slim_1x3(skip)
        else:
            skip = self.skip(x)
        if self.seperable:
            skip = self.skip_1x1(skip)


        x = torch.concat((A, B, C), dim=1) + skip

        return x
    


class DecoderB(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(DecoderB, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_1x1 = nn.Conv2d(channels,channels//8, kernel_size=1, padding=0, bias=False)
        self.A_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.B_4x4 = nn.ConvTranspose2d(channels,channels//4, kernel_size=4, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.B_4x1 = nn.ConvTranspose2d(channels,channels//4, kernel_size=(4,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.B_1x4 = nn.ConvTranspose2d(channels//4,channels//4, kernel_size=(1,4), padding=(0,1), stride=(1,2), groups=channels//4 if self.seperable else 1, bias=False)
        self.B_1x1 = nn.Conv2d(channels//4,channels//4, kernel_size=1, padding=0, bias=False)

        self.C_1x1_1 = nn.Conv2d(channels,channels//8, kernel_size=1, padding=0, bias=False)
        self.C_8x8 = nn.ConvTranspose2d(channels//8,channels//8, kernel_size=8, padding=3, stride=2, groups=channels//8 if self.seperable else 1, bias=False)
        self.C_8x1 = nn.ConvTranspose2d(channels//8,channels//8, kernel_size=(8,1), padding=(3,0), stride=(2,1), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x8 = nn.ConvTranspose2d(channels//8,channels//8, kernel_size=(1,8), padding=(0,3), stride=(1,2), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//8,channels//8, kernel_size=1, padding=0, bias=False)


        self.skip = nn.ConvTranspose2d(channels,channels//2, kernel_size=4, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_4x1 = nn.ConvTranspose2d(channels,channels//2, kernel_size=(4,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_1x4 = nn.ConvTranspose2d(channels//2,channels//2, kernel_size=(1,4), padding=(0,1), stride=(1,2), groups=channels//2 if self.seperable else 1, bias=False)
        self.skip_1x1 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0, bias=False)



        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = self.A_1x1(x)
        A = self.A_upsample(A)
        A = self.activation(A)

        # B branch
        if self.slim:
            B = self.B_4x1(x)
            B = self.B_1x4(B)
        else:
            B = self.B_4x4(x)
        if self.seperable:
            B = self.B_1x1(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if self.slim:
            C = self.C_8x1(C)
            C = self.C_1x8(C)
        else:
            C = self.C_8x8(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)

        # skip
        if self.slim:
            skip = self.skip_slim_4x1(x)
            skip = self.skip_slim_1x4(skip)
        else:
            skip = self.skip(x)
        if self.seperable:
            skip = self.skip_1x1(skip)
        
        

        x = torch.concat((A, B, C), dim=1) + skip

        return x






class SameC(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(SameC, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_1x1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0, bias=False)
        self.A_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.B_1x1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0, bias=False)


        self.C_1x1_1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0, bias=False)
        self.C_7x7 = nn.Conv2d(channels//4, channels//4, kernel_size=7, stride=1, padding=3, groups=channels//4 if self.seperable else 1, bias=False)
        self.C_7x1 = nn.Conv2d(channels//4, channels//4, kernel_size=(7,1), stride=1, padding=(3,0), groups=channels//4 if self.seperable else 1, bias=False)
        self.C_1x7 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,7), stride=1, padding=(0,3), groups=channels//4 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0, bias=False)


        self.D_1x1_1 = nn.Conv2d(channels, channels//4, kernel_size=1, padding=0, bias=False)

        self.D_5x5 = nn.Conv2d(channels//4, channels//4, kernel_size=5, stride=1, padding=2, groups=channels//4 if self.seperable else 1, bias=False)
        self.D_5x1 = nn.Conv2d(channels//4, channels//4, kernel_size=(5,1), stride=1, padding=(2,0), groups=channels//4 if self.seperable else 1, bias=False)
        self.D_1x5 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,5), stride=1, padding=(0,2), groups=channels//4 if self.seperable else 1, bias=False)
        self.D_1x1_2 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0, bias=False)

        self.D_7x7 = nn.Conv2d(channels//4, channels//4, kernel_size=7, stride=1, padding=3, groups=channels//4 if self.seperable else 1, bias=False)
        self.D_7x1 = nn.Conv2d(channels//4, channels//4, kernel_size=(7,1), stride=1, padding=(3,0), groups=channels//4 if self.seperable else 1, bias=False)
        self.D_1x7 = nn.Conv2d(channels//4, channels//4, kernel_size=(1,7), stride=1, padding=(0,3), groups=channels//4 if self.seperable else 1, bias=False)
        self.D_1x1_3 = nn.Conv2d(channels//4, channels//4, kernel_size=1, padding=0, bias=False)


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
        if self.slim:
            C = self.C_7x1(C)
            C = self.C_1x7(C)
        else:
            C = self.C_7x7(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)


        # D branch
        D = self.D_1x1_1(x)
        D = self.activation(D)
        if self.slim:
            D = self.D_5x1(D)
            D = self.D_1x5(D)
        else:
            D = self.D_5x5(D)
        if self.seperable:
            D = self.D_1x1_2(D)
        D = self.activation(D)

        if self.slim:
            D = self.D_7x1(D)
            D = self.D_1x7(D)
        else:
            D = self.D_7x7(D)
        if self.seperable:
            D = self.D_1x1_3(D)
        D = self.activation(D)


        x = torch.concat((A, B, C, D), dim=1) + org

        return x





class EncoderC(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(EncoderC, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.B_1x1_1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0, bias=False)
        self.B_3x3 = nn.Conv2d(channels//2,channels//2, kernel_size=3, padding=1, stride=2, groups=channels//2 if self.seperable else 1, bias=False)
        self.B_3x1 = nn.Conv2d(channels//2,channels//2, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels//2 if self.seperable else 1, bias=False)
        self.B_1x3 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels//2 if self.seperable else 1, bias=False)
        self.B_1x1_2 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0, bias=False)


        self.C_1x1_1 = nn.Conv2d(channels,channels//2, kernel_size=1, padding=0, bias=False)
        self.C_7x7 = nn.Conv2d(channels//2,channels//2, kernel_size=7, padding=3, groups=channels//2 if self.seperable else 1, bias=False)
        self.C_7x1 = nn.Conv2d(channels//2,channels//2, kernel_size=(7,1), padding=(3,0), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x7 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,7), padding=(0,3), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0, bias=False)

        self.C_3x3 = nn.Conv2d(channels//2,channels//2, kernel_size=3, stride=2, padding=1, groups=channels//2 if self.seperable else 1, bias=False)
        self.C_3x1 = nn.Conv2d(channels//2, channels//2, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x3 = nn.Conv2d(channels//2,channels//2, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x1_3 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0, bias=False)


        self.skip = nn.Conv2d(channels,channels*2, kernel_size=3, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_3x1 = nn.Conv2d(channels,channels*2, kernel_size=(3,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_1x3 = nn.Conv2d(channels*2,channels*2, kernel_size=(1,3), padding=(0,1), stride=(1,2), groups=channels*2 if self.seperable else 1, bias=False)
        self.skip_1x1 = nn.Conv2d(channels*2,channels*2, kernel_size=1, padding=0, bias=False)



        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = self.A_pool(x)
        A = self.activation(A)

        # B branch
        B = self.B_1x1_1(x)
        B = self.activation(B)
        if self.slim:
            B = self.B_3x1(B)
            B = self.B_1x3(B)
        else:
            B = self.B_3x3(B)
        if self.seperable:
            B = self.B_1x1_2(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if self.slim:
            C = self.C_7x1(C)
            C = self.C_1x7(C)
        else:
            C = self.C_7x7(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)
        
        if self.slim:
            C = self.C_3x1(C)
            C = self.C_1x3(C)
        else:
            C = self.C_3x3(C)
        if self.seperable:
            C = self.C_1x1_3(C)
        C = self.activation(C)

        # skip
        if self.slim:
            skip = self.skip_slim_3x1(x)
            skip = self.skip_slim_1x3(skip)
        else:
            skip = self.skip(x)
        if self.seperable:
            skip = self.skip_1x1(skip)

        x = torch.concat((A, B, C), dim=1) + skip

        return x
    



class DecoderC(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(DecoderC, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.A_1x1 = nn.Conv2d(channels,channels//8, kernel_size=1, padding=0, bias=False)

        self.B_1x1_1 = nn.Conv2d(channels,channels//4, kernel_size=1, padding=0, bias=False)
        self.B_4x4 = nn.ConvTranspose2d(channels//4,channels//4, kernel_size=4, padding=1, stride=2, groups=channels//4 if self.seperable else 1, bias=False)
        self.B_4x1 = nn.ConvTranspose2d(channels//4,channels//4, kernel_size=(4,1), padding=(1,0), stride=(2,1), groups=channels//4 if self.seperable else 1, bias=False)
        self.B_1x4 = nn.ConvTranspose2d(channels//4,channels//4, kernel_size=(1,4), padding=(0,1), stride=(1,2), groups=channels//4 if self.seperable else 1, bias=False)
        self.B_1x1_2 = nn.Conv2d(channels//4,channels//4, kernel_size=1, padding=0, bias=False)


        self.C_1x1_1 = nn.Conv2d(channels,channels//8, kernel_size=1, padding=0, bias=False)
        self.C_7x7 = nn.Conv2d(channels//8,channels//8, kernel_size=7, padding=3, groups=channels//8 if self.seperable else 1, bias=False)
        self.C_7x1 = nn.Conv2d(channels//8,channels//8, kernel_size=(7,1), padding=(3,0), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x7 = nn.Conv2d(channels//8,channels//8, kernel_size=(1,7), padding=(0,3), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//8,channels//8, kernel_size=1, padding=0, bias=False)

        self.C_4x4 = nn.ConvTranspose2d(channels//8,channels//8, kernel_size=4, stride=2, padding=1, groups=channels//8 if self.seperable else 1, bias=False)
        self.C_4x1 = nn.ConvTranspose2d(channels//8, channels//8, kernel_size=(4,1), padding=(1,0), stride=(2,1), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x4 = nn.ConvTranspose2d(channels//8,channels//8, kernel_size=(1,4), padding=(0,1), stride=(1,2), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x1_3 = nn.Conv2d(channels//8,channels//8, kernel_size=1, padding=0, bias=False)


        self.skip = nn.ConvTranspose2d(channels,channels//2, kernel_size=4, padding=1, stride=2, groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_4x1 = nn.ConvTranspose2d(channels,channels//2, kernel_size=(4,1), padding=(1,0), stride=(2,1), groups=channels if self.seperable else 1, bias=False)
        self.skip_slim_1x4 = nn.ConvTranspose2d(channels//2,channels//2, kernel_size=(1,4), padding=(0,1), stride=(1,2), groups=channels//2 if self.seperable else 1, bias=False)
        self.skip_1x1 = nn.Conv2d(channels//2,channels//2, kernel_size=1, padding=0, bias=False)



        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch
        A = self.A_upsample(x)
        A = self.A_1x1(A)
        A = self.activation(A)

        # B branch
        B = self.B_1x1_1(x)
        B = self.activation(B)
        if self.slim:
            B = self.B_4x1(B)
            B = self.B_1x4(B)
        else:
            B = self.B_4x4(B)
        if self.seperable:
            B = self.B_1x1_2(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if self.slim:
            C = self.C_7x1(C)
            C = self.C_1x7(C)
        else:
            C = self.C_7x7(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)
        
        if self.slim:
            C = self.C_4x1(C)
            C = self.C_1x4(C)
        else:
            C = self.C_4x4(C)
        if self.seperable:
            C = self.C_1x1_3(C)
        C = self.activation(C)

        # skip
        if self.slim:
            skip = self.skip_slim_4x1(x)
            skip = self.skip_slim_1x4(skip)
        else:
            skip = self.skip(x)
        if self.seperable:
            skip = self.skip_1x1(skip)
        x = torch.concat((A, B, C), dim=1) + skip

        return x
    
    






class ChannelReduce(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(ChannelReduce, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_1x1 = nn.Conv2d(channels, channels//8, kernel_size=1, padding=0, bias=False)

        self.B_1x1_1 = nn.Conv2d(channels, channels//8, kernel_size=1, padding=0, bias=False)
        self.B_5x5 = nn.Conv2d(channels//8, channels//8, kernel_size=5, stride=1, padding=2, groups=channels//8 if self.seperable else 1, bias=False)
        self.B_5x1 = nn.Conv2d(channels//8, channels//8, kernel_size=(5,1), stride=1, padding=(2,0), groups=channels//8 if self.seperable else 1, bias=False)
        self.B_1x5 = nn.Conv2d(channels//8, channels//8, kernel_size=(1,5), stride=1, padding=(0,2), groups=channels//8 if self.seperable else 1, bias=False)
        self.B_1x1_2 = nn.Conv2d(channels//8, channels//8, kernel_size=1, padding=0, bias=False)


        self.C_1x1_1 = nn.Conv2d(channels, channels//8, kernel_size=1, padding=0, bias=False)
        self.C_3x3 = nn.Conv2d(channels//8, channels//8, kernel_size=3, stride=1, padding=1, groups=channels//8 if self.seperable else 1, bias=False)
        self.C_3x1 = nn.Conv2d(channels//8, channels//8, kernel_size=(3,1), stride=1, padding=(1,0), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x3 = nn.Conv2d(channels//8, channels//8, kernel_size=(1,3), stride=1, padding=(0,1), groups=channels//8 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//8, channels//8, kernel_size=1, padding=0, bias=False)


        self.D_1x1_1 = nn.Conv2d(channels, channels//8, kernel_size=1, padding=0, bias=False)
        self.D_7x7 = nn.Conv2d(channels//8, channels//8, kernel_size=7, stride=1, padding=3, groups=channels//8 if self.seperable else 1, bias=False)
        self.D_7x1 = nn.Conv2d(channels//8, channels//8, kernel_size=(7,1), stride=1, padding=(3,0), groups=channels//8 if self.seperable else 1, bias=False)
        self.D_1x7 = nn.Conv2d(channels//8, channels//8, kernel_size=(1,7), stride=1, padding=(0,3), groups=channels//8 if self.seperable else 1, bias=False)
        self.D_1x1_2 = nn.Conv2d(channels//8, channels//8, kernel_size=1, padding=0, bias=False)


        self.skip_1x1 = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, bias=False)

        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch

        A = self.A_1x1(x)
        A = self.activation(A)

        # B branch
        B = self.B_1x1_1(x)
        B = self.activation(B)
        if self.slim:
            B = self.B_5x1(B)
            B = self.B_1x5(B)
        else:
            B = self.B_5x5(B)
        if self.seperable:
            B = self.B_1x1_2(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if self.slim:
            C = self.C_3x1(C)
            C = self.C_1x3(C)
        else:
            C = self.C_3x3(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)


        # D branch
        D = self.D_1x1_1(x)
        D = self.activation(D)
        if self.slim:
            D = self.D_7x1(D)
            D = self.D_1x7(D)
        else:
            D = self.D_7x7(D)
        if self.seperable:
            D = self.D_1x1_2(D)
        D = self.activation(D)


        skip = self.skip_1x1(org)

        x = torch.concat((A, B, C, D), dim=1) + skip

        return x



class ChannelIncreace(nn.Module): # Based on inception v4
    def __init__(
        self,
        channels: int,
        slim: bool = False,
        seperable: bool = False):
        super(ChannelIncreace, self).__init__()
        self.slim = slim
        self.seperable = seperable
        self.A_1x1 = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, bias=False)

        self.B_1x1_1 = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, bias=False)
        self.B_5x5 = nn.Conv2d(channels//2, channels//2, kernel_size=5, stride=1, padding=2, groups=channels//2 if self.seperable else 1, bias=False)
        self.B_5x1 = nn.Conv2d(channels//2, channels//2, kernel_size=(5,1), stride=1, padding=(2,0), groups=channels//2 if self.seperable else 1, bias=False)
        self.B_1x5 = nn.Conv2d(channels//2, channels//2, kernel_size=(1,5), stride=1, padding=(0,2), groups=channels//2 if self.seperable else 1, bias=False)
        self.B_1x1_2 = nn.Conv2d(channels//2, channels//2, kernel_size=1, padding=0, bias=False)


        self.C_1x1_1 = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, bias=False)
        self.C_3x3 = nn.Conv2d(channels//2, channels//2, kernel_size=3, stride=1, padding=1, groups=channels//2 if self.seperable else 1, bias=False)
        self.C_3x1 = nn.Conv2d(channels//2, channels//2, kernel_size=(3,1), stride=1, padding=(1,0), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x3 = nn.Conv2d(channels//2, channels//2, kernel_size=(1,3), stride=1, padding=(0,1), groups=channels//2 if self.seperable else 1, bias=False)
        self.C_1x1_2 = nn.Conv2d(channels//2, channels//2, kernel_size=1, padding=0, bias=False)


        self.D_1x1_1 = nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, bias=False)
        self.D_7x7 = nn.Conv2d(channels//2, channels//2, kernel_size=7, stride=1, padding=3, groups=channels//2 if self.seperable else 1, bias=False)
        self.D_7x1 = nn.Conv2d(channels//2, channels//2, kernel_size=(7,1), stride=1, padding=(3,0), groups=channels//2 if self.seperable else 1, bias=False)
        self.D_1x7 = nn.Conv2d(channels//2, channels//2, kernel_size=(1,7), stride=1, padding=(0,3), groups=channels//2 if self.seperable else 1, bias=False)
        self.D_1x1_2 = nn.Conv2d(channels//2, channels//2, kernel_size=1, padding=0, bias=False)


        self.skip_1x1 = nn.Conv2d(channels, channels*2, kernel_size=1, padding=0, bias=False)

        self.activation = nn.ELU()  

    def forward(self, x):
        org = x

        # A branch

        A = self.A_1x1(x)
        A = self.activation(A)

        # B branch
        B = self.B_1x1_1(x)
        B = self.activation(B)
        if self.slim:
            B = self.B_5x1(B)
            B = self.B_1x5(B)
        else:
            B = self.B_5x5(B)
        if self.seperable:
            B = self.B_1x1_2(B)
        B = self.activation(B)


        # C branch
        C = self.C_1x1_1(x)
        C = self.activation(C)
        if self.slim:
            C = self.C_3x1(C)
            C = self.C_1x3(C)
        else:
            C = self.C_3x3(C)
        if self.seperable:
            C = self.C_1x1_2(C)
        C = self.activation(C)


        # D branch
        D = self.D_1x1_1(x)
        D = self.activation(D)
        if self.slim:
            D = self.D_7x1(D)
            D = self.D_1x7(D)
        else:
            D = self.D_7x7(D)
        if self.seperable:
            D = self.D_1x1_2(D)
        D = self.activation(D)


        skip = self.skip_1x1(org)

        x = torch.concat((A, B, C, D), dim=1) + skip

        return x
    






class Compress(nn.Module): #depthwise Seperable Conv with down sampling
    def __init__(self,
        slim: bool = False,
        seperable: bool = False):
        super(Compress, self).__init__()
        self.slim = slim
        self.seperable = seperable
        ##############################################################
        #Same
        self._0 = Stem(4, slim=slim, seperable=seperable)
        self._1 = SameA(16, slim=slim, seperable=seperable)
        self._2 = SameA(16, slim=slim, seperable=seperable)
        self._3 = SameA(16, slim=slim, seperable=seperable)
        self._4 = SameA(16, slim=slim, seperable=seperable)
        self._5 = SameA(16, slim=slim, seperable=seperable)
        self._6 = SameA(16, slim=slim, seperable=seperable)
        self._5 = EncoderA(16, slim=slim, seperable=seperable)
        self._6 = SameA(32, slim=slim, seperable=seperable)
        self._7 = SameA(32, slim=slim, seperable=seperable)
        self._8 = SameA(32, slim=slim, seperable=seperable)
        self._9 = SameA(32, slim=slim, seperable=seperable)
        self._10 = EncoderA(32, slim=slim, seperable=seperable)


        self._11 = SameB(64, slim=slim, seperable=seperable)
        self._12 = SameB(64, slim=slim, seperable=seperable)
        self._13 = EncoderB(64, slim=slim, seperable=seperable)
        self._14 = SameB(128, slim=slim, seperable=seperable)
        self._15 = SameB(128, slim=slim, seperable=seperable)
        self._16 = EncoderB(128, slim=slim, seperable=seperable)

        self._17 = SameC(256, slim=slim, seperable=seperable)
        self._18 = SameC(256, slim=slim, seperable=seperable)
        self._19 = EncoderC(256, slim=slim, seperable=seperable)
        self._20 = SameC(512, slim=slim, seperable=seperable)
        self._21 = SameC(512, slim=slim, seperable=seperable)
        self._22 = ChannelReduce(512, slim=slim, seperable=seperable)
        
        ###############################################################
        #Decoder
        self._23 = ChannelIncreace(256, slim=slim, seperable=seperable)
        self._24 = SameC(512, slim=slim, seperable=seperable)
        self._25 = SameC(512, slim=slim, seperable=seperable)
        self._26 = DecoderC(512, slim=slim, seperable=seperable)
        self._27 = SameC(256, slim=slim, seperable=seperable)
        self._28 = SameC(256, slim=slim, seperable=seperable)
        
        self._29 = DecoderB(256, slim=slim, seperable=seperable)
        self._30 = SameB(128, slim=slim, seperable=seperable)
        self._31 = SameB(128, slim=slim, seperable=seperable)
        self._32 = DecoderB(128, slim=slim, seperable=seperable)
        self._33 = SameB(64, slim=slim, seperable=seperable)
        self._34 = SameB(64, slim=slim, seperable=seperable)

        self._35 = DecoderA(64, slim=slim, seperable=seperable)
        self._36 = SameA(32, slim=slim, seperable=seperable)
        self._37 = SameA(32, slim=slim, seperable=seperable)
        self._38 = SameA(32, slim=slim, seperable=seperable)
        self._39 = SameA(32, slim=slim, seperable=seperable)
        self._40 = DecoderA(32, slim=slim, seperable=seperable)
        self._41 = SameA(16, slim=slim, seperable=seperable)
        self._42 = SameA(16, slim=slim, seperable=seperable)
        #self._43 = DecoderA(16, slim=slim, seperable=seperable)
        #self._44 = SameA(8, slim=slim, seperable=seperable)
        #self._45 = SameA(8, slim=slim, seperable=seperable)
        self._46 = DeStem(slim=slim, seperable=seperable)
        
        self.activation = nn.ELU()   


        #self.initialize_weights()
        

    def forward(self, x):
        x = self._0(x)
        x = self._1(x)
        x = self._2(x)
        x = self._3(x)
        x = self._4(x)
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
        x = self._20(x)
        x = self._21(x)
        x = self._22(x)


        x = self._23(x)
        x = self._24(x)
        x = self._25(x)
        x = self._26(x)
        x = self._27(x)
        x = self._28(x)
        x = self._29(x)
        x = self._30(x)
        x = self._31(x)
        x = self._32(x)
        x = self._33(x)
        x = self._34(x)
        x = self._35(x)
        x = self._36(x)
        x = self._37(x)
        x = self._38(x)
        x = self._39(x) 
        x = self._40(x)
        x = self._41(x)
        x = self._42(x)
        #x = self._43(x)
        #x = self._44(x)
        #x = self._45(x)
        x = self._46(x)

        
        
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight)