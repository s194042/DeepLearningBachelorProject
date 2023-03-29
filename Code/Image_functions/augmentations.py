#from . import images
import numpy as np
import cupy as cp
import copy
import torch

torch.set_grad_enabled(False)

#@torch.jit.script
def applyFilet(image,filter):
    """
    image_r = image_c[:,:,0]
    image_g = image_c[:,:,1]
    image_b = image_c[:,:,2]
    image_c[:,:,0] = torch.fft.irfft2(torch.multiply(torch.fft.rfft2(image_r), torch.fft.rfft2(filter, image_r.shape)))  ## Try just convolution
    image_c[:,:,1] = torch.fft.irfft2(torch.multiply(torch.fft.rfft2(image_g), torch.fft.rfft2(filter, image_g.shape)))
    image_c[:,:,2] = torch.fft.irfft2(torch.multiply(torch.fft.rfft2(image_b), torch.fft.rfft2(filter, image_b.shape)))
    """
    
    #print(image_c.shape)
    #print(filter.shape)
    image = torch.nn.functional.conv2d(image, filter, padding='same', groups=3)#, memory_format=torch.channels_last)
    torch.clamp(image, min=0., max=255., out=image)
    image = torch.round(image)
    return image

#@torch.jit.script
def applyFilter(image, filter, times: int = 1):
    image_c = torch.clone(image)
    image_c = image_c.permute(2,0,1)
    image_c = torch.unsqueeze(image_c, dim=0)
    image_c.to(memory_format=torch.channels_last)
    filter = torch.unsqueeze(filter, 1)
    filter = filter.to(memory_format=torch.channels_last)

    for _ in range(times):
        image_c = applyFilet(image_c, filter)
    image_c = torch.squeeze(image_c, 0)
    image_c = image_c.permute(1,2,0)
    return image_c

#@torch.jit.script
def sharpen(image, times: int = 1):
    filter = torch.tensor([[    [0,-1,0],
                                [-1,5,-1],
                                [0,-1,0]],
                                
                                [[0,-1,0],
                                [-1,5,-1],
                                [0,-1,0]],
                                
                                [[0,-1,0],
                                [-1,5,-1],
                                [0,-1,0]]], device=torch.device('cuda'), dtype=torch.float)
    
    return applyFilter(image, filter, times)

#@torch.jit.script
def box_blur_3x3(image, times: int = 1):
    div = 9
    filter = torch.div(torch.ones(3, 3, 3, device=torch.device('cuda'), dtype=torch.float),div)
    return applyFilter(image, filter, times)
    
#@torch.jit.script
def box_blur_5x5(image, times: int = 1):
    div = 25
    filter = torch.div(torch.ones(3,5, 5, device=torch.device('cuda'), dtype=torch.float),div)
    return applyFilter(image, filter, times)

#@torch.jit.script
def Gaussioan_blur_3x3(image, times: int = 1):
    div = 16
    filter = torch.div(torch.tensor([[   [1,2,1],
                                        [2,4,2],
                                        [1,2,1]],
                                        
                                        [   [1,2,1],
                                        [2,4,2],
                                        [1,2,1]],
                                        
                                        [   [1,2,1],
                                        [2,4,2],
                                        [1,2,1]]], device=torch.device('cuda'), dtype=torch.float), div)
    return applyFilter(image, filter, times)

#@torch.jit.script
def Gaussioan_blur_5x5(image, times: int = 1):
    div = 256
    filter = torch.div(torch.tensor([[   [1,4,6,4,1],
                                        [4,16,24,16,4],
                                        [6,24,36,24,6],
                                        [4,16,24,16,4],
                                        [1,4,6,4,1]],
                                        
                                        [   [1,4,6,4,1],
                                        [4,16,24,16,4],
                                        [6,24,36,24,6],
                                        [4,16,24,16,4],
                                        [1,4,6,4,1]],
                                        
                                        [   [1,4,6,4,1],
                                        [4,16,24,16,4],
                                        [6,24,36,24,6],
                                        [4,16,24,16,4],
                                        [1,4,6,4,1]]], device=torch.device('cuda'), dtype=torch.float), div)
    return applyFilter(image, filter, times)

#@torch.jit.script
def Unsharp_masking_5x5(image, times: int = 1):
    div = -256
    filter = torch.div(torch.tensor([[   [1,4,6,4,1],
                                        [4,16,24,16,4],
                                        [6,24,-476,24,6],
                                        [4,16,24,16,4],
                                        [1,4,6,4,1]],
                                        
                                        [   [1,4,6,4,1],
                                        [4,16,24,16,4],
                                        [6,24,-476,24,6],
                                        [4,16,24,16,4],
                                        [1,4,6,4,1]],
                                        
                                        [   [1,4,6,4,1],
                                        [4,16,24,16,4],
                                        [6,24,-476,24,6],
                                        [4,16,24,16,4],
                                        [1,4,6,4,1]]], device=torch.device('cuda'), dtype=torch.float), div)
    return applyFilter(image, filter, times)

#@torch.jit.script
def salt(image, rand, rate: float):
    image_c = torch.clone(image)
    image_c[rand<rate] = 255
    return image_c

#@torch.jit.script
def peper(image, rand, rate: float):
    image_c = torch.clone(image)
    image_c[rand<rate] = 0
    return image_c

#@torch.jit.script
def Guasian_noice(image, randn, rate: float):
    rate *= 100
    image_c = torch.clone(image)
    image_c = torch.add(image_c, torch.multiply(randn, rate))
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c

#@torch.jit.script
def Poiss_noice(image, rand, rate: float):
    image_c = torch.clone(image)
    rate *= 100
    poisson = torch.poisson(torch.multiply(rand, rate))
    image_c = torch.add(image_c, poisson)
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c 

#@torch.jit.script
def Exp_noice(image, randn, rate: float):
    image_c = torch.clone(image)
    rate *= 100
    image_c = torch.add(image_c,torch.multiply(torch.exp(randn), rate))
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c 


#@torch.jit.script
def Uniform_noice(image, rand, max: float):
    image_c = torch.clone(image)
    image_c = torch.add(image_c,torch.multiply(torch.subtract(rand, 0.5), max*2))
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c

#@torch.jit.script
def uniform_add(image, amount: float):
    image_c = torch.add(torch.clone(image), amount)
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c 

#@torch.jit.script
def uniform_decimal_multiplication(image, amount: float):
    image_c = torch.multiply(torch.clone(image), amount)
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c 