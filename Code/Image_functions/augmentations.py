#from . import images
import numpy as np
import cupy as cp
import copy
import torch

torch.set_grad_enabled(False)

#@torch.jit.script
def applyFilet(image,filter):
    image_c = torch.clone(image)
    image_r = image_c[:,:,0]
    image_g = image_c[:,:,1]
    image_b = image_c[:,:,2]
    image_c[:,:,0] = torch.fft.irfft2(torch.multiply(torch.fft.rfft2(image_r), torch.fft.rfft2(filter, image_r.shape)))  ## Try just convolution
    image_c[:,:,1] = torch.fft.irfft2(torch.multiply(torch.fft.rfft2(image_g), torch.fft.rfft2(filter, image_g.shape)))
    image_c[:,:,2] = torch.fft.irfft2(torch.multiply(torch.fft.rfft2(image_b), torch.fft.rfft2(filter, image_b.shape)))
    
    
    torch.clamp(image_c, min=0., max=255., out=image_c)
    image_c = torch.round(image_c)
    return image_c

#@torch.jit.script
def applyFilter(image, filter, times: int = 1):
    for _ in range(times):
        image = applyFilet(image, filter)
    return image

#@torch.jit.script
def sharpen(image, times: int = 1):
    filter = torch.tensor([    [0,-1,0],
                                [-1,5,-1],
                                [0,-1,0]], device=torch.device('cuda'))
    
    return applyFilter(image, filter, times)

#@torch.jit.script
def box_blur_3x3(image, times: int = 1):
    div = 9
    filter = torch.div(torch.ones(3,3, device=torch.device('cuda')),div)
    return applyFilter(image, filter, times)
    
#@torch.jit.script
def box_blur_5x5(image, times: int = 1):
    div = 25
    filter = torch.div(torch.ones(5,5, device=torch.device('cuda')),div)
    
    return applyFilter(image, filter, times)

#@torch.jit.script
def Gaussioan_blur_3x3(image, times: int = 1):
    div = 16
    filter = torch.div(torch.tensor([   [1,2,1],
                                        [2,4,2],
                                        [1,2,1]], device=torch.device('cuda')), div)
    
    return applyFilter(image, filter, times)

#@torch.jit.script
def Gaussioan_blur_5x5(image, times: int = 1):
    div = 256
    filter = torch.div(torch.tensor([   [1,4,6,4,1],
                                        [4,16,24,16,4],
                                        [6,24,36,24,6],
                                        [4,16,24,16,4],
                                        [1,4,6,4,1]], device=torch.device('cuda')), div)
    
    return applyFilter(image, filter, times)

#@torch.jit.script
def Unsharp_masking_5x5(image, times: int = 1):
    div = -256
    filter = torch.div(torch.tensor([   [1,4,6,4,1],
                                        [4,16,24,16,4],
                                        [6,24,-476,24,6],
                                        [4,16,24,16,4],
                                        [1,4,6,4,1]], device=torch.device('cuda')), div)
    
    return applyFilter(image, filter, times)

#@torch.jit.script
def salt(image, rate: float):
    image_c = torch.clone(image)
    rnd = torch.rand(image_c.size(), device=torch.device('cuda'))
    image_c[rnd<rate] = 255
    return image_c

#@torch.jit.script
def peper(image, rate: float):
    image_c = torch.clone(image)
    rnd = torch.rand(image_c.size(), device=torch.device('cuda'))
    image_c[rnd<rate] = 0
    return image_c

#@torch.jit.script
def Guasian_noice(image, rate: float):
    rate *= 100
    image_c = torch.clone(image)
    gauss = torch.randn(image_c.size(), device=torch.device('cuda'))
    image_c = torch.add(image_c, torch.multiply(gauss, rate))
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c

#@torch.jit.script
def Poiss_noice(image, rate: float):
    image_c = torch.clone(image)
    rate *= 100
    poisson = torch.poisson(torch.multiply(torch.rand(image_c.size(), device=torch.device('cuda')), rate))
    image_c = torch.add(image_c, poisson)
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c 

#@torch.jit.script
def Exp_noice(image, rate: float):
    image_c = torch.clone(image)
    rate *= 100
    image_c = torch.add(image_c,torch.multiply(torch.exp(torch.randn(image.size(), device=torch.device('cuda'))), rate))
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c 


#@torch.jit.script
def Uniform_noice(image, max: float):
    image_c = torch.clone(image)
    uni = torch.rand(image_c.size(), device=torch.device('cuda'))
    image_c = torch.add(image_c,torch.multiply(torch.subtract(uni, 0.5), max*2))
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c

##@torch.jit.script
def uniform_add(image, amount: float):
    image_c = torch.add(torch.clone(image), amount)
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c 

##@torch.jit.script
def uniform_decimal_multiplication(image, amount: float):
    image_c = torch.multiply(torch.clone(image), amount)
    torch.clamp(image_c, min=0., max=255., out=image_c)
    return image_c 