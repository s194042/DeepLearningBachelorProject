#from . import images
import numpy as np
import cupy as cp
import copy

def applyFilet(image,filter, cupy=False):
    if cupy:
        image_r = image[:,:,0]
        image_g = image[:,:,1]
        image_b = image[:,:,2]
        image[:,:,0] = cp.fft.irfft2(cp.multiply(cp.fft.rfft2(image_r), cp.fft.rfft2(filter, cp.shape(image_r))))
        image[:,:,1] = cp.fft.irfft2(cp.multiply(cp.fft.rfft2(image_g), cp.fft.rfft2(filter, cp.shape(image_g))))
        image[:,:,2] = cp.fft.irfft2(cp.multiply(cp.fft.rfft2(image_b), cp.fft.rfft2(filter, cp.shape(image_b))))
        cp.clip(image, a_min=0, a_max=255, out=image)
        image_c = cp.rint(image)
        
    else:
        image_c = copy.deepcopy(image).astype('float64')
        image_r = image_c[:,:,0]
        image_g = image_c[:,:,1]
        image_b = image_c[:,:,2]
        image_c[:,:,0] = np.fft.irfft2(np.fft.rfft2(image_r) * np.fft.rfft2(filter, image_r.shape))
        image_c[:,:,1] = np.fft.irfft2(np.fft.rfft2(image_g) * np.fft.rfft2(filter, image_g.shape))
        image_c[:,:,2] = np.fft.irfft2(np.fft.rfft2(image_b) * np.fft.rfft2(filter, image_b.shape))
        np.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = np.rint(image_c)
    return image_c

def applyFilter(image, filter, times=1, cupy=False):
    if cupy:
        filter = cp.asfarray(filter)
        image_c = cp.asfarray(image)
        for _ in range(times):
            image_c = applyFilet(image_c, filter, cupy)
        image_c = cp.asnumpy(image_c)

    else:
        filter = np.array(filter)
        image_c = copy.deepcopy(image)
        for _ in range(times):
            image_c = applyFilet(image_c, filter, cupy)
    return image_c.astype(np.uint8)

def sharpen(image, times = 1, cupy=False):
    filter = [  [0,-1,0],
                [-1,5,-1],
                [0,-1,0]]
    
    return applyFilter(image, filter, times, cupy)


def box_blur_3x3(image, times = 1, cupy=False):
    div = 9
    filter = [  [1/div,1/div,1/div],
                [1/div,1/div,1/div],
                [1/div,1/div,1/div]]
    
    return applyFilter(image, filter, times, cupy)
    

def box_blur_5x5(image, times = 1, cupy=False):
    div = 25
    filter = [ [1/div,1/div,1/div,1/div,1/div],
                [1/div,1/div,1/div,1/div,1/div],
                [1/div,1/div,1/div,1/div,1/div],
                [1/div,1/div,1/div,1/div,1/div],
                [1/div,1/div,1/div,1/div,1/div]]
    
    return applyFilter(image, filter, times, cupy)


def Gaussioan_blur_3x3(image, times = 1, cupy=False):
    div = 16
    filter = np.array([ [1/div,2/div,1/div],
                        [2/div,4/div,2/div],
                        [1/div,2/div,1/div]])
    
    return applyFilter(image, filter, times, cupy)


def Gaussioan_blur_5x5(image, times = 1, cupy=False):
    div = 256
    filter = [  [1/div,4/div,6/div,4/div,1/div],
                [4/div,16/div,24/div,16/div,4/div],
                [6/div,24/div,36/div,24/div,6/div],
                [4/div,16/div,24/div,16/div,4/div],
                [1/div,4/div,6/div,4/div,1/div]]
    
    return applyFilter(image, filter, times, cupy)


def Unsharp_masking_5x5(image, times = 1, cupy=False):
    div = -256
    filter = [  [1/div,4/div,6/div,4/div,1/div],
                [4/div,16/div,24/div,16/div,4/div],
                [6/div,24/div,-476/div,24/div,6/div],
                [4/div,16/div,24/div,16/div,4/div],
                [1/div,4/div,6/div,4/div,1/div]]
    
    return applyFilter(image, filter, times, cupy)

def salt(image, rate, cupy=False):
    if cupy:
        image_c = cp.array(image)
        row,col,ch = cp.shape(image_c)
        rnd = cp.random.rand(row,col,ch)
        image_c[rnd<rate] = 255
        image_c = cp.asnumpy(image_c)

    else:
        image_c = copy.deepcopy(image)
        row,col,ch = cp.shape(image_c)
        rnd = np.random.rand(row,col,ch)
        image_c[rnd<rate] = 255
    return image_c


def peper(image, rate, cupy=False):
    if cupy:
        image_c = cp.array(image)
        row,col,ch = cp.shape(image_c)
        rnd = cp.random.rand(row,col,ch)
        image_c[rnd<rate] = 0
        image_c = cp.asnumpy(image_c)

    else:
        image_c = copy.deepcopy(image)
        row,col,ch = cp.shape(image_c)
        rnd = np.random.rand(row,col,ch)
        image_c[rnd<rate] = 0
    return image_c


def Guasian_noice(image, rate, cupy=False):
    if cupy:
        image_c = cp.asfarray(image)
        gauss = cp.random.normal(0, rate*100, cp.shape(image_c))
        image_c = cp.add(image_c, gauss)
        cp.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = cp.rint(image_c)
        image_c = cp.asnumpy(image_c)
    else:
        image_c = copy.deepcopy(image).astype('float64')
        gauss = np.random.normal(0, rate*100, image_c.shape)
        image_c = image_c + gauss
        np.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = np.rint(image_c)
    return image_c.astype("uint8")


def Poiss_noice(image, rate, cupy=False):
    if cupy:
        image_c = cp.asfarray(image)
        poisson = cp.random.poisson(rate*100, cp.shape(image_c))
        image_c = cp.add(image_c, poisson)
        cp.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = cp.rint(image_c)
        image_c = cp.asnumpy(image_c)
    else:
        image_c = copy.deepcopy(image).astype('float64')
        poisson = np.random.poisson(rate*100, image_c.shape)
        image_c = image_c + poisson
        np.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = np.rint(image_c)
    return image_c.astype("uint8")   


def Exp_noice(image, rate, cupy=False):
    if cupy:
        image_c = cp.asfarray(image)
        exponential = cp.random.exponential(rate*100, cp.shape(image_c))
        image_c = cp.add(image_c, exponential)
        cp.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = cp.rint(image_c)
        image_c = cp.asnumpy(image_c)
    else:
        image_c = copy.deepcopy(image).astype('float64')
        exponential = np.random.exponential(rate*100, image_c.shape)
        image_c = image_c + exponential
        np.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = np.rint(image_c)
    return image_c.astype("uint8")   


def Uniform_noice(image, max, cupy=False):
    if cupy:
        image_c = cp.asfarray(image)
        uniform = cp.random.uniform(-max, max, cp.shape(image_c))
        image_c = cp.add(image_c, uniform)
        cp.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = cp.rint(image_c)
        image_c = cp.asnumpy(image_c)
    else:
        image_c = copy.deepcopy(image).astype('float64')
        uniform = np.random.uniform(-max, max, image_c.shape)
        image_c = image_c + uniform
        np.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = np.rint(image_c)
    return image_c.astype("uint8")


def uniform_add(image, amount, cupy=False):
    cupy = False # the overhead of cupy makes this simple one slower than no cupy
    if cupy:
        image_c = cp.asfarray(image)
        image_c = cp.add(image_c, amount)
        cp.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = cp.asnumpy(image_c)
    else:
        image_c = copy.deepcopy(image).astype('int32')
        image_c = image_c + amount
        np.clip(image_c, a_min=0, a_max=255, out=image_c)
    
    return image_c.astype("uint8")  

def uniform_decimal_multiplication(image, amount, cupy=False):
    cupy = False # the overhead of cupy makes this simple one slower than no cupy
    if cupy:
        image_c = cp.asfarray(image)
        image_c = cp.multiply(image_c, amount)
        cp.clip(image_c, a_min=0, a_max=255, out=image_c)
        image_c = cp.asnumpy(image_c)
    else:
        image_c = copy.deepcopy(image).astype('int32')
        image_c = image_c * amount
        np.clip(image_c, a_min=0, a_max=255, out=image_c)
    return image_c.astype("uint8")   