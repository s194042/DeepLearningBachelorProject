#from . import images
import numpy as np
import copy

def applyFilet(image,filter):
    image_c = copy.deepcopy(image).astype('float64')
    image_r = image_c[:,:,0]
    image_g = image_c[:,:,1]
    image_b = image_c[:,:,2]

    
    image_c[:,:,0] = np.fft.irfft2(np.fft.rfft2(image_r) * np.fft.rfft2(filter, image_r.shape))
    image_c[:,:,1] = np.fft.irfft2(np.fft.rfft2(image_g) * np.fft.rfft2(filter, image_g.shape))
    image_c[:,:,2] = np.fft.irfft2(np.fft.rfft2(image_b) * np.fft.rfft2(filter, image_b.shape))

    image_c[image_c<0] = 0
    image_c[image_c>255] = 255
    image_c = np.rint(image_c)
    return image_c.astype("uint8")  

def sharpen(image, times = 1):
    filter = np.array([ [0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
    image_c = copy.deepcopy(image)
    for _ in range(times):
        image_c = applyFilet(image_c, filter)
    return image_c


def box_blur_3x3(image, times = 1):
    div = 9
    filter = np.array([ [1/div,1/div,1/div],
                        [1/div,1/div,1/div],
                        [1/div,1/div,1/div]])
    image_c = copy.deepcopy(image)
    for _ in range(times):
        image_c = applyFilet(image_c, filter)
    return image_c
    

def box_blur_5x5(image, times = 1):
    div = 25
    filter = np.array([ [1/div,1/div,1/div,1/div,1/div],
                        [1/div,1/div,1/div,1/div,1/div],
                        [1/div,1/div,1/div,1/div,1/div],
                        [1/div,1/div,1/div,1/div,1/div],
                        [1/div,1/div,1/div,1/div,1/div]])
    image_c = copy.deepcopy(image)
    for _ in range(times):
        image_c = applyFilet(image_c, filter)
    return image_c


def Gaussioan_blur_3x3(image, times = 1):
    div = 16
    filter = np.array([ [1/div,2/div,1/div],
                        [2/div,4/div,2/div],
                        [1/div,2/div,1/div]])
    image_c = copy.deepcopy(image)
    for _ in range(times):
        image_c = applyFilet(image_c, filter)
    return image_c


def Gaussioan_blur_5x5(image, times = 1):
    div = 256
    filter = np.array([ [1/div,4/div,6/div,4/div,1/div],
                        [4/div,16/div,24/div,16/div,4/div],
                        [6/div,24/div,36/div,24/div,6/div],
                        [4/div,16/div,24/div,16/div,4/div],
                        [1/div,4/div,6/div,4/div,1/div]])
    image_c = copy.deepcopy(image)
    for _ in range(times):
        image_c = applyFilet(image_c, filter)
    return image_c


def Unsharp_masking_5x5(image, times = 1):
    div = -256
    filter = np.array([ [1/div,4/div,6/div,4/div,1/div],
                        [4/div,16/div,24/div,16/div,4/div],
                        [6/div,24/div,-476/div,24/div,6/div],
                        [4/div,16/div,24/div,16/div,4/div],
                        [1/div,4/div,6/div,4/div,1/div]])
    image_c = copy.deepcopy(image)
    for _ in range(times):
        image_c = applyFilet(image_c, filter)
    return image_c


def salt(image, rate):
    image_c = copy.deepcopy(image)
    row,col,ch = image_c.shape
    rnd = np.random.rand(row,col,ch)
    image_c[rnd<rate] = 255
    return image_c


def peper(image, rate):
    image_c = copy.deepcopy(image)
    row,col,ch = image_c.shape
    rnd = np.random.rand(row,col,ch)
    image_c[rnd<rate] = 0
    return image_c


def Guasian_noice(image, rate):
    image_c = copy.deepcopy(image).astype('float64')
    row,col,ch = image_c.shape
    gauss = np.random.normal(0, rate*100, image_c.shape)
    gauss = gauss.reshape(row,col,ch)
    image_c = image_c + gauss
    image_c[image_c<0] = 0
    image_c[image_c>255] = 255
    image_c = np.rint(image_c)
    return image_c.astype("uint8")


def Poiss_noice(image, rate):
    image_c = copy.deepcopy(image).astype('float64')
    row,col,ch = image_c.shape
    poisson = np.random.poisson(rate*100, image_c.shape)
    poisson = poisson.reshape(row,col,ch)
    image_c = image_c + poisson
    image_c[image_c<0] = 0
    image_c[image_c>255] = 255
    image_c = np.rint(image_c)
    return image_c.astype("uint8")   


def Exp_noice(image, rate):
    image_c = copy.deepcopy(image).astype('float64')
    row,col,ch = image_c.shape
    exponential = np.random.exponential(rate*100, image_c.shape)
    exponential = exponential.reshape(row,col,ch)
    image_c = image_c + exponential
    image_c[image_c<0] = 0
    image_c[image_c>255] = 255
    image_c = np.rint(image_c)
    return image_c.astype("uint8")   


def Uniform_noice(image, max):
    image_c = copy.deepcopy(image).astype('float64')
    row,col,ch = image_c.shape
    uniform = np.random.uniform(-max, max, image_c.shape)
    uniform = uniform.reshape(row,col,ch)
    image_c = image_c + uniform
    image_c[image_c<0] = 0
    image_c[image_c>255] = 255
    image_c = np.rint(image_c)
    return image_c.astype("uint8")


def uniform_add(image, amount):
    image_c = copy.deepcopy(image).astype('int32')
    image_c = image_c + amount
    image_c[image_c<0] = 0
    image_c[image_c>255] = 255
    return image_c.astype("uint8")  

def uniform_decimal_multiplication(image, amount):
    image_c = copy.deepcopy(image).astype('float64')
    image_c = image_c * amount
    image_c[image_c<0] = 0
    image_c[image_c>255] = 255
    image_c = np.rint(image_c)
    return image_c.astype("uint8")   