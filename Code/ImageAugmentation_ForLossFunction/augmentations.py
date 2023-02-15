#from . import images
import numpy as np
import copy

def applyFilet(image,filter):
    image_c = copy.deepcopy(image)
    image_r = image_c[:,:,0]
    image_g = image_c[:,:,1]
    image_b = image_c[:,:,2]

    
    image_c[:,:,0] = np.fft.irfft2(np.fft.rfft2(image_r) * np.fft.rfft2(filter, image_r.shape))
    image_c[:,:,1] = np.fft.irfft2(np.fft.rfft2(image_g) * np.fft.rfft2(filter, image_g.shape))
    image_c[:,:,2] = np.fft.irfft2(np.fft.rfft2(image_b) * np.fft.rfft2(filter, image_b.shape))

    return image_c

def sharpen(image):
    filter = np.array([ [0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
    return applyFilet(image, filter)


def box_blur_3x3(image):
    div = 9
    filter = np.array([ [1/div,1/div,1/div],
                        [1/div,1/div,1/div],
                        [1/div,1/div,1/div]])
    return applyFilet(image, filter)


def Gaussioan_blur_3x3(image):
    div = 16
    filter = np.array([ [1/div,2/div,1/div],
                        [2/div,4/div,2/div],
                        [1/div,2/div,1/div]])
    return applyFilet(image, filter)


def Gaussioan_blur_5x5(image):
    div = 256
    filter = np.array([ [1/div,4/div,6/div,4/div,1/div],
                        [4/div,16/div,24/div,16/div,4/div],
                        [6/div,24/div,36/div,24/div,6/div],
                        [4/div,16/div,24/div,16/div,4/div],
                        [1/div,4/div,6/div,4/div,1/div]])
    return applyFilet(image, filter)


def Unsharp_masking_5x5(image):
    div = -256
    filter = np.array([ [1/div,4/div,6/div,4/div,1/div],
                        [4/div,16/div,24/div,16/div,4/div],
                        [6/div,24/div,-476/div,24/div,6/div],
                        [4/div,16/div,24/div,16/div,4/div],
                        [1/div,4/div,6/div,4/div,1/div]])
    return applyFilet(image, filter)