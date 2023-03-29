import rawpy
import os
import numpy as np
import math
import torch

path_to_images = 'C:/Users/Rani/Desktop/ai_training_immages/_0' # foldernames are 1 ... 16

@torch.jit.script
def load_nef(path: str = 'C:/Users/Rani/Desktop/ai_training_immages/1_pt/0.pt'):
    path: str = 'C:/Users/Rani\Desktop/ai_training_immages/1_pt/0.pt'
    img = torch.load(path)
    return img

@torch.jit.script
def get_compressed_imgs(cut_img, size: int):
    return cut_img[::size,::size]

@torch.jit.script
def get_split(img, i: int, j: int): 
    width=512 
    height=768
    return img[i*width:i*width+width, j*height:j*height+height]


#def get_all_splits(img, split: int):
    
#    for compressed_img in get_compressed_imgs(img):
#        for img in get_split(compressed_img):
#            yield img
        


#total_transformations = 4*50*106 #50 is the cuts and downsample, 4 is the flips, 71 is the transforms
@torch.jit.script
def get_images(path_to_images:str, img:int,):
    return load_nef(path_to_images +'/' + str(img) + ".NEF")#torch.load(path_to_images +'/' + pic).float()#load_nef(path_to_images +'/' + pic)


@torch.jit.script
def get_all_flips_of_image(image, flip: int):
    if flip == 0:
        return image
    elif flip == 1:
        return torch.fliplr(image)
    im = torch.flipud(image)
    if flip == 2:
        return im
    else:
        return torch.fliplr(image)


@torch.jit.script
def get_img_with_everything_no_load(img, everything:int = 0):
    img_nr = everything//50//4 #50 splits and 4 flips
    split = everything//4 % 50
    flip = everything % 4
    
    #original_width, original_height, _ = 3264, 4928
    #sizes = min(original_height//height, original_width//width)
    cut_img = img[:3072, :4608]
    #cut_width, cut_height, _ = cut_img.size()
    #sizes = [size for size in range(1, sizes+1) if cut_width//size % width == 0 and cut_height//size % height == 0]
    sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 6]
    i_s = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 0]
    j_s = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0]
    cut_img = get_compressed_imgs(cut_img, size=sizes[split])
    i = i_s[split]
    j = j_s[split]
    cut_img = get_split(cut_img, i=i, j=j)
    cut_img = get_all_flips_of_image(cut_img, flip=flip)
    return cut_img
    

def get_all_imgs_with_everything(path, start:int= 0, end:int = 6645*50*4):
    img = load_nef(path)
    for i in range(start, end):
        yield get_img_with_everything_no_load(img, everything = int(i))