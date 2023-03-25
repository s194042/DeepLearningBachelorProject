import rawpy
import os
import numpy as np
import math
import torch

path_to_images = 'C:/Users/Rani/Desktop/ai_training_immages' # foldernames are 1 ... 16

def load_nef(path = 'C:/Users/Rani/Desktop/ai_training_immages' + "/1/_DSC0001.nef"):
    img = torch.from_numpy(rawpy.imread(path).postprocess()).float().to(torch.device('cuda'))
    
    width, height, _ = img.shape
    if width < height:
        print(img.shape)
        return img
    return torch.rot90(img)


def get_compressed_imgs(img, width=512, height=768):
    original_width, original_height, _ = img.size()
    sizes = min(original_height//height, original_width//width)
    cut_img = img[:original_width//width*width, :original_height//height*height]
    cut_width, cut_height, _ = cut_img.size()
    for size in range(1,sizes+1):
        if cut_width//size % width == 0 and cut_height//size % height == 0:
            yield cut_img[::size,::size]


def get_split(img, width=512, height=768):
    org_width, org_height, _ = img.size()
    for i in range(org_width//width):
        for j in range(org_height//height):
            yield img[i*width:i*width+width, j*height:j*height+height]


def get_all_splits(img, width=512, height=768):
    for compressed_img in get_compressed_imgs(img):
        for img in get_split(compressed_img):
            yield img
        


total_transformations = 4*50*71 #50 is the cuts and downsample, 4 is the flips, 71 is the transforms

def get_all_images(path_to_images, width=512, height=768, start = 0):
    folders = [str(i) for i in range(1,15)]
    i = 0
    for f in folders:
        for _, _, pics in os.walk(path_to_images + '/' + f):
            for pic in pics:
                
                if i < math.ceil(start/total_transformations):
                    i += 1
                    continue
                torch_img = load_nef(path_to_images + '/' + f +'/' + pic)
                for img in get_all_splits(torch_img):
                    yield img


def get_all_flips_of_image(image):
    yield image
    yield torch.fliplr(image)
    image = torch.flipud(image)
    yield image
    yield torch.fliplr(image)


def get_all_imgs_with_everything(path_to_images, start = 0):
    for imgs in get_all_images(path_to_images, start=start):
        for flips in get_all_flips_of_image(imgs):
            yield flips