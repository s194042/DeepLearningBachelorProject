import rawpy
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image


path_to_images = 'C:/Users/Rani/Desktop/ai_training_immages' # foldernames are 1 ... 16

def load_nef(path):
    img = rawpy.imread(path).postprocess()
    width, height, _ = img.shape
    if width < height:
        return img
    return np.rot90(img)


def get_compressed_imgs(img, width=512, height=768):
    original_width, original_height, _ = img.shape
    sizes = min(original_height//height, original_width//width)
    cut_img = img[:original_width//width*width, :original_height//height*height]
    cut_width, cut_height, _ = cut_img.shape
    for size in range(1,sizes+1):
        if cut_width//size % width == 0 and cut_height//size % height == 0:
            yield cut_img[::size,::size]


def get_split(img, width=512, height=768):
    org_width, org_height, _ = img.shape
    for i in range(org_width//width):
        for j in range(org_height//height):
            yield img[i*width:i*width+width, j*height:j*height+height]


def get_all_splits(img, width=512, height=768):
    for compressed_img in get_compressed_imgs(img):
        for img in get_split(compressed_img):
            yield img
        



def get_all_images(path_to_images, width=512, height=768):
    folders = [str(i) for i in range(5,17)]
    for f in folders:
        for _, _, pics in os.walk(path_to_images + '/' + f):
            for pic in pics:
                npImg = load_nef(path_to_images + '/' + f +'/' + pic)
                for img in get_all_splits(npImg):
                    yield img


