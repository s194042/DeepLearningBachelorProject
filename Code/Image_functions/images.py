import rawpy
import os
import numpy as np
import math

path_to_images = 'C:/Users/Rani/Desktop/ai_training_immages' # foldernames are 1 ... 16

def load_nef(path = 'C:/Users/Rani/Desktop/ai_training_immages' + "/16/_DSC0570.nef"):
    img = rawpy.imread(path).postprocess()
    width, height, _ = img.shape
    if width < height:
        return np.float32(img)
    return np.float32(np.rot90(img))


def get_compressed_imgs(img, width=512, height=768):
    original_width, original_height, _ = img.shape
    sizes = min(original_height//height, original_width//width)
    cut_img = img[:original_width//width*width, :original_height//height*height]
    cut_width, cut_height, _ = cut_img.shape
    for size in range(1,sizes+1):
        if cut_width//size % width == 0 and cut_height//size % height == 0:
            yield np.array(cut_img[::size,::size],dtype=np.uint8)


def get_split(img, width=512, height=768):
    org_width, org_height, _ = img.shape
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
                npImg = load_nef(path_to_images + '/' + f +'/' + pic)
                for img in get_all_splits(npImg):
                    yield img


def get_all_flips_of_image(image):
    yield image
    yield np.fliplr(image)
    image = np.flipud(image)
    yield image
    yield np.fliplr(image)


def get_all_imgs_with_everything(path_to_images, start = 0):
    for imgs in get_all_images(path_to_images, start=start):
        for flips in get_all_flips_of_image(imgs):
            yield flips