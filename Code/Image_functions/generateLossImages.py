import math
import torch
import images
import augmentations
import numpy as np
import torchsummary

## loss of 0 is perfect, loss with 1 is really bad

def get_images_with_loss_of_0(img, cupy=False):
    yield img
    yield augmentations.Guasian_noice(img, 0.02, cupy)
    yield augmentations.Uniform_noice(img, 2, cupy)
    yield augmentations.Exp_noice(img, 0.01, cupy)
    yield augmentations.Poiss_noice(img, 0.03, cupy)
    yield augmentations.uniform_add(img, 5, cupy)
    yield augmentations.uniform_add(img, -4, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 0.95, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 1.05, cupy)
    yield augmentations.Unsharp_masking_5x5(img, 1, cupy)

def get_images_with_loss_of_0_2(img, cupy=False):
    yield augmentations.Guasian_noice(img, 0.05, cupy)
    yield augmentations.Uniform_noice(img, 6, cupy)
    yield augmentations.Exp_noice(img, 0.035, cupy)
    yield augmentations.Poiss_noice(img, 0.15, cupy)
    yield augmentations.uniform_add(img, 15, cupy)
    yield augmentations.uniform_add(img, -15, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 0.88, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 1.12, cupy)
    yield augmentations.sharpen(img, 1, cupy)
    yield augmentations.box_blur_3x3(img, 1, cupy)
    yield augmentations.Gaussioan_blur_3x3(img, 1, cupy)

def get_images_with_loss_of_0_4(img, cupy=False):
    yield augmentations.Guasian_noice(img, 0.15, cupy)
    yield augmentations.Uniform_noice(img, 20, cupy)
    yield augmentations.Exp_noice(img, 0.1, cupy)
    yield augmentations.Poiss_noice(img, 0.4, cupy)
    yield augmentations.uniform_add(img, 50, cupy)
    yield augmentations.uniform_add(img, -50, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 0.6, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 1.4, cupy)
    yield augmentations.salt(img, 0.01, cupy)
    yield augmentations.peper(img, 0.01, cupy)
    yield augmentations.box_blur_5x5(img, 1, cupy)
    yield augmentations.Gaussioan_blur_5x5(img, 1, cupy)


def get_images_with_loss_of_0_6(img, cupy=False):
    yield augmentations.Guasian_noice(img, 0.3, cupy)
    yield augmentations.Uniform_noice(img, 45, cupy)
    yield augmentations.Exp_noice(img, 0.3, cupy)
    yield augmentations.Poiss_noice(img, 0.7, cupy)
    yield augmentations.uniform_add(img, 90, cupy)
    yield augmentations.uniform_add(img, -90, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 0.4, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 1.6, cupy)
    yield augmentations.sharpen(img, 2, cupy)
    yield augmentations.Unsharp_masking_5x5(img, 4, cupy)
    yield augmentations.salt(img, 0.05, cupy)
    yield augmentations.peper(img, 0.05, cupy)


def get_images_with_loss_of_0_8(img, cupy=False):
    yield augmentations.Guasian_noice(img, 0.6, cupy)
    yield augmentations.Uniform_noice(img, 90, cupy)
    yield augmentations.Exp_noice(img, 0.5, cupy)
    yield augmentations.Poiss_noice(img, 1, cupy)
    yield augmentations.uniform_add(img, 120, cupy)
    yield augmentations.uniform_add(img, -120, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 0.2, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 1.8, cupy)
    yield augmentations.sharpen(img, 2, cupy)
    yield augmentations.Unsharp_masking_5x5(img, 5, cupy)
    yield augmentations.box_blur_5x5(img, 20, cupy)
    yield augmentations.Gaussioan_blur_5x5(img, 20, cupy)
    yield augmentations.salt(img, 0.25, cupy)
    yield augmentations.peper(img, 0.3, cupy)


def get_images_with_loss_of_1(img, cupy=False):
    yield augmentations.Guasian_noice(img, 1.2, cupy)
    yield augmentations.Uniform_noice(img, 160, cupy)
    yield augmentations.Exp_noice(img, 1, cupy)
    yield augmentations.Poiss_noice(img, 1.3, cupy)
    yield augmentations.uniform_add(img, 140, cupy)
    yield augmentations.uniform_add(img, -140, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 0.1, cupy)
    yield augmentations.uniform_decimal_multiplication(img, 2.3, cupy)
    yield augmentations.sharpen(img, 3, cupy)
    yield augmentations.Unsharp_masking_5x5(img, 7, cupy)
    yield augmentations.salt(img, 0.5, cupy)
    yield augmentations.peper(img, 0.6, cupy)


class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)
    def __getitem__(self, index):
        return self.generator_func.__next__()
    def __len__(self):
        return 101984400

def get_image_pairs_transforms_with_loss(path='C:/Users/Rani/Desktop/ai_training_immages', cupy=False):
    imgs = images.get_all_imgs_with_everything(path)
    for img in imgs:


        
        for aug_0 in get_images_with_loss_of_0(img, cupy):
            yield torch.tensor(np.concatenate(((img - 128)/255, img-aug_0), axis=2), dtype=torch.float).permute(2,0,1).to("cuda"), torch.tensor(0.0).to("cuda")
        for aug_0 in get_images_with_loss_of_1(img, cupy):
            yield torch.tensor(np.concatenate(((img - 128)/255, img-aug_0), axis=2), dtype=torch.float).permute(2,0,1).to("cuda"), torch.tensor(0.999).to("cuda")
        for aug_0 in get_images_with_loss_of_0_2(img, cupy):
            yield torch.tensor(np.concatenate(((img - 128)/255, img-aug_0), axis=2), dtype=torch.float).permute(2,0,1).to("cuda"), torch.tensor(0.2000).to("cuda")
        for aug_0 in get_images_with_loss_of_0_6(img, cupy):
            yield torch.tensor(np.concatenate(((img - 128)/255, img-aug_0), axis=2), dtype=torch.float).permute(2,0,1).to("cuda"), torch.tensor(0.6000).to("cuda")
        for aug_0 in get_images_with_loss_of_0_4(img, cupy):
            yield torch.tensor(np.concatenate(((img - 128)/255, img-aug_0), axis=2), dtype=torch.float).permute(2,0,1).to("cuda"), torch.tensor(0.4000).to("cuda")
        for aug_0 in get_images_with_loss_of_0_8(img, cupy):
            yield torch.tensor(np.concatenate(((img - 128)/255, img-aug_0), axis=2), dtype=torch.float).permute(2,0,1).to("cuda"), torch.tensor(0.8000).to("cuda")
        
