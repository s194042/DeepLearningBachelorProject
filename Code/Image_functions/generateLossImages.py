
import torch
import images
import augmentations
import numpy as np

## loss of 0 is perfect, loss with 1 is really bad

def get_images_with_loss_of_0(img): #10
    yield img
    yield augmentations.Guasian_noice(img, 0.02)
    yield augmentations.Uniform_noice(img, 2)
    yield augmentations.Exp_noice(img, 0.01)
    yield augmentations.Poiss_noice(img, 0.03)
    yield augmentations.uniform_add(img, 5)
    yield augmentations.uniform_add(img, -4)
    yield augmentations.uniform_decimal_multiplication(img, 0.95)
    yield augmentations.uniform_decimal_multiplication(img, 1.05)
    yield augmentations.Unsharp_masking_5x5(img, 1)

def get_images_with_loss_of_0_1(img): #8
    yield augmentations.Guasian_noice(img, 0.035)
    yield augmentations.Uniform_noice(img, 4)
    yield augmentations.Exp_noice(img, 0.02)
    yield augmentations.Poiss_noice(img, 0.08)
    yield augmentations.uniform_add(img, 9)
    yield augmentations.uniform_add(img, -9)
    yield augmentations.uniform_decimal_multiplication(img, 0.91)
    yield augmentations.uniform_decimal_multiplication(img, 1.09)

def get_images_with_loss_of_0_2(img): #11
    yield augmentations.Guasian_noice(img, 0.05)
    yield augmentations.Uniform_noice(img, 6)
    yield augmentations.Exp_noice(img, 0.035)
    yield augmentations.Poiss_noice(img, 0.15)
    yield augmentations.uniform_add(img, 15)
    yield augmentations.uniform_add(img, -15)
    yield augmentations.uniform_decimal_multiplication(img, 0.88)
    yield augmentations.uniform_decimal_multiplication(img, 1.12)
    yield augmentations.sharpen(img, 1)
    yield augmentations.box_blur_3x3(img, 1)
    yield augmentations.Gaussioan_blur_3x3(img, 1)

def get_images_with_loss_of_0_3(img): #8
    yield augmentations.Guasian_noice(img, 0.1)
    yield augmentations.Uniform_noice(img, 12)
    yield augmentations.Exp_noice(img, 0.06)
    yield augmentations.Poiss_noice(img, 0.3)
    yield augmentations.uniform_add(img, 25)
    yield augmentations.uniform_add(img, -25)
    yield augmentations.uniform_decimal_multiplication(img, 0.75)
    yield augmentations.uniform_decimal_multiplication(img, 1.25)

def get_images_with_loss_of_0_4(img): #12
    yield augmentations.Guasian_noice(img, 0.15)
    yield augmentations.Uniform_noice(img, 20)
    yield augmentations.Exp_noice(img, 0.1)
    yield augmentations.Poiss_noice(img, 0.4)
    yield augmentations.uniform_add(img, 50)
    yield augmentations.uniform_add(img, -50)
    yield augmentations.uniform_decimal_multiplication(img, 0.6)
    yield augmentations.uniform_decimal_multiplication(img, 1.4)
    yield augmentations.salt(img, 0.01)
    yield augmentations.peper(img, 0.01)
    yield augmentations.box_blur_5x5(img, 1)
    yield augmentations.Gaussioan_blur_5x5(img, 1)

def get_images_with_loss_of_0_5(img): #10
    yield augmentations.Guasian_noice(img, 0.22)
    yield augmentations.Uniform_noice(img, 30)
    yield augmentations.Exp_noice(img, 0.2)
    yield augmentations.Poiss_noice(img, 0.55)
    yield augmentations.uniform_add(img, 70)
    yield augmentations.uniform_add(img, -70)
    yield augmentations.uniform_decimal_multiplication(img, 0.5)
    yield augmentations.uniform_decimal_multiplication(img, 1.5)
    yield augmentations.salt(img, 0.02)
    yield augmentations.peper(img, 0.02)


def get_images_with_loss_of_0_6(img): #12
    yield augmentations.Guasian_noice(img, 0.3)
    yield augmentations.Uniform_noice(img, 45)
    yield augmentations.Exp_noice(img, 0.3)
    yield augmentations.Poiss_noice(img, 0.7)
    yield augmentations.uniform_add(img, 90)
    yield augmentations.uniform_add(img, -90)
    yield augmentations.uniform_decimal_multiplication(img, 0.4)
    yield augmentations.uniform_decimal_multiplication(img, 1.6)
    yield augmentations.sharpen(img, 2)
    yield augmentations.Unsharp_masking_5x5(img, 4)
    yield augmentations.salt(img, 0.05)
    yield augmentations.peper(img, 0.05)

def get_images_with_loss_of_0_7(img): #10
    yield augmentations.Guasian_noice(img, 0.45)
    yield augmentations.Uniform_noice(img, 70)
    yield augmentations.Exp_noice(img, 0.4)
    yield augmentations.Poiss_noice(img, 0.9)
    yield augmentations.uniform_add(img, 105)
    yield augmentations.uniform_add(img, -105)
    yield augmentations.uniform_decimal_multiplication(img, 0.3)
    yield augmentations.uniform_decimal_multiplication(img, 1.7)
    yield augmentations.salt(img, 0.12)
    yield augmentations.peper(img, 0.15)


def get_images_with_loss_of_0_8(img): #14
    yield augmentations.Guasian_noice(img, 0.6)
    yield augmentations.Uniform_noice(img, 90)
    yield augmentations.Exp_noice(img, 0.5)
    yield augmentations.Poiss_noice(img, 1)
    yield augmentations.uniform_add(img, 120)
    yield augmentations.uniform_add(img, -120)
    yield augmentations.uniform_decimal_multiplication(img, 0.2)
    yield augmentations.uniform_decimal_multiplication(img, 1.8)
    yield augmentations.sharpen(img, 2)
    yield augmentations.Unsharp_masking_5x5(img, 5)
    yield augmentations.box_blur_5x5(img, 20)
    yield augmentations.Gaussioan_blur_5x5(img, 20)
    yield augmentations.salt(img, 0.25)
    yield augmentations.peper(img, 0.3)

def get_images_with_loss_of_0_9(img): #10
    yield augmentations.Guasian_noice(img, 0.9)
    yield augmentations.Uniform_noice(img, 130)
    yield augmentations.Exp_noice(img, 0.8)
    yield augmentations.Poiss_noice(img, 1.15)
    yield augmentations.uniform_add(img, 130)
    yield augmentations.uniform_add(img, -130)
    yield augmentations.uniform_decimal_multiplication(img, 0.15)
    yield augmentations.uniform_decimal_multiplication(img, 2.)
    yield augmentations.salt(img, 0.4)
    yield augmentations.peper(img, 0.5)

def get_images_with_loss_of_1(img): #12
    yield augmentations.Guasian_noice(img, 1.2)
    yield augmentations.Uniform_noice(img, 160)
    yield augmentations.Exp_noice(img, 1)
    yield augmentations.Poiss_noice(img, 1.3)
    yield augmentations.uniform_add(img, 140)
    yield augmentations.uniform_add(img, -140)
    yield augmentations.uniform_decimal_multiplication(img, 0.1)
    yield augmentations.uniform_decimal_multiplication(img, 2.3)
    yield augmentations.sharpen(img, 3)
    yield augmentations.Unsharp_masking_5x5(img, 7)
    yield augmentations.salt(img, 0.5)
    yield augmentations.peper(img, 0.6)


class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)
    def __getitem__(self, index):
        return self.generator_func.__next__()
    def __len__(self):
        return 6645*50*4*117-1 #-1 just to be safe




def get_image_pairs_transforms_with_loss(path='C:/Users/Rani/Desktop/ai_training_immages', start = 0):
    
    imgs = images.get_all_imgs_with_everything(path, start)
    for img in imgs:  # total is 117
        for aug_0 in get_images_with_loss_of_0(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.001, device=torch.device('cuda'))
        for aug_0 in get_images_with_loss_of_1(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.999, device=torch.device('cuda'))
        for aug_0 in get_images_with_loss_of_0_2(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.2, device=torch.device('cuda'))
        for aug_0 in get_images_with_loss_of_0_6(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.6, device=torch.device('cuda'))
        for aug_0 in get_images_with_loss_of_0_4(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.4, device=torch.device('cuda'))
        for aug_0 in get_images_with_loss_of_0_8(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.8, device=torch.device('cuda'))
        
        for aug_0 in get_images_with_loss_of_0_1(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.1, device=torch.device('cuda'))
        for aug_0 in get_images_with_loss_of_0_9(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.9, device=torch.device('cuda'))
        for aug_0 in get_images_with_loss_of_0_7(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.7, device=torch.device('cuda'))
        for aug_0 in get_images_with_loss_of_0_3(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.3, device=torch.device('cuda'))
        for aug_0 in get_images_with_loss_of_0_5(img):
            yield torch.concatenate(((img - 128)/255, img-aug_0), axis=2).permute(2,0,1), torch.tensor(0.5, device=torch.device('cuda'))
        
