
import torch
import images
import augmentations
import numpy as np
torch.set_grad_enabled(False)
## loss of 0 is perfect, loss with 1 is really bad

def get_images_with_loss_of_0(img, rand, randn): #10
    yield img
    yield augmentations.Guasian_noice(img, randn, 0.02)
    yield augmentations.Uniform_noice(img, rand, 2)
    yield augmentations.Exp_noice(img, randn, 0.01)
    yield augmentations.Poiss_noice(img, rand, 0.03)
    yield augmentations.uniform_add(img, 5)
    yield augmentations.uniform_add(img, -4)
    yield augmentations.uniform_decimal_multiplication(img, 0.95)
    yield augmentations.uniform_decimal_multiplication(img, 1.05)
    yield augmentations.Unsharp_masking_5x5(img, 1)

def get_images_with_loss_of_0_1(img, rand, randn): #8
    yield augmentations.Guasian_noice(img, randn, 0.035)
    yield augmentations.Uniform_noice(img, rand, 4)
    yield augmentations.Exp_noice(img, randn, 0.015)
    yield augmentations.Poiss_noice(img, rand, 0.09)
    yield augmentations.uniform_add(img, 9)
    yield augmentations.uniform_add(img, -9)
    yield augmentations.uniform_decimal_multiplication(img, 0.91)
    yield augmentations.uniform_decimal_multiplication(img, 1.14)

def get_images_with_loss_of_0_2(img, rand, randn): #11
    yield augmentations.Guasian_noice(img, randn, 0.05)
    yield augmentations.Uniform_noice(img, rand, 6)
    yield augmentations.Exp_noice(img, randn, 0.02)
    yield augmentations.Poiss_noice(img, rand, 0.18)
    yield augmentations.uniform_add(img, 15)
    yield augmentations.uniform_add(img, -15)
    yield augmentations.uniform_decimal_multiplication(img, 0.88)
    yield augmentations.uniform_decimal_multiplication(img, 1.22)
    yield augmentations.sharpen(img, 1)
    yield augmentations.box_blur_3x3(img, 1)
    yield augmentations.Gaussioan_blur_3x3(img, 1)

def get_images_with_loss_of_0_3(img, rand, randn): #8
    yield augmentations.Guasian_noice(img, randn, 0.1)
    yield augmentations.Uniform_noice(img, rand, 12)
    yield augmentations.Exp_noice(img, randn, 0.035)
    yield augmentations.Poiss_noice(img, rand, 0.35)
    yield augmentations.uniform_add(img, 25)
    yield augmentations.uniform_add(img, -25)
    yield augmentations.uniform_decimal_multiplication(img, 0.75)
    yield augmentations.uniform_decimal_multiplication(img, 1.40)

def get_images_with_loss_of_0_4(img, rand, randn): #12
    yield augmentations.Guasian_noice(img, randn, 0.15)
    yield augmentations.Uniform_noice(img, rand, 20)
    yield augmentations.Exp_noice(img, randn, 0.07)
    yield augmentations.Poiss_noice(img, rand, 0.48)
    yield augmentations.uniform_add(img, 50)
    yield augmentations.uniform_add(img, -50)
    yield augmentations.uniform_decimal_multiplication(img, 0.6)
    yield augmentations.uniform_decimal_multiplication(img, 1.6)
    yield augmentations.salt(img, rand, 0.01)
    yield augmentations.peper(img, rand, 0.01)
    yield augmentations.box_blur_5x5(img, 1)
    yield augmentations.Gaussioan_blur_5x5(img, 1)

def get_images_with_loss_of_0_5(img, rand, randn): #10
    yield augmentations.Guasian_noice(img, randn, 0.22)
    yield augmentations.Uniform_noice(img, rand, 30)
    yield augmentations.Exp_noice(img, randn, 0.14)
    yield augmentations.Poiss_noice(img, rand, 0.65)
    yield augmentations.uniform_add(img, 70)
    yield augmentations.uniform_add(img, -70)
    yield augmentations.uniform_decimal_multiplication(img, 0.5)
    yield augmentations.uniform_decimal_multiplication(img, 1.9)
    yield augmentations.salt(img, rand, 0.02)
    yield augmentations.peper(img, rand, 0.02)

def get_images_with_loss_of_0_6(img, rand, randn): #12
    yield augmentations.Guasian_noice(img, randn, 0.3)
    yield augmentations.Uniform_noice(img, rand, 45)
    yield augmentations.Exp_noice(img, randn, 0.2)
    yield augmentations.Poiss_noice(img, rand, 0.8)
    yield augmentations.uniform_add(img, 90)
    yield augmentations.uniform_add(img, -90)
    yield augmentations.uniform_decimal_multiplication(img, 0.4)
    yield augmentations.uniform_decimal_multiplication(img, 2.3)
    #yield augmentations.sharpen(img, 2) mooved down
    #yield augmentations.Unsharp_masking_5x5(img, 4)
    yield augmentations.salt(img, rand, 0.05)
    yield augmentations.peper(img, rand, 0.05)
    yield augmentations.sharpen(img, 2)

def get_images_with_loss_of_0_7(img, rand, randn): #10
    yield augmentations.Guasian_noice(img, randn, 0.45)
    yield augmentations.Uniform_noice(img, rand, 70)
    yield augmentations.Exp_noice(img, randn, 0.3)
    yield augmentations.Poiss_noice(img, rand, 1.1)
    yield augmentations.uniform_add(img, 105)
    yield augmentations.uniform_add(img, -105)
    yield augmentations.uniform_decimal_multiplication(img, 0.3)
    yield augmentations.uniform_decimal_multiplication(img, 2.6)
    yield augmentations.salt(img, rand, 0.12)
    yield augmentations.peper(img, rand, 0.15)

def get_images_with_loss_of_0_8(img, rand, randn): #14
    yield augmentations.Guasian_noice(img, randn, 0.6)
    yield augmentations.Uniform_noice(img, rand, 90)
    yield augmentations.Exp_noice(img, randn, 0.4)
    yield augmentations.Poiss_noice(img, rand, 1.4)
    yield augmentations.uniform_add(img, 120)
    yield augmentations.uniform_add(img, -120)
    yield augmentations.uniform_decimal_multiplication(img, 0.2)
    yield augmentations.uniform_decimal_multiplication(img, 3.2)
    # yield augmentations.sharpen(img, 3) mooved down
    #yield augmentations.Unsharp_masking_5x5(img, 5)
    #yield augmentations.box_blur_5x5(img, 20)
    #yield augmentations.Gaussioan_blur_5x5(img, 20)
    yield augmentations.salt(img, rand, 0.25)
    yield augmentations.peper(img, rand, 0.3)
    yield augmentations.sharpen(img, 3)

def get_images_with_loss_of_0_9(img, rand, randn): #10
    yield augmentations.Guasian_noice(img, randn, 0.9)
    yield augmentations.Uniform_noice(img, rand, 130)
    yield augmentations.Exp_noice(img, randn, 0.6)
    yield augmentations.Poiss_noice(img, rand, 1.6)
    yield augmentations.uniform_add(img, 130)
    yield augmentations.uniform_add(img, -130)
    yield augmentations.uniform_decimal_multiplication(img, 0.15)
    yield augmentations.uniform_decimal_multiplication(img, 4)
    yield augmentations.salt(img, rand, 0.4)
    yield augmentations.peper(img, rand, 0.5)

def get_images_with_loss_of_1(img, rand, randn): #11
    yield augmentations.Guasian_noice(img, randn, 1.2)
    yield augmentations.Uniform_noice(img, rand, 160)
    yield augmentations.Exp_noice(img, randn, 0.8)
    yield augmentations.Poiss_noice(img, rand, 2)
    yield augmentations.uniform_add(img, 140)
    yield augmentations.uniform_add(img, -140)
    yield augmentations.uniform_decimal_multiplication(img, 0.1)
    yield augmentations.uniform_decimal_multiplication(img, 4.)
    # yield augmentations.sharpen(img, 4) mooved down
    #yield augmentations.Unsharp_masking_5x5(img, 7)
    yield augmentations.salt(img, rand, 0.5)
    yield augmentations.peper(img, rand, 0.6)
    yield augmentations.sharpen(img, 5)


def get_image_pairs_transforms_with_loss(path: str='C:/Users/Rani/Desktop/ai_training_immages/_0', start: int = 0, end: int=6.645*50*4*106-1):
    si = 512, 768
    extra = torch.zeros(512, 768, 2, device=torch.device('cuda'))
    imgs = images.get_all_images_without_anything(path)
    for img in imgs:
        cat = torch.concatenate((torch.divide(torch.subtract(img, 128), 128), extra), axis=2)
        rand = torch.rand(img.size(), device=torch.device('cuda'))
        randn = torch.randn(img.size(), device=torch.device('cuda'))
        for i in range(1):
            for aug_0 in get_images_with_loss_of_0(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.0, device=torch.device('cuda'))
            for aug_0 in get_images_with_loss_of_1(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(1.0, device=torch.device('cuda'))
            for aug_0 in get_images_with_loss_of_0_2(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.2, device=torch.device('cuda'))
            for aug_0 in get_images_with_loss_of_0_6(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.6, device=torch.device('cuda'))
            for aug_0 in get_images_with_loss_of_0_4(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.4, device=torch.device('cuda'))
            for aug_0 in get_images_with_loss_of_0_8(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.8, device=torch.device('cuda'))
            
            for aug_0 in get_images_with_loss_of_0_1(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.1, device=torch.device('cuda'))
            for aug_0 in get_images_with_loss_of_0_9(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.9, device=torch.device('cuda'))
            for aug_0 in get_images_with_loss_of_0_7(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.7, device=torch.device('cuda'))
            for aug_0 in get_images_with_loss_of_0_3(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.3, device=torch.device('cuda'))
            for aug_0 in get_images_with_loss_of_0_5(img, rand, randn):
                yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.5, device=torch.device('cuda'))








def get_images_with_loss_of_0_0_1(img, rand, randn, idx): #10
    id = idx%11
    if id == 0:
        return img
    elif id == 1:
        return augmentations.Guasian_noice(img, randn, 0.02)
    elif id == 2:
        return augmentations.Uniform_noice(img, rand, 2)
    elif id == 3:
        return augmentations.Exp_noice(img, randn, 0.01)
    elif id == 4:
        return augmentations.Poiss_noice(img, rand, 0.03)
    elif id == 5:
        return augmentations.uniform_add(img, 5)
    elif id == 6:
        return augmentations.uniform_add(img, -4)
    elif id == 7:
        return augmentations.uniform_decimal_multiplication(img, 0.95)
    elif id == 8:
        return augmentations.uniform_decimal_multiplication(img, 1.05)
    else:
        return augmentations.Unsharp_masking_5x5(img, 1)
    
    
def get_images_with_loss_of_0_1_1(img, rand, randn, idx): #8
    id = idx%8
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 0.035)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 4)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.015)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 0.09)
    elif id == 4:
        return augmentations.uniform_add(img, 9)
    elif id == 5:
        return augmentations.uniform_add(img, -9)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 0.91)
    else:
        return augmentations.uniform_decimal_multiplication(img, 1.14)
    

def get_images_with_loss_of_0_2_1(img, rand, randn, idx): #11
    id = idx%14
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 0.05)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 6)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.02)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 0.18)
    elif id == 4:
        return augmentations.uniform_add(img, 15)
    elif id == 5:
        return augmentations.uniform_add(img, -15)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 0.88)
    elif id == 7:
        return augmentations.uniform_decimal_multiplication(img, 1.22)
    elif id in {8,9}:
        return augmentations.sharpen(img, 1)
    elif id in {10,11}:
        return augmentations.box_blur_3x3(img, 1)
    else:
        return augmentations.Gaussioan_blur_3x3(img, 1)


def get_images_with_loss_of_0_3_1(img, rand, randn, idx): #8
    id = idx%8
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 0.1)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 12)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.035)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 0.35)
    elif id == 4:
        return augmentations.uniform_add(img, 25)
    elif id == 5:
        return augmentations.uniform_add(img, -25)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 0.75)
    else:
        return augmentations.uniform_decimal_multiplication(img, 1.40)    
   
    
def get_images_with_loss_of_0_4_1(img, rand, randn, idx): #12
    id = idx%14
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 0.15)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 20)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.07)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 0.48)
    elif id == 4:
        return augmentations.uniform_add(img, 50)
    elif id == 5:
        return augmentations.uniform_add(img, -50)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 0.6)
    elif id == 7:
        return augmentations.uniform_decimal_multiplication(img, 1.6) 
    elif id == 8:
        return augmentations.salt(img, rand, 0.01)
    elif id == 9:
        return augmentations.peper(img, rand, 0.01)
    elif id in {10,11}:
        return augmentations.box_blur_5x5(img, 1)
    else:
        return augmentations.Gaussioan_blur_5x5(img, 1)

def get_images_with_loss_of_0_5_1(img, rand, randn, idx): #10
    id = idx%10
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 0.22)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 30)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.14)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 0.65)
    elif id == 4:
        return augmentations.uniform_add(img, 70)
    elif id == 5:
        return augmentations.uniform_add(img, -70)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 0.5)
    elif id == 7:
        return augmentations.uniform_decimal_multiplication(img, 1.9) 
    elif id == 8:
        return augmentations.salt(img, rand, 0.02)
    else:
        return augmentations.peper(img, rand, 0.02)

def get_images_with_loss_of_0_6_1(img, rand, randn, idx): #12
    id = idx%12
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 0.3)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 45)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.2)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 0.8)
    elif id == 4:
        return augmentations.uniform_add(img, 90)
    elif id == 5:
        return augmentations.uniform_add(img, -90)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 0.4)
    elif id == 7:
        return augmentations.uniform_decimal_multiplication(img, 2.3)
    elif id == 8:
        return augmentations.salt(img, rand, 0.05)
    elif id == 9:
        return augmentations.peper(img, rand, 0.05)
    else:
        return augmentations.sharpen(img, 2)


def get_images_with_loss_of_0_7_1(img, rand, randn, idx): #10
    id = idx%10
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 0.45)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 70)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.3)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 1.1)
    elif id == 4:
        return augmentations.uniform_add(img, 105)
    elif id == 5:
        return augmentations.uniform_add(img, -105)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 1.7)
    elif id == 7:
        return augmentations.uniform_decimal_multiplication(img, 2.6)
    elif id == 8:
        return augmentations.salt(img, rand, 0.12)
    else:
        return augmentations.peper(img, rand, 0.15)

    
def get_images_with_loss_of_0_8_1(img, rand, randn, idx): #14
    id = idx%12
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 0.6)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 90)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.4)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 1.4)
    elif id == 4:
        return augmentations.uniform_add(img, 120)
    elif id == 5:
        return augmentations.uniform_add(img, -120)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 0.2)
    elif id == 7:
        return augmentations.uniform_decimal_multiplication(img, 3.2)
    elif id == 8:
        return augmentations.salt(img, rand, 0.25)
    elif id == 9:
        return augmentations.peper(img, rand, 0.3)
    else:
        return augmentations.sharpen(img, 3)

    
    

def get_images_with_loss_of_0_9_1(img, rand, randn, idx): #10
    id = idx%10
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 0.9)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 130)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.6)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 1.6)
    elif id == 4:
        return augmentations.uniform_add(img, 130)
    elif id == 5:
        return augmentations.uniform_add(img, -130)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 0.15)
    elif id == 7:
        return augmentations.uniform_decimal_multiplication(img, 4)
    elif id == 8:
        return augmentations.salt(img, rand, 0.4)
    else:
        return augmentations.peper(img, rand, 0.5)
    

def get_images_with_loss_of_1_1(img, rand, randn, idx): #11
    id = idx%11
    if id == 0:
        return augmentations.Guasian_noice(img, randn, 1.2)
    elif id == 1:
        return augmentations.Uniform_noice(img, rand, 160)
    elif id == 2:
        return augmentations.Exp_noice(img, randn, 0.8)
    elif id == 3:
        return augmentations.Poiss_noice(img, rand, 2)
    elif id == 4:
        return augmentations.uniform_add(img, 140)
    elif id == 5:
        return augmentations.uniform_add(img, -140)
    elif id == 6:
        return augmentations.uniform_decimal_multiplication(img, 0.1)
    elif id == 7:
        return augmentations.uniform_decimal_multiplication(img, 5)
    elif id == 8:
        return augmentations.salt(img, rand, 0.5)
    elif id == 9:
        return augmentations.peper(img, rand, 0.6)
    else:
        return augmentations.sharpen(img, 4)
    
    






def get_image_pairs_transforms_with_loss2(path: str='C:/Users/Rani/Desktop/ai_training_immages/_0', epoch= 0, start: int = 0, end: int=6.645*50*4*106-1):
    si = 512, 768
    extra = torch.zeros(512, 768, 2, device=torch.device('cuda'))
    imgs = images.get_all_imgs_with_everything(path, start, end)
    
    for idx, img in enumerate(imgs, start=epoch):
        cat = torch.concatenate((torch.divide(torch.subtract(img, 128), 128), extra), axis=2)
        rand = torch.rand(img.size(), device=torch.device('cuda'))
        randn = torch.randn(img.size(), device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_0_0_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.0, device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_1_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(1.0, device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_0_2_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.2, device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_0_6_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.6, device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_0_4_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.4, device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_0_8_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.8, device=torch.device('cuda'))
        
        aug_0 = get_images_with_loss_of_0_1_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.1, device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_0_9_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.9, device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_0_7_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.7, device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_0_3_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.3, device=torch.device('cuda'))
        aug_0 = get_images_with_loss_of_0_5_1(img, rand, randn, idx)
        yield torch.concatenate((cat, torch.subtract(img, aug_0)), axis=2).permute(2,0,1), torch.tensor(0.5, device=torch.device('cuda'))




class MakeIter(torch.utils.data.IterableDataset):
    #@torch.jit.script
    def __init__(self, path = 'C:/Users/Rani/Desktop/ai_training_immages/', start_index: int = 0, folder: str ="_0", epoch:int = 0, startup=False, **kwargs):
        super(MakeIter).__init__()
        self.start_index = start_index
        self.kwargs = kwargs
        self.folder = folder
        self.epoch = epoch
        self.startup = startup
        self.path = path

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info == None or worker_info.num_workers == 1 or worker_info.num_workers == 0:
            folder = self.folder
        elif worker_info.num_workers == 2:
            folder = self.folder + str(worker_info.id + 1)
        else:
            folder = self.folder + str(worker_info.id + 1)
        if self.startup:
            return iter(get_image_pairs_transforms_with_loss(path = self.path + folder))
        else:
            return iter(get_image_pairs_transforms_with_loss2(path = self.path + folder, epoch=self.epoch))
    def __getitem__(self, index:int):
        return self.generator_func.__next__()
    
    def __len__(self):
        return 6000*50*4*11# not sure if there are 6000 because of bug in raw loading #645*50*4*106-1 #-1 just to be safe