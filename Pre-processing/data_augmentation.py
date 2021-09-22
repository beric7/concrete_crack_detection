# Data Augmentation
#
# Dependency: imgaug
# Author: Xuan Li
# Time: 7/06/2020

import os
from os import listdir
from os.path import splitext
from glob import glob
from tqdm import tqdm
import imageio
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

dir_img = '/Users/xuanli/Desktop/CFD/img/'
dir_mask = '/Users/xuanli/Desktop/CFD/mask/'
dir_out_img = '/Users/xuanli/Desktop/CFD/train_aug/img/'
dir_out_mask = '/Users/xuanli/Desktop/CFD/train_aug/mask/'
augNumPerImage = 30

# Customize this augmentation behavior in this function.
def get_augmenter():
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-0.1, 0.1), translate_percent=(-0.025, 0.025), shear=(-0.025, 0.025), scale=(0.975, 1.025)),
    ], random_order=True)
    return aug
    
def get_item(idx):
    mask_file = glob(dir_mask + idx + '.*')
    img_file = glob(dir_img + idx + '.*')

    assert len(mask_file) == 1, \
        f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
    assert len(img_file) == 1, \
        f'Either no image or multiple images found for the ID {idx}: {img_file}'
    # Open files here
    mask = imageio.imread(mask_file[0])
    img = imageio.imread(img_file[0])
    
    assert img.shape[:2] == mask.shape[:2], \
        f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

    return img, mask

if __name__ == "__main__":
    assert os.path.isdir(dir_img) , 'Input image directory does not exist'
    assert os.path.isdir(dir_mask) , 'Input mask directory does not exist'
    try:
        os.mkdir(dir_out_img)
        logging.info('Created output image directory')
    except OSError:
        pass
    try:
        os.mkdir(dir_out_mask)
        logging.info('Created output mask directory')
    except OSError:
        pass
    
    in_ids = [file for file in listdir(dir_img) if not file.startswith('.')]
    seq = get_augmenter()
    with tqdm(total=len(in_ids), desc=f'Image Processing', unit='img') as pbar:
        for id in in_ids:
            pathsplit = splitext(id)
            img, mask = get_item(pathsplit[0])
            for i in range(augNumPerImage):
                images_aug_i, masks_aug_i = seq(image=img, segmentation_maps=SegmentationMapsOnImage(mask, shape=img.shape))
                imageio.imwrite("{}_aug_{}{}".format(dir_out_img + pathsplit[0], i, pathsplit[1]), images_aug_i)
                imageio.imwrite("{}_aug_{}{}".format(dir_out_mask + pathsplit[0], i, ".png"), masks_aug_i.get_arr())
            imageio.imwrite("{}_aug_{}{}".format(dir_out_img + pathsplit[0], augNumPerImage, pathsplit[1]), img)
            imageio.imwrite("{}_aug_{}{}".format(dir_out_mask + pathsplit[0], augNumPerImage, ".png"), mask)
            pbar.update(1)
    
