# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:38:49 2020

@author: Eric Bianchi
"""
import sys

from image_utils import random_sort_images

# blackAndWhite(source_image_folder, destination):
source_mask = './LCW_masks_complete_512/'
source_image = './LCW_images_512x512/'
destination_mask_test = './Test/Masks/'
destination_image_test = './Test/Images/'
destination_mask_train = './Train/Masks'
destination_image_train = './Train/Images/'
percentage = 0.1

# random_sort_images(source_image_folder, destination, seed=10, percentage=0.1)
random_sort_images(source_mask, 
                   source_image, 
                   destination_mask_test, 
                   destination_image_test, 
                   destination_mask_train, 
                   destination_image_train,
                   percentage=percentage)
