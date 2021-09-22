# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:18:09 2021

@author: Admin
"""
from image_utils import*
source_mask_folder = './LCW_masks/'
destination_mask_norm = './LCW_masks_norm/'
destination_binary = './LCW_masks_binary/'

# background_to_white(source_mask_folder, destination_mask_norm)
mask_to_binary_image(destination_mask_norm, destination_binary)