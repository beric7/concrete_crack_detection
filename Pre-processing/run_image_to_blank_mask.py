# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:35:29 2021

@author: Admin
"""
from image_utils import*

source_image_folder= './LCW_images/'
destination = './LCW_masks_complete/'
extenstion = 'png'
image_to_blank_bin_mask(source_image_folder, destination, extenstion)