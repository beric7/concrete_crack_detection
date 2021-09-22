# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:38:49 2020

@author: Eric Bianchi
"""
import sys
import numpy as np
from dilate_mask import dilate

# rescale(source_image_folder, destination, dimension):
kernel = np.ones((2,2),np.uint8)
iterations = 1

source = './LCW_masks_binary_jpeg/'
destination = './LCW_bin_mask_dialated_jpeg/'
dilate(source, destination, iterations, kernel, 'jpeg')
