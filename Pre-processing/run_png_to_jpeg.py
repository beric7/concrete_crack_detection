# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:13:36 2021

@author: Admin
"""
from image_utils import png_to_jpeg

source_image_folder = './LCW_bin_mask_512x512_global_png/'
source_binary_folder = './np_png_binaries/'
destination = './LCW_jpeg_masks/'
class_color_dict = {0:0, 255:255}
png_to_jpeg(source_image_folder, source_binary_folder, destination, class_color_dict)