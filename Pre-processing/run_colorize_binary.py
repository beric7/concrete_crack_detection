# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:09:01 2021

@author: Admin
"""
from image_utils import colorize_binary

source_binary_folder = './LCW_jpeg_masks_bin/'
destination = './LCW_png_masks_final/'
class_color_dict = {0:0, 255:255}
extension = 'png'

colorize_binary(source_binary_folder, destination, class_color_dict, extension)