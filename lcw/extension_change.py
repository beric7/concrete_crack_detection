# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:10:15 2020

@author: Eric Bianchi
"""
import sys

from image_utils import extension_change

source_image_folder = './LCW_bin_mask_512x512_global_jpeg/'
destination = './LCW_bin_mask_512x512_global_jpeg/'
extension = 'jpeg'
extension_change(source_image_folder, destination, extension)