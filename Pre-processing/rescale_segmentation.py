# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:38:49 2020

@author: Eric Bianchi
"""
import sys

from image_utils import rescale_mask

# rescale(source_image_folder, destination, dimension):
dimension = 512
source = './Test/Masks/'
destination = './Test/Masks_512x512/'
rescale_mask(source, destination, dimension)
