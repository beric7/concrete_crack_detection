# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:38:49 2020

@author: Eric Bianchi
"""
import sys

sys.path.insert(0, 'E://Python/general_utils/')

from image_utils import rescale

# rescale(source_image_folder, destination, dimension):
dimension = 512
source = './LCW_images/'
destination = '.LCW_images_512x512/'
rescale(source, destination, dimension)
