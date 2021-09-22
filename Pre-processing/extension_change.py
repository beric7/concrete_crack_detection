# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:10:15 2020

@author: Eric Bianchi
"""
import sys

sys.path.insert(0, 'S://Python/general_utils/')

from image_utils import extension_change

source_image_folder = './Test/Masks/'
destination = './Test/Masks/'
extension = 'jpg'
extension_change(source_image_folder, destination, extension)