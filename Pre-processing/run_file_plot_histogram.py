# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:09:43 2021

@author: Admin
"""

import argparse
import sys
from histogram_image_ import* 
from build_image_file_list import*

directory = './LCW_masks/'
bins = 10
image_file_paths, image_names = buildImageFileList(directory)
df = img_height_width_csv(image_file_paths)
plotHistogram(df, bins)
