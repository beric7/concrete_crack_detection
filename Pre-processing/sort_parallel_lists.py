# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:24:35 2021

@author: Admin
"""
import os
import shutil
from tqdm import tqdm
path_to_list_dir = 'F://PROJECTS/in-painting/RFR-Inpainting-master/Bridge_LCW/data/LCW_complete/cracked_lcw/'
path_to_target_dir = 'F://PROJECTS/in-painting/RFR-Inpainting-master/Bridge_LCW/data/LCW_complete/images/'
path_to_dest_dir = 'F://PROJECTS/in-painting/RFR-Inpainting-master/Bridge_LCW/data/LCW_complete/cracked_images_lcw/'

if not os.path.exists(path_to_dest_dir): # if it doesn't exist already
    os.makedirs(path_to_dest_dir)  
 
for image_name in tqdm(os.listdir(path_to_list_dir)):
    image_name = image_name.split('.')[0]
    image_src = path_to_target_dir + image_name +'.jpeg'
    image_dest = path_to_dest_dir + image_name +'.jpeg'
    shutil.copyfile(image_src, image_dest)
    
