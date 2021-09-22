# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:10:37 2020

@author: Eric Bianchi
"""
import os
import cv2

def color_conversion_UNet(source_mask_folder, destination, extension):
    
    # Ensure that destination folder exists or is created
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
    
    # run through the source folder for each mask file
    for filename in os.listdir(source_mask_folder):
        
        # read the file as a cv2 image
        mask = cv2.imread(source_mask_folder + '/' + filename) 
        
        # COLOR to Greyscale
        grey_scale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Greyscale to Binary
        binary = cv2.threshold(grey_scale, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # save the image in the destination folder
        head, tail = filename.split('.')
        filename = head + '.' + extension              
               
        cv2.imwrite(destination + '/' + filename, binary)