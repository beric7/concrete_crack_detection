# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:52:00 2020

@author: Eric Bianchi
"""
import os
import cv2
from tqdm import tqdm

def dilate(source_mask_folder, destination, iterations, kernel, extension):  
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in tqdm(os.listdir(source_mask_folder)):
        
        mask = cv2.imread(source_mask_folder + '/' + filename) 
        dialated_mask = cv2.dilate(mask, kernel, iterations)
        
        head, tail = filename.split('.')
        filename = head + '.' + extension              
               
        cv2.imwrite(destination + '/' + filename, dialated_mask)