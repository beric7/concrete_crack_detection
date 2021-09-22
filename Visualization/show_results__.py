# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 01:01:40 2020

@author: Eric Bianchi
"""

import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm   
import os

def generate_images(model, image_path, name, destination_mask, destination_overlays):
    
    if not os.path.exists(destination_mask): # if it doesn't exist already
        os.makedirs(destination_mask)  
        
    if not os.path.exists(destination_overlays): # if it doesn't exist already
        os.makedirs(destination_overlays)  
 
    image = cv2.imread(image_path)
    # assumes that the image is png...
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('show', image)
    # cv2.waitKey(0)
    width, height = image.shape[1], image.shape[0]
    min_ = min(width, height)
    dim = [min_, min_]
	#process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<image.shape[1] else image.shape[1]
    crop_height = dim[1] if dim[1]<image.shape[0] else image.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = image[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    image = cv2.resize(crop_img, (512,512), interpolation=cv2.INTER_NEAREST)
    img = image.transpose(2,0,1)
    img = img.reshape(1,3,512,512)
    
    with torch.no_grad():
        mask_pred = model(torch.from_numpy(img).type(torch.cuda.FloatTensor))
        
    # color mapping corresponding to classes
    # ---------------------------------------------------------------------
    # 0 = background
    # 1 = crack
    # ---------------------------------------------------------------------
    import numpy as np
    mapping_1 = {0:np.array([0,0,0], dtype=np.uint8), 1:np.array([255,255,255], dtype=np.uint8)}
    
    mapping_2 = {0:np.array([0,0,0], dtype=np.uint8), 1:np.array([0,0,128], dtype=np.uint8)}

    
    y_pred_tensor = mask_pred
    pred = torch.argmax(y_pred_tensor, dim=1)
    y_pred = pred.data.cpu().numpy()
    
    import numpy as np
    height, width, channels = image.shape
    mask_1 = np.zeros((height, width, channels), dtype=np.uint8)
    mask_2 = np.zeros((height, width, channels), dtype=np.uint8)
    
    color = mapping_1[0]   
    
    for k in mapping_1:
        # Get all indices for current class
        idx = (pred==torch.tensor(k, dtype=torch.uint8))
        idx_np = (y_pred==k)[0]
        # color = mapping[k]
        mask_1[idx_np] = (mapping_1[k])
        mask_2[idx_np] = (mapping_2[k])
    
    # cv2.imshow('show', mask)
    # cv2.waitKey(0)
    image = img[0,...].transpose(1,2,0)
    # cv2.imshow('show', image)
    # cv2.waitKey(0)
    # overlay the mask on the image using the alpha combination blending
    overlay = cv2.addWeighted(image, 1, mask_2, 0.65, 0)

    
    # overlays
    cv2.imwrite(destination_mask+'/'+name, mask_1)
    cv2.imwrite(destination_overlays+'/'+name, overlay)
