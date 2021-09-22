# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:16:26 2021

@author: Eric Bianchi
"""
import cv2
import numpy as np
from tqdm import tqdm
import os


class concat:
    def __init__(self, source_image_dir, destination_dir, image_list_dir, image_name, height, width):
        self.source_image_dir = source_image_dir
        self.destination_dir = destination_dir
        self.image_list_dir = image_list_dir
        self.image_name = image_name
        self.height = height
        self.width = width

    def concat_images(self, pad=False, recurrent=False):
        # make a list of the images
        image_list = make_image_list(self.image_list_dir)
        # make 2d array
        array_2d = make_matrix_from_list_jpeg(self.source_image_dir, image_list)
        
        if pad:
            i = 0
            for item in array_2d:
                array_2d[i] = np.pad(array_2d[i], ((4,4), (4,4), (0,0)), mode='constant', constant_values=255)
                i +=1
        # configure array to output form
        list_array = make_list_array(array_2d, self.height, self.width)
        # concatinate image
        img_tile = concat_vh(list_array)
        # save concatinated image
        save_concat_image(destination_dir, image_name, img_tile)
    
    
# @param: source image_dir = what folder are we using?
# @param: h = number of images concat for height of resultant image
# @param: w = number of images concat for width of resultant image
# @return: img_array = resultant concatenated image
def make_matrix(source_image_dir):
    image_array = []    
    for image_name in tqdm(os.listdir(source_image_dir)):
        
        image_path = source_image_dir + image_name
        img = cv2.imread(image_path)
        image_array.append(img)
        
        return image_array

def make_image_list(image_dir):
    image_list_array = []
    for image_name in os.listdir(image_dir):
        image_list_array.append(image_name)
    return image_list_array

##########################################################################
# HELPER FUNCTIONS
    
# define a function for vertically  
# concatenating images of the  
# same size  and horizontally 
def concat_vh(list_2d):     
      # return final image 
    return cv2.vconcat([cv2.hconcat(list_h)  
                        for list_h in list_2d]) 

def make_matrix_from_list_jpeg(image_dir, source_image_list):
    image_array = [] 
    
    for image_name in tqdm(source_image_list):
        image_name = image_name.split('.')[-2]+'.jpeg'
        image_path = image_dir + image_name
        img = cv2.imread(image_path)
        image_array.append(img)
    
    return image_array

def make_matrix_from_list(image_dir, source_image_list):
    image_array = [] 
    
    for image_name in tqdm(source_image_list):
        image_path = image_dir + image_name
        img = cv2.imread(image_path)
        image_array.append(img)
    
    return image_array

def save_images_from_im_list(image_list, source_dir, target_dir):
    
    if not os.path.exists(target_dir): # if it doesn't exist already
        os.makedirs(target_dir)
    
    for image_name in tqdm(image_list):
        image_name = image_name.split('.')[-2]+'.jpeg'
        image_path = source_dir + image_name
        img = cv2.imread(image_path)
        cv2.imwrite(destination_dir+image_name, img)
        
def save_concat_image(destination_dir, image_name, image):
    if not os.path.exists(destination_dir): # if it doesn't exist already
        os.makedirs(destination_dir)
    cv2.imwrite('./'+destination_dir+image_name+'_concat.jpeg', image)
    

def make_list_array(array_2d, dim1, dim2):
    array = []
    for i in range(0, dim1):
        temp = []
        for j in range(0, dim2):
            temp.append(array_2d[dim2*i+j])
        array.append(temp)
    return array

def concat_lists(image_list, directory_list, destination_dir, pad=False, orientation=1):
    if not os.path.exists(destination_dir): # if it doesn't exist already
        os.makedirs(destination_dir)
    for image_name in image_list:
        i = 0
        image_name = image_name.split('.')[-2]+'.jpeg'
        img = cv2.imread(directory_list[i]+image_name)
        if img is None:
            image_name = image_name.split('.')[-2]+'.jpg'
            img = cv2.imread(directory_list[i]+image_name)
        mask_name = image_name.split('.')[-2]+'.png'
        img = cv2.imread(directory_list[i]+image_name)
        print(image_name)
        if pad:    
            img = np.pad(img,((4,4), (4,4), (0,0)), mode='constant', constant_values=255)
        i = 1
        for i in range(1, len(directory_list)):
            print(directory_list[i]+mask_name)
            
            temp = cv2.imread(directory_list[i]+image_name)
            if temp is None:
                temp = cv2.imread(directory_list[i]+mask_name)
            if pad:    
                temp = np.pad(temp, ((4,4), (4,4), (0,0)), mode='constant', constant_values=255)
            if orientation == 1:
                img = cv2.hconcat([img, temp])
            else: 
                img = cv2.vconcat([img, temp])
            i+=1
        cv2.imwrite(destination_dir+image_name, img)
        
def rescale(source_image_folder, destination, dimension):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in tqdm(os.listdir(source_image_folder)):
        im1 = cv2.imread(source_image_folder + '/' + filename) 
        
        image = cv2.resize(im1, (dimension,dimension))            
               
        cv2.imwrite(destination + '/' + filename, image)

#rescale('./bearings/', './bearings/', 512)'''

image_name = 'f4'
source_image_dir = './sample_masks/f4/'
destination_dir = './concatinated_samples/f4/'
image_list_dir = source_image_dir

dim1 = 1
dim2 = 4

image_list = make_image_list(source_image_dir)
directory_list = ['./Test/Images_512x512/','./Test/Masks_512x512/', './predicted_masks/var_1/', './combined_overlays/var_1/']
concat_lists(image_list, directory_list, './multi_concat/', pad=True, orientation=1)

dim1 = 1
dim2 = 5

concat_obj = concat('./multi_concat/', destination_dir, image_list_dir, image_name, dim1, dim2)
concat_obj.concat_images(True)

