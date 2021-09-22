# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

@Author Eric Bianchi
"""
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import random
from numpy import save
from tqdm import tqdm

def buildImageFileList(BASE, TEST_IMAGES_DIR, sort_ID_string):
    
    imageFilePaths = []
    image_names = []
    
    UNUSABLE_FILETYPE_DIR = "DATA/Extraction/Sorted Data/"
    
    for imageFileName in os.listdir(TEST_IMAGES_DIR):
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        if imageFileName.endswith(".JPG"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        if imageFileName.endswith(".png"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        if imageFileName.endswith(".jpeg"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
            
        # TODO: This does remove the file of interest... may not want to do this in the future. 
        if imageFileName.endswith(".emf"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/EMF/" + imageFileName)
        if imageFileName.endswith(".wmf"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/WMF/" + imageFileName)  
        if imageFileName.endswith(".gif"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/GIF/" + imageFileName)
        if imageFileName.endswith(".tif"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/TIF/" + imageFileName)
        if imageFileName.endswith(".tiff"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/TIFF/" + imageFileName) 
        if imageFileName.endswith(".wdp"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/WDP/" + imageFileName)
            
        
    return imageFilePaths, image_names


# Step 1: adjust paths.
BASE = 'D://'

IMAGE_DIR = BASE + 'DATA/Datasets/ML_project/images/'


def prettyLabel(classDirectory):
    
    imageFilePaths = []
    image_names = []
    
    for class_folder in os.listdir(classDirectory):
        i = 0
        CLASS_PATH = classDirectory + class_folder + '/'
        COPY_CLASS_PATH = classDirectory + 'copy-' + class_folder
        os.mkdir(COPY_CLASS_PATH)
        
        for imageFileName in os.listdir(CLASS_PATH):
            shutil.copyfile(CLASS_PATH + imageFileName, COPY_CLASS_PATH + '/' + class_folder + '_' + str(i) + '.jpg')
            imageFilePaths.append(CLASS_PATH + class_folder + '_' + str(i))
            image_names.append(CLASS_PATH + class_folder + '_' + str(i))
            i = i + 1            
    print('done')
    
    return imageFilePaths , image_names

def verticalFlip(imageDir):
        
    for image in os.listdir(imageDir):
        img = cv2.imread(imageDir + '/' + image)
        imageFlip = cv2.flip(img, 1)
        cv2.imwrite(imageDir + '/' + image + '_vFlip.jpg', imageFlip)           
    print('done')
    
def rescale_ex(source_image_folder, destination, dimension, extension):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in os.listdir(source_image_folder):
        im1 = Image.open(source_image_folder + '/' + filename) 
        
        image = im1.resize((dimension,dimension)) 
  
        head, tail = filename.split('.')
        filename = head + '.' + extension              
               
        image.save(destination + '/' + filename, extension)
        
def rescale(source_image_folder, destination, dimension):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in tqdm(os.listdir(source_image_folder)):
        im1 = cv2.imread(source_image_folder + '/' + filename) 
        
        image = cv2.resize(im1, (dimension,dimension))            
               
        cv2.imwrite(destination + '/' + filename, image)

def rescale_mask(source_image_folder, destination, dimension):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in tqdm(os.listdir(source_image_folder)):
        im1 = cv2.imread(source_image_folder + '/' + filename) 
        
        image = cv2.resize(im1, (dimension,dimension), interpolation=cv2.INTER_NEAREST)            
               
        cv2.imwrite(destination + '/' + filename, image)       

def rescale_binary_mask(source_image_folder, destination, dimension):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in tqdm(os.listdir(source_image_folder)):
        
        im1 = np.load(source_image_folder + '/' + filename)
        im1 = im1.astype(np.uint8)
        image = cv2.resize(im1, (dimension,dimension), interpolation=cv2.INTER_NEAREST)           
        save(destination+'/'+filename, image)                 

def extension_change(source_image_folder, destination, extension):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in os.listdir(source_image_folder):
        image = Image.open(source_image_folder + '/' + filename) 
  
        head, tail = filename.split('.')
        filename = head + '.' + extension    

        image = image.convert("RGB")          
               
        image.save(destination + '/' + filename)
        
def blackAndWhite(source_image_folder, destination):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in os.listdir(source_image_folder):
        im1 = cv2.imread(source_image_folder + '/' + filename)
        
        # makes the assumption that it is a jpg file.
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)          
               
        cv2.imwrite(destination+'/'+filename, im1)

def erodeDialate(source_mask_folder, destination, ksize, iterations, extension):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
    
    kernel = np.ones((ksize,ksize),np.uint8)
        
    for filename in os.listdir(source_mask_folder):
        
        mask = cv2.imread(source_mask_folder + '/' + filename) 
        erosion_mask = cv2.erode(mask, kernel,iterations)
        dialated_mask = cv2.dilate(erosion_mask, kernel, iterations)
        
        head, tail = filename.split('.')
        filename = head + '.' + extension              
               
        cv2.imwrite(destination + '/' + filename, dialated_mask)

def insertName(source_mask_folder, destination, insert_name):  
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
    
    for filename in os.listdir(source_mask_folder):
        
        mask = cv2.imread(source_mask_folder + '/' + filename) 

        cv2.imwrite(destination + '/' + insert_name + filename, mask)
   

def dilate(source_mask_folder, destination, ksize, iterations, extension):  
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    kernel = np.ones((ksize,ksize),np.uint8)
    
    for filename in os.listdir(source_mask_folder):
        
        mask = cv2.imread(source_mask_folder + '/' + filename) 
        dialated_mask = cv2.dilate(mask, kernel, iterations)
        
        head, tail = filename.split('.')
        filename = head + '.' + extension              
               
        cv2.imwrite(destination + '/' + filename, dialated_mask)

def sort_images_into_folder(source_image_folder, destination, count_per, start=0):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    folderID = start
    x = 1
    
    for filename in os.listdir(source_image_folder):
        
        if x % count_per == 0:       
            x = 1     
            folderID+= 1
        if not os.path.exists(destination+'Folder_'+str(folderID)): # if it doesn't exist already
            os.makedirs(destination+'Folder_'+str(folderID))    
        shutil.move(source_image_folder + '/' + filename, destination+'Folder_'+str(folderID)+'/'+filename)
        x+=1

def rename_segmentation(source_folder, source_image_folder, source_mask_folder, source_json_folder, source_ohev,
                        test_or_train, image_destination, mask_destination, json_destination, ohev_destination):
    
    if not os.path.exists(test_or_train + image_destination): # if it doesn't exist already
        os.makedirs(test_or_train + image_destination)
    
    if not os.path.exists(test_or_train + mask_destination): # if it doesn't exist already
        os.makedirs(test_or_train + mask_destination)
        
    if not os.path.exists(test_or_train + json_destination): # if it doesn't exist already
        os.makedirs(test_or_train + json_destination) 
        
    if not os.path.exists(test_or_train + ohev_destination): # if it doesn't exist already
        os.makedirs(test_or_train + ohev_destination) 
        
    x = 0
    for filename in tqdm(os.listdir(source_folder)):
        
        new_image_name = filename
        
        name = filename.split('.')[0]
        
        old_json_name = name + '.json'
        old_mask_name = name + '.png'  
        old_ohev_name = name + '.npy'
        
        
        new_image_name = str(x) + '.jpeg'
        new_json_name = str(x) + '.json'
        new_mask_name = str(x) + '.png'
        new_ohev_name = str(x) + '.npy'
        
        shutil.copy(source_image_folder + '/' + filename, test_or_train + image_destination + new_image_name)     
        shutil.copy(source_mask_folder + '/' + old_mask_name, test_or_train + mask_destination + new_mask_name)   
        shutil.copy(source_json_folder + '/' + old_json_name, test_or_train + json_destination + new_json_name)     
        shutil.copy(source_ohev + '/' + old_ohev_name, test_or_train + ohev_destination + new_ohev_name) 
        x = x + 1


def sharpenImage(source_image_folder, destination):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in os.listdir(source_image_folder):
        im1 = cv2.imread(source_image_folder + '/' + filename)
        
        # makes the assumption that it is a jpg file.
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)     
        
        
               
        cv2.imwrite(destination+'/'+filename, im1)


def random_sort_images(source_mask_folder, source_image_folder, destination_mask_test, 
                       destination_image_test, destination_mask_train, 
                       destination_image_train, percentage=0.1):

    if not os.path.exists(destination_mask_test): # if it doesn't exist already
        os.makedirs(destination_mask_test)
        
    if not os.path.exists(destination_image_test): # if it doesn't exist already
        os.makedirs(destination_image_test)  
        
    if not os.path.exists(destination_mask_train): # if it doesn't exist already
        os.makedirs(destination_mask_train)
        
    if not os.path.exists(destination_image_train): # if it doesn't exist already
        os.makedirs(destination_image_train)  
    
    
    # list all files in dir
    files = [f for f in os.listdir(source_image_folder) if os.path.isfile(source_image_folder + f)]
    
    random.shuffle(files)
    # small_list = files[:int(len(files)*percentage)]
    
    count = 0
    
    for filename in files:
        # send to test folder or train folder
        if count < int(len(files)*percentage):
            print(20 * '#')
            print('Testing')
            print(20 * '#')
            shutil.copy(source_image_folder + '/' + filename, 
                        destination_image_test+ '/' + filename)
            print(filename)
            filename_mask, ext = filename.split('.')
            shutil.copy(source_mask_folder + '/' + filename_mask+'.png', 
                        destination_mask_test+ '/' + filename_mask+'.png')
        else:
            print(20 * '#')
            print('Training')
            print(20 * '#')
            shutil.copy(source_image_folder + '/' + filename, 
                        destination_image_train+ '/' + filename)
            print(filename)
            filename_mask, ext = filename.split('.')
            shutil.copy(source_mask_folder + '/' + filename_mask+'.png', 
                        destination_mask_train+ '/' + filename_mask+'.png')
            
        count = count + 1

def select_class(source_image_folder, destination, remove_classes_array):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
    
    # these files in this directory are binary files...
    for filename in tqdm(os.listdir(source_image_folder)):
        bn_file = np.load(source_image_folder + '/' + filename)
        for selected_class in remove_classes_array:
            bn_file[bn_file == selected_class] = 0

        save(destination+'/'+filename, bn_file)

def colorize_binary(source_image_folder, destination, class_color_dict):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
    
    # these files in this directory are one hot encoded files...
    for filename in tqdm(os.listdir(source_image_folder)):
        bn_file = np.load(source_image_folder + '/' + filename)
        new_bn = np.zeros((bn_file.shape[0], bn_file.shape[1], 3))
        selected_class = 0
        for selected_class in class_color_dict:
            new_bn[bn_file == selected_class] = class_color_dict[selected_class]
            selected_class = selected_class + 1
        
        numpy_img = np.asarray(new_bn, dtype=np.uint8)
        filename_mask, ext = filename.split('.')
        cv2.imwrite(destination+'/'+filename_mask+'.png', numpy_img)


def build_image_file_list(source_directory):
    imageFilePaths = []
    image_names = []
    
    for imageFileName in os.listdir(source_directory):
        imageFilePaths.append(source_directory + imageFileName)
        image_names.append(imageFileName)
        
    return imageFilePaths, image_names