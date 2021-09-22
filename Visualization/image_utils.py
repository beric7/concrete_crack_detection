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
        
def extension_change(source_image_folder, destination, extension):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in tqdm(os.listdir(source_image_folder)):
        image = Image.open(source_image_folder + '/' + filename) 
  
        head, tail = filename.split('.')
        filename = head + '.' + extension    

        image = image.convert("RGB")          
               
        image.save(destination + '/' + filename, extension)
        
def greyscale(source_image_folder, destination):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in os.listdir(source_image_folder):
        im1 = cv2.imread(source_image_folder + '/' + filename)
        
        # makes the assumption that it is a jpg file.
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)          
               
        cv2.imwrite(destination+'/'+filename, im1)

# def dialateErode(source_image_folder, destination, ksize, iterations):
#     kernel = np.ones((ksize,ksize),np.uint8)
#     dialated_mask = cv2.dilate(erosion,kernel,iterations)
#     erosion = cv2.erode(dialated_mask,kernel,iterations)

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

def random_sort_images_(source_mask_folder, source_cured_mask_folder, source_image_folder, destination_mask_test, 
                        destination_cured_mask_test, destination_image_test, destination_mask_train, 
                        destination_cured_mask_train, destination_image_train, percentage=0.1):

    if not os.path.exists(destination_mask_test): # if it doesn't exist already
        os.makedirs(destination_mask_test)
    
    if not os.path.exists(destination_cured_mask_test): # if it doesn't exist already
        os.makedirs(destination_cured_mask_test)
        
    if not os.path.exists(destination_image_test): # if it doesn't exist already
        os.makedirs(destination_image_test)  
        
    if not os.path.exists(destination_mask_train): # if it doesn't exist already
        os.makedirs(destination_mask_train)
        
    if not os.path.exists(destination_cured_mask_train): # if it doesn't exist already
        os.makedirs(destination_cured_mask_train)    
        
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
            shutil.copy(source_mask_folder + '/' + filename, 
                        destination_mask_test+ '/' + filename)
            shutil.copy(source_cured_mask_folder + '/' + filename, 
                        destination_cured_mask_test+ '/' + filename)
        else:
            print(20 * '#')
            print('Training')
            print(20 * '#')
            shutil.copy(source_image_folder + '/' + filename, 
                        destination_image_train+ '/' + filename)
            print(filename)
            shutil.copy(source_mask_folder + '/' + filename, 
                        destination_mask_train+ '/' + filename)
            shutil.copy(source_cured_mask_folder + '/' + filename, 
                        destination_cured_mask_train+ '/' + filename)           
        count = count + 1

def random_sort_images__(source_mask_folder, source_image_folder, destination_mask_test, 
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
            shutil.copy(source_mask_folder + '/' + filename, 
                        destination_mask_test+ '/' + filename)
        else:
            print(20 * '#')
            print('Training')
            print(20 * '#')
            shutil.copy(source_image_folder + '/' + filename, 
                        destination_image_train+ '/' + filename)
            print(filename)
            shutil.copy(source_mask_folder + '/' + filename, 
                        destination_mask_train+ '/' + filename)
            
        count = count + 1

def background_to_white(source_mask_folder, destination):
   
   if not os.path.exists(destination): # if it doesn't exist already
       os.makedirs(destination)
   
   # these files in this directory are one hot encoded files...
   for filename in tqdm(os.listdir(source_mask_folder)):
       mask = cv2.imread(source_mask_folder + '/' + filename)
       mask[np.all(mask == [0,0,0], axis=-1)] = [255,255,255]
       # numpy_mask = np.asarray(mask, dtype=np.uint8)
       cv2.imwrite(destination+'/'+filename, mask)   
       
       
def background_to_black(source_mask_folder, destination):
   
   if not os.path.exists(destination): # if it doesn't exist already
       os.makedirs(destination)
   
   # these files in this directory are one hot encoded files...
   for filename in tqdm(os.listdir(source_mask_folder)):
       mask = cv2.imread(source_mask_folder + '/' + filename)
       average = np.average(mask)
       if average > 100:
           mask[np.all(mask == [255,255,255], axis=-1)] = [0,0,0]
           # numpy_mask = np.asarray(mask, dtype=np.uint8)
       cv2.imwrite(destination+'/'+filename, mask)   

def colorize_binary(source_binary_folder, destination, class_color_dict, extension):
    
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
    
    # these files in this directory are one hot encoded files...
    for filename in tqdm(os.listdir(source_binary_folder)):
        bn_file = np.load(source_binary_folder + '/' + filename)
        new_bn = np.zeros((bn_file.shape[0], bn_file.shape[1], 3))
        selected_class = 0
        for selected_class in class_color_dict:
            new_bn[bn_file == selected_class] = class_color_dict[selected_class]
            selected_class = selected_class + 1
        
        numpy_img = np.asarray(new_bn, dtype=np.uint8)
        filename_mask, ext = filename.split('.')
        cv2.imwrite(destination+'/'+filename_mask+'.'+extension, numpy_img)   

        
def mask_to_binary_image(source_mask_folder, destination):
    
    # Ensure that destination folder exists or is created
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
    
    # run through the source folder for each mask file
    for filename in tqdm(os.listdir(source_mask_folder)):
        
        # read the file as a cv2 image
        grey_scale = cv2.imread(source_mask_folder + '/' + filename, cv2.IMREAD_GRAYSCALE) 

        average = np.average(grey_scale)
        if average > 100:   
            # Greyscale to Binary
            binary = cv2.threshold(grey_scale, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        else:
            binary = cv2.threshold(grey_scale, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
        filename, ext = filename.split('.')
        np.save(destination+filename+'.npy', binary)
        # save the image in the destination folder
        # cv2.imwrite(destination + '/' + filename, binary)
        
def image_to_blank_bin_mask(source_image_folder, destination, extenstion):
    
    # Ensure that destination folder exists or is created
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
    
    # run through the source folder for each mask file
    for filename in tqdm(os.listdir(source_image_folder)):
        
        # read the file as a cv2 image
        img = cv2.imread(source_image_folder + '/' + filename) 
        height = img.shape[1]
        width = img.shape[0]
        
        blank_mask = np.zeros((width, height))
        filename, ext = filename.split('.')
        
        # save the image in the destination folder
        cv2.imwrite(destination + '/' + filename+'.'+extenstion, blank_mask)
        
def png_to_binary(source_image_folder, destination):
    # Ensure that destination folder exists or is created
    if not os.path.exists(destination): # if it doesn't exist already
        os.makedirs(destination)
        
    for filename in tqdm(os.listdir(source_image_folder)):
        img = cv2.imread(source_image_folder + '/' + filename)
        filename, ext = filename.split('.')
        np.save(destination+filename+'.npy', img)
    
def binary_to_jpeg(source_binary_folder, destination, class_color_dict):
    colorize_binary(source_binary_folder, destination, class_color_dict, 'jpeg')
    
    
def png_to_jpeg(source_image_folder, source_binary_folder, destination, class_color_dict):
    png_to_binary(source_image_folder, source_binary_folder)
    binary_to_jpeg(source_binary_folder, destination, class_color_dict)
    