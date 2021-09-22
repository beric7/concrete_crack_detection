# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:09:41 2020

@author: Eric Bianchi
"""
import os
import cv2
from PIL import Image

"""
@param: TEST_IMAGES_DIR = the folder of images which will be sorted through.
"""
def buildImageFileList_sorted(TEST_IMAGES_DIR):
    imageFilePaths = []
    image_names = []
    
    directory = os.listdir(TEST_IMAGES_DIR)
    sortedDir = sorted(directory,key=lambda x: int(x.split('_')[0]))
    
    for imageFileName in sortedDir:
        path = TEST_IMAGES_DIR + imageFileName
        if checkImage(path):
            if imageFileName.endswith(".jpg"):
                imageFilePaths.append(path)
            elif imageFileName.endswith(".JPG"):
                imageFilePaths.append(path)
            elif imageFileName.endswith(".png"):
                im = Image.open(path)
                rgb_im = im.convert('RGB')
                (head, tail) = imageFileName.split(".")
                rgb_im.save(TEST_IMAGES_DIR + "/" + head + '.jpg')
                imageFilePaths.append(TEST_IMAGES_DIR + "/" + head + '.jpg')
            elif imageFileName.endswith(".jpeg"):
                imageFilePaths.append(path)
            image_names.append(imageFileName)
        else:
            continue
        
    return imageFilePaths, image_names

def buildImageFileList(TEST_IMAGES_DIR):
    imageFilePaths = []
    image_names = []
    
    for imageFileName in os.listdir(TEST_IMAGES_DIR):
        path = TEST_IMAGES_DIR + imageFileName
        if imageFileName.endswith(".png"):
            im = Image.open(path)
            rgb_im = im.convert('RGB')
            (head, tail) = imageFileName.split(".")
            rgb_im.save(TEST_IMAGES_DIR + "/" + head + '.jpg')
            imageFilePaths.append(TEST_IMAGES_DIR + "/" + head + '.jpg')
            image_names.append(imageFileName)
        else:
            imageFilePaths.append(path)
            image_names.append(imageFileName)
        
    return imageFilePaths, image_names

"""
@param: image = checks to see if the image can be read with openCV
"""
def checkImage(image):
     try:
         cv2.imread(image).shape
         return True
     except:
         return False
"""
@param: directory = creates a dictionary of image names (without their extension, 
                                                         and then their parent file path)
"""
def buildImageFileDict(directory):
    imageDict={}
    for file in os.listdir(directory):
        head, tail = os.path.split(file)
        name, extension = tail.split('.')
        imageDict.update({name:tail})
          
    return imageDict
         