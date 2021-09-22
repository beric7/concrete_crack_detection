gr# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:47:13 2020

@author: Eric Bianchi
"""

# # Standard imports
import cv2
import numpy as np
import os
import math
from PIL import Image

def resize_image(image):
    h = image.shape[0]
    w = image.shape[1]
    max_dim = max(h,w)
    
    if (max_dim) > 300:
        reduce = max_dim/300
        h = int(h/reduce)
        w = int(w/reduce)
        image = cv2.resize(image, (w, h)) 
    return image
    

def blobDetection():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
     
     # Change thresholds
    #params.minThreshold = 0;
    #params.maxThreshold = 255;
     
    # Filter by Area.
    params.filterByArea = False
    params.minArea = 10
     
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
     
    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = 0.5
     
    # Filter by Inertia
    params.filterByInertia =False
    #params.minInertiaRatio = 0.5

    # Create a detector with the parameters 
    detector = cv2.SimpleBlobDetector_create(params) 
    
    return detector

def convert_HSV(hsv_min, hsv_max):
    # lime = 0,255,0
    # H: 0-179, S: 0-255, V: 0-255
    # Lime 'H' = 120
    
    # Green_lower = 80, 25, 25
    # Green_upper = 150, 100, 100
    # Pink_lower = 310, 10, 50
    # Pink_upper = 330, 100, 100
    # Red_lower = 0, 50, 50
    # Red_upper = 10, 100, 100
    # Blue_lower = 220, 44, 96
    # Blue_upper = 240, 98, 50
    
    min_h = hsv_min[0]
    min_h = min_h/360*179
    min_s = hsv_min[1]
    min_s = min_s/100*255
    min_v = hsv_min[2]
    min_v = min_v/100*255
    
    max_h = hsv_max[0]
    max_h = max_h/360*179
    max_s = hsv_max[1]
    max_s = max_s/100*255
    max_v = hsv_max[2]
    max_v = max_v/100*255
    
    MIN = np.array([min_h, min_s, min_v],np.uint8)
    MAX = np.array([max_h, max_s, max_v],np.uint8)
    
    return MIN, MAX
    
def laser_detection(image, image_path):
    original = image
    
    # Red detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_min = [340, 10, 50]
    hsv_max = [356, 100, 100]
    MIN, MAX = convert_HSV(hsv_min, hsv_max)
    
    # create the mask using the colors
    mask = cv2.inRange(hsv, MIN, MAX)
    
    # dialiate the pixels to expand the points.
    kernel = np.ones((2,2),np.uint8)
    dialated_mask = cv2.dilate(mask,kernel,iterations = 1)
    result_mask = cv2.bitwise_and(image, image, mask=dialated_mask)
    
    result = cv2.cvtColor(result_mask, cv2.COLOR_BGR2GRAY)
    watershed = cv2.threshold(result, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("watershed", watershed);
    # make detector
    detector = blobDetection()
    
    inverted_img = cv2.bitwise_not(watershed)
    cv2.imshow("inverted_img", inverted_img);
    
    # Detect blobs.
    keypoints = detector.detect(inverted_img)
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("keypoints", im_with_keypoints);
    cv2.imshow('mask',mask)
    cv2.imshow('res',result)
    cv2.waitKey(0)
    
    # save the resulting image with the original side by side.
    # folder, file = os.path.split(image_path)
    # file, extension = file.split('.')
    # newIm = np.hstack((original, im_with_keypoints))
    # cv2.imwrite(folder + "/laser_combined_result" + "." + extension, newIm)
    
    return keypoints


def mask_detection(image, image_path):
    original = image

    # Pink Detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_min = [330, 10, 100]
    hsv_max = [335, 100, 100]
    MIN, MAX = convert_HSV(hsv_min, hsv_max)
    
    # create the mask using the colors
    mask = cv2.inRange(hsv, MIN, MAX)
    
    # erode and then dialiate to remove noise
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dialated_mask = cv2.dilate(erosion,kernel,iterations = 1)

    # get the pixel area of the mask
    area = np.sum(mask)/255
    
    # make the result of the dialated mask on the image.
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # show the masks
    # cv2.imshow('frame',image)
    cv2.imshow('mask',mask)
    cv2.imshow('res',result)
    cv2.waitKey(0)
    
    # save the resulting image with the original side by side.
    folder, file = os.path.split(image_path)
    file, extension = file.split('.')
    newIm = np.hstack((original, result))
    cv2.imwrite(folder + "/combined_result" + "." + extension, newIm)
    
    print('area in pixels: ' + str(area))
    return area

def pixel_to_real_world(keypoints, laser_offset):
    
    point1 = keypoints[0].pt
    point2 = keypoints[1].pt
    # print(str(point1))
    # print(str(point2))
    distance = ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**(1/2)
    # print('pixel distance: ' + str(distance))
    pixel_per_inch = (distance / laser_offset)
    # print('pixels per inch: ' + str(pixel_per_inch))
    
    return pixel_per_inch

def draw_polygon(keypoints, img):
    

    pts = np.array([keypoints[0],
                    keypoints[1],
                    keypoints[2],
                    keypoints[3]], np.int32)
    pts = pts.reshape((-1,1,2))
    dark = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
    cv2.polylines(dark,[pts],True,(0,255,255))
    return dark

def quantify_mask(image_path, laser_offset):
    # assuming that we have masked the spalling to a known pixel color
    # assuming that we have a known laser offset.
    # assuming that we have a workable laser color HSV range
    
    # print('laser offset: ' + str(laser_offset))
    # path1 = 'D://' + 'Pink_spalling.png'
    image = cv2.imread(image_path)
    resized_image = resize_image(image) 
    cv2.imshow('blurred',resized_image)
    cv2.waitKey(0)
    keypoints = laser_detection(resized_image, image_path)
    area_in_pixels = mask_detection(resized_image, image_path)
    
    pixel_per_inch = pixel_to_real_world(keypoints, laser_offset)
    
    area_in_inches = area_in_pixels/pixel_per_inch**2
    
    print('defect area size is approximately: ' + str(area_in_inches))
    
    return resized_image, keypoints

def order_points(keypoint_array):
    
    temp = keypoint_array
    left = []
    
    left_most = np.argmin(temp[:,0])
    left.append(temp[left_most])
    temp = np.delete(temp, left_most, 0)
    
    second_left_most = np.argmin(temp[:,0])
    left.append(temp[second_left_most])
    temp = np.delete(temp, second_left_most, 0)

    right = temp

    if left[0][1] > left[1][1]:
        bottom_left = left[0]
        top_left = left[1]
    else:
        bottom_left = left[1]
        top_left = left[0]
        
    if right[0][1] > right[1][1]:
        bottom_right = right[0]
        top_right= right[1]
    else:
        bottom_right = right[1]
        top_right = right[0]
    
    return [bottom_left, top_left, top_right, bottom_right]

def find_angle(a, b, c):
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
    
def return_average_angle(ordered_points):
    angle_1 = find_angle(ordered_points[3],ordered_points[0],ordered_points[1])
    angle_2 = find_angle(ordered_points[0],ordered_points[1],ordered_points[2])

    return (angle_1+angle_2)/2

# image_path = 'D://masks/Masked_Patched_Corrosion_png_v1/' + 'skewed_4_laser.png'
image_path = 'D://masks/Masked_Patched_Corrosion_png_v1/plum_4_laser.png'
laser_offset = 4 # inches
resized_image, keypoints = quantify_mask(image_path, laser_offset)

# [bottom_left, top_left, top_right, bottom_right]
keypoint_array = np.array([[keypoints[0].pt[0], keypoints[0].pt[1]],
         [keypoints[1].pt[0], keypoints[1].pt[1]],
         [keypoints[2].pt[0], keypoints[2].pt[1]],
         [keypoints[3].pt[0], keypoints[3].pt[1]]])

ordered_points = order_points(keypoint_array)

# polygon = draw_polygon(ordered_points, resized_image)

average_angle = return_average_angle(ordered_points)
error = abs((average_angle-90)/90)
print(error)

if error < 0.02: 
    print('This is a good picture to take!')
else:
    print('This is a bad picture to take!')

# cv2.imshow('img',polygon)
# cv2.waitKey(0)

