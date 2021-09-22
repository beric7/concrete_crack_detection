# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:48:03 2020

@author: Eric Bianchi
"""
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import numpy as np
import cv2
import os
from write_to_csv import dictToCSV
from read_csv_file import readFile
from tqdm import tqdm
import pandas as pd

def img_height_width_csv(image_paths):
    """
    Parameters
    ----------
    imageArray : array-type
        array of images.

    Returns
    -------
    histogram

    """
    height = []
    width = []
    i = 0
    
    for file in tqdm(image_paths):
        image = cv2.imread(file)
        height.append(image.shape[1])
        width.append(image.shape[0])
        i = i + 1
    
    dict_ = {'image path':image_paths,'height':height, 'width':width}
    df = pd.DataFrame(dict_)
    return df
    # arrayToCSV(width,['width'], width_csv)
    
def stylize_histogram(bin_number, data, x_label, y_label, title):
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x=data, color='#0504aa', bins=bin_number, alpha=0.7)
    
    maxfreq = n.max()
    
    # Set a clean upper y-axis limit.
    ax.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
      
    #grid
    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color='white', lw = 1)

    # ticks
    xticks = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]
    xticks_labels = [ "{:.2f}-{:.2f}".format(value, bins[idx+1]) for idx, value in enumerate(bins[:-1])]
    plt.xticks(xticks, labels = xticks_labels)
    ax.tick_params(axis='x', which='both',length=0, labelrotation=45)
    plt.yticks([])
    
    # Hide the right and top spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for idx, value in enumerate(n):
        if value > 0:
            plt.text(xticks[idx], value+5, int(value), ha='center', fontsize = 8)
    plt.title(title, loc = 'center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    

def plotHistogram(df, bin_number):
    
    height = df['height']
    width = df['width']
    image_path = df['image path']
    ratio = height / width
    
    if not os.path.exists('./sorted/sub_image/'):
        os.makedirs('./sorted/sub_image/')
    new_height = []
    new_width = []
    new_ratio = []
    for i in tqdm(range(0, len(ratio))):
        if ratio[i] < 4 and ratio[i] > 0.25:
            if height[i] > 80 and width[i] > 80:
                new_height.append(height[i])
                new_width.append(width[i])
                new_ratio.append(ratio[i])
                img = cv2.imread(image_path[i])
                name = os.path.basename(image_path[i])
                cv2.imwrite('./sorted/sub_image/'+name, img)
                
    
    stylize_histogram(bin_number, new_height, 'Height (pixels)', 'Frequency', 'Image Heights')
    stylize_histogram(bin_number, new_width, 'Width (pixels)', 'Frequency', 'Image Widths')
    stylize_histogram(bin_number, new_ratio, 'Ratio', 'Frequency', 'Height to Width Ratio')
