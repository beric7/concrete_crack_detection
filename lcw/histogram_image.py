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

def img_height_width_csv(imageArray, image_paths, dict_destination):
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
    image_path_array = []
    i = 0
    
    for file in tqdm(imageArray):
        image = cv2.imread(file)
        height.append(image.shape[1])
        width.append(image.shape[0])
        image_path_array.append(image_paths[i])
        i = i + 1
    
    dict_ = {'image path':image_path_array,'height':height, 'width':width}
    df = pd.DataFrame(dict_)
    dictToCSV(dict_, ['image path', 'height', 'width'], dict_destination)
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
    xticks_labels = [ "{:.0f}-{:.0f}".format(value, bins[idx+1]) for idx, value in enumerate(bins[:-1])]
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
    

def plotHistogram(height, width, bin_number):
    
    h, fields = readFile(height)
    w, fields = readFile(width)
    
    height = []
    width = []
    ratio = []
    
    for i in range(0,len(h)):
        height.append(h[i][0])
        width.append(w[i][0])
        ratio.append(int(h[i][0])/int(w[i][0]))
    
    stylize_histogram(bin_number, height, 'Height (pixels)', 'Frequency', 'Image Heights')
    stylize_histogram(bin_number, width, 'Width (pixels)', 'Frequency', 'Image Widths')
    stylize_histogram(bin_number, ratio, 'Ratio', 'Frequency', 'Height to Width Ratio')
    '''
    fig, ax1 = plt.subplots()
    
    n, bins, patches = ax1.hist(x=height, bins=5, color='#0504aa',
                                alpha=0.7)
    plt.xticks(bins)
    plt.grid(color='white', lw = 0.5, axis='x')
    ax1.set_xlabel('Value')
    ax1.tick_params(labelrotation=45)
    ax1.set_ylabel('Frequency')
    ax1.set_title('Image Heights')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    ax1.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    
    fig, ax2 = plt.subplots()
    n, bins, patches = ax2.hist(x=width, bins=10, color='#FF0000',
                                alpha=0.7, rwidth=0.85)
    ax2.grid(axis='y', alpha=0.75)
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Value')
    ax2.tick_params(labelrotation=45)
    ax2.set_title('Image Widths')
    maxfreq = np.max(n)
    # Set a clean upper y-axis limit.
    ax2.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    
    fig, ax3 = plt.subplots()
    n, bins, patches = ax3.hist(x=ratio, bins=10, color='#FF0000',
                                alpha=0.7, rwidth=0.85)
    ax3.grid(axis='y', alpha=0.75)
    ax3.set_ylabel('Frequency')
    ax3.set_xlabel('Value')
    ax3.tick_params(labelrotation=45)
    ax3.set_title('Image ratio')
    maxfreq = np.max(n)
    # Set a clean upper y-axis limit.
    #ax3.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    '''
    
