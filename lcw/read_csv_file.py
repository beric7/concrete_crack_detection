# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:07:32 2020

@author: Eric Bianchi
"""

import csv

def readFile(filename):
    """

    Parameters
    ----------
    [filename]: .csv
        The csv file that is being read.

    Returns
    -------
    [rows] : string
        Array of the rows in the csv file.
    [fields] : string
        Array of the headers in the csv file.

    """
    # initializing the titles and rows list 
    fields = [] 
    rows = [] 
    
    # reading csv file 
    with open(filename, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
          
        # extracting field names through first row 
        fields = next(csvreader) 
      
        # # extracting each data row one by one 
        # for row in csvreader: 
        #     rows.append(row) 
        
        rows = list(csv.reader(csvfile))
            
    return rows, fields
