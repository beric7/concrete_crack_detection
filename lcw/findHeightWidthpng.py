# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:42:18 2020

@author: Eric Bianchi
"""

import sys

sys.path.insert(0, 'D://Python/general_utils/')

from classification_utils import findHeightWidth

directory = 'D://DATA/Datasets/COCO_VDOT_1300/Train/Train Images/'
csv_file = 'new_csvfile.csv'

findHeightWidth(directory, csv_file)