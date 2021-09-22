# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:14:27 2020

@author: Eric Bianchi
"""
from model_plus import createDeepLabv3Plus
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
import numpy as np
import os
import cv2

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def process_im(image_path):
    
    image = cv2.imread(image_path)
    width, height = image.shape[1], image.shape[0]
    min_ = min(width, height)
    dim = [min_, min_]
    # process crop width and height for max available dimension:
    crop_width = dim[0] if dim[0]<image.shape[1] else image.shape[1]
    crop_height = dim[1] if dim[1]<image.shape[0] else image.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = image[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    image = cv2.resize(crop_img, (512,512), interpolation=cv2.INTER_NEAREST)
    img = image.transpose(2,0,1)
    
    return img

def generate_images(model, image_path, gt_mask_path):
    
    # image
    img = process_im(image_path)
    img = img.reshape(1,3,512,512)
    _, channels, height, width = img.shape
    mask = process_im(gt_mask_path)
    gt_mask = torch.empty(height, width, dtype=torch.long)
    import numpy as np
    mapping = {(0,0,0): 0, (255,255,255): 1}
    target = torch.from_numpy(mask)
    tt = target.data.cpu().numpy()
    
        
    for k in mapping:
         # Get all indices for current class
         idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
         validx = (idx.sum(0) == 3)  # Check that all channels match
         gt_mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
         t = gt_mask.data.cpu().numpy()

    
    with torch.no_grad():
        mask_pred = model(torch.from_numpy(img).type(torch.cuda.FloatTensor))
    # color mapping corresponding to classes
    # ---------------------------------------------------------------------
    # 0 = background
    # 1 = concrete
    # 2 = steel
    # 3 = metal decking
    # ---------------------------------------------------------------------

    y_pred_tensor = mask_pred
    pred = torch.argmax(y_pred_tensor, dim=1)
    y_pred = pred.data.cpu().numpy()
    
    y_pred = y_pred.ravel()
    y_true = gt_mask.data.cpu().numpy().ravel()
    
    confm = confusion_matrix(y_true, y_pred, labels=[0,1])
    iOU = jaccard_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return confm, iOU, f1


def make_image_list(image_dir):
    image_list_array = []
    for image_name in os.listdir(image_dir):
        image_list_array.append(image_name)
    return image_list_array

def iterate_data(model, source_image_dir):
    n = 0
    confm_sum = np.zeros((2,2))
    iOU_sum = 0
    f1_sum = 0
    image_list_array = make_image_list(source_image_dir+'Images/')
    for image_name in tqdm(image_list_array):
        image_path = source_image_dir + 'Images/' + image_name
        name = image_name.split('.')[-2]
        mask_name = name + '.png'
        gt_mask_path = source_image_dir + 'Masks/' + mask_name
        confm, iOU, f1 = generate_images(model, image_path, gt_mask_path)
        
        confm_sum +=confm
        iOU_sum += iOU
        f1_sum += f1
        n += 1
        
    iOU = iOU_sum / n
    f1 = f1_sum / n
    
    return iOU, f1, confm_sum


