'''
    This part of code is for action unit extraction of each image (input or generated output).
    This module used the pretrained action unit recognition model OpenGraphAU
    Reference Github link: https://github.com/CVI-SZU/ME-GraphAU
'''
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import logging
sys.path.append("../utils")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/Action-Units-Heatmaps-master")
import glob, time, dlib, matplotlib.pyplot as plt, numpy as np
import AUmaps


def au_extract(imgs):
    '''
        This function can either deal with single image or batch.
    '''
    AUdetector = AUmaps.AUdetector('../Action-Units-Heatmaps-master/model/shape_predictor_68_face_landmarks.dat', enable_cuda=True)

    pred, map, img = AUdetector.detectAU(imgs)
    # print(" ***** raw au data ********")
    # print(pred)
    # print(" ***** au data after truncation ******")
    # print(pred[index_to_keep])
        
    # get the prediction for each action unit.
#     infostr_probs,  infostr_aus = dataset_info(pred, 0.5)
    
#     print("the output infostr_probs: ")
#     print(infostr_probs)
#     print(infostr_aus)
    
    return pred
