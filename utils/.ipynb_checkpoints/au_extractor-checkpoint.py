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
from graphAU.dataset import pil_loader
from graphAU.model.ANFL import MEFARG
from graphAU.utils import *
from graphAU.conf import get_config,set_logger,set_outdir,set_env

index_to_keep = [0, 1, 2, 4, 6, 9, 21, 22]


def au_extract(imgs):
    '''
        This function can either deal with single image or batch.
    '''
    # variable configuration.
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)

    dataset_info = hybrid_prediction_infolist
    img_path = conf.input

    net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
    
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)


    net.eval()
    img_transform = image_eval()
    #img = pil_loader(img_path)
    
    imgs_ = img_transform(imgs)
    
    if len(imgs_.shape) == 3: # a single image
        imgs_ = imgs_.unsqueeze(0)
    
    if torch.cuda.is_available():
        net = net.cuda()
        imgs_ = imgs_.cuda()
        
    with torch.no_grad():
        pred = net(imgs_)
        pred = pred.squeeze().cpu().numpy()
    # print(" ***** raw au data ********")
    # print(pred)
    # print(" ***** au data after truncation ******")
    # print(pred[index_to_keep])
        
    # get the prediction for each action unit.
#     infostr_probs,  infostr_aus = dataset_info(pred, 0.5)
    
#     print("the output infostr_probs: ")
#     print(infostr_probs)
#     print(infostr_aus)
    
    return pred[index_to_keep]
