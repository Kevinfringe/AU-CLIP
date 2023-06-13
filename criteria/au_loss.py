"""
    This part of code is to calculate the action unit loss between source image and generated image.
"""
import sys
import os
import torch
import torch.nn as nn
import logging

# Comment below three lines when not testing this class solely.
# sys.path.append("../")
# sys.path.append("/hy-tmp/StyleCLIP-main/models")
# sys.path.append("/hy-tmp/StyleCLIP-main/utils")

from graphAU.dataset import pil_loader
from graphAU.model.ANFL import MEFARG
from graphAU.utils import *
from graphAU.conf import get_config,set_logger,set_outdir,set_env

import numpy as np
from torch import nn
from torchvision.transforms import ToPILImage

class AULoss(nn.Module):
    def __init__(self, opts):
        super(AULoss, self).__init__()
        self.opts = opts
        self.batch_size = opts.batch_size
        self.device = 'cuda:0'
        self.loss = nn.L1Loss().to(self.device).eval()
        
        # Config and initialize graphAU for au detection.
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
        
        self.img_transform = image_eval()
        
        # Set requires_grad=False for the parameters of net
        for param in net.parameters():
            param.requires_grad = False
        
        self.au_net = net
    
    def au_detect(self, src_imgs, gen_imgs):
        src_imgs = self.img_transform(src_imgs)
        gen_imgs = self.img_transform(gen_imgs)
        
        src_au = self.au_net(src_imgs)
        gen_au = self.au_net(gen_imgs)
        
        return src_au, gen_au

    def forward(self, src_img, gen_imgs, isTrain=True):
        
        src_au, gen_au = self.au_detect(src_img, gen_imgs)
        
        
        loss = self.loss(src_au, gen_au)
        
        #print("the loss is: " + str(loss))
        
        
        return loss
    
# Test code for loss function
# import torch

# # Set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Create an instance of AULoss
# from mapper.options.train_options import TrainOptions
# opts = TrainOptions().parse()
# loss_fn = AULoss(opts)
# loss_fn = loss_fn.to(device)

# # Generate random source and generated images
# batch_size = opts.batch_size
# num_channels = 3
# image_height = 256
# image_width = 256
# src_images = torch.randn(batch_size, num_channels, image_height, image_width).to(device)
# gen_images = torch.randn(batch_size, num_channels, image_height, image_width).to(device)

# # Calculate the loss
# loss = loss_fn(src_images, gen_images)

# print("Loss:", loss.item())