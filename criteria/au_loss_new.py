"""
    This part of code is to calculate the action unit loss between source image and generated image.
    Note: This file is abandoned since the failure of action unit intensity value usage.
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
# sys.path.append("/hy-tmp/StyleCLIP-main_prev/Action-Units-Heatmaps-master")

from graphAU.dataset import pil_loader
from graphAU.model.ANFL import MEFARG
from graphAU.utils import *
from graphAU.conf import get_config,set_logger,set_outdir,set_env

import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToPILImage
import AUmaps

class AULoss(nn.Module):
    def __init__(self, opts):
        super(AULoss, self).__init__()
        self.opts = opts
        self.batch_size = opts.batch_size
        self.device = 'cuda:0'
        self.loss = nn.L1Loss().to(self.device)
        
        # Config and initialize action unit intensity estimator.
        # pretrained model initialization.
        self.AUdetector = AUmaps.AUdetector('../Action-Units-Heatmaps-master/model/shape_predictor_68_face_landmarks.dat', enable_cuda=True)
        self.net = self.AUdetector.AUdetector
        
        # The transformation of image: resize to 256 * 256; normalize into (0, 1)
        self.img_transform = transforms.Compose([
            transforms.Resize(256)
            # TODO: add normalization here.
        ])
        
        # Set requires_grad=False for the parameters of net
        for param in self.net.parameters():
            param.requires_grad = False
        
    
    def au_detect(self, src_imgs, gen_imgs):
        src_imgs = self.img_transform(src_imgs)
        gen_imgs = self.img_transform(gen_imgs)
        
        output_src = self.net(src_imgs.unsqueeze(0))[-1][0,:,:,:]
        ourput_gen = self.net(gen_imgs.unsqueeze(0))[-1][0,:,:,:]
        
        pred_src = torch.zeros(5, requires_grad=True)
        pred_gen = torch.zeros(5, requires_grad=True)
        
        pred_src_new = pred_src.clone()
        pred_gen_new = pred_gen.clone()
        for k in range(0,5):
            tmp = output_src[k,:,:].max()
            if tmp < 0:
                tmp = 0
            elif tmp > 5:
                tmp = 5
            pred_src_new[k] = tmp

        pred_src = pred_src_new
        
        for k in range(0,5):
            tmp = ourput_gen[k,:,:].data.max()
            if tmp < 0:
                tmp = 0
            elif tmp > 5:
                tmp = 5
            pred_gen_new[k] = tmp
        pred_gen = pred_gen_new
        
        return pred_src/5, pred_gen/5

#     def au_detect(self, src_imgs, gen_imgs):
        
#         src_imgs = self.img_transform(src_imgs)
#         gen_imgs = self.img_transform(gen_imgs)

#         output_src = self.net(src_imgs.unsqueeze(0))[-1][0,:,:,:]
#         ourput_gen = self.net(gen_imgs.unsqueeze(0))[-1][0,:,:,:]

#         pred_src = torch.zeros(5, device=self.device, requires_grad=True)
#         pred_gen = torch.zeros(5, device=self.device, requires_grad=True)

#         for k in range(0,5):
#             tmp = output_src[k,:,:].max()
#             tmp = tmp.clamp(min=0, max=5)  
#             pred_src[k] = tmp

#         for k in range(0,5):
#             tmp = ourput_gen[k,:,:].max()
#             tmp = tmp.clamp(min=0, max=5)  
#             pred_gen[k] = tmp

#         return pred_src/5, pred_gen/5

    def forward(self, src_imgs, gen_imgs, isTrain=True):
        
        # Handle each image within a batch.
        loss = 0.0
        print(src_imgs.shape)
        if isTrain:
            batch_size = self.opts.batch_size
        else:
            batch_size = self.opts.test_batch_size
        for i in range(batch_size):
            src_img = src_imgs[i]
            gen_img = gen_imgs[i]
            
            src_au, gen_au = self.au_detect(src_img, gen_img)
        
            loss += self.loss(src_au, gen_au)
        
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
# src_images = torch.randn(batch_size, num_channels, image_height, image_width,requires_grad=True).to(device)
# gen_images = torch.randn(batch_size, num_channels, image_height, image_width,requires_grad=True).to(device)

# # Calculate the loss
# loss = loss_fn(src_images, gen_images)

# loss.backward()

# print("Loss:", loss.item())