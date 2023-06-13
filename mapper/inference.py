import os
from argparse import Namespace

import torchvision
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
import time

from tqdm import tqdm
import glob, time, dlib, matplotlib.pyplot as plt, numpy as np

sys.path.append(".")
sys.path.append("../")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/models")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/utils")

from mapper.datasets.latents_dataset import CustomizedDataset

from mapper.inference_opts import TrainOptions
from mapper.styleclip_mapper import AU2W_Mapper
from utils.au_extractor_new import au_extract

log_dir = "./generatated_image_au_loss_only/"

opts = TrainOptions().parse()

def interpolation_au():
    
    device = 'cuda:0'
    # Initialize network
    net = AU2W_Mapper(opts).to(device)
    # Initialize testset loader.
    test_latents = torch.load(opts.latents_test_path)

    test_dataset_celeba = CustomizedDataset(latents=test_latents.cpu(),
                                            opts=opts,
                                            aus_path=opts.test_aus,
                                            index_list=None)

    test_dataset = test_dataset_celeba
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=1,  # the batch size set to 1 since image is supposed to processed individually.
                                  shuffle=False,
                                  num_workers=int(opts.test_workers),
                                  drop_last=True)
    
    print("Number of test samples: {}".format(len(test_dataset)))
    
    net.eval()
    agg_loss_dict = []
    alpha = 0.1
    loss = nn.MSELoss()
    au_loss = 0.0

    for batch_idx, [src_w, src_au, tar_w, tar_au] in enumerate(test_dataloader):
        
        src_w = src_w.to(device)
        src_au = src_au.to(device)
        tar_w = tar_w.to(device)
        tar_au = tar_au.to(device)
        output_images = []
        path = None
        
        x_tar, _ = net.decoder([tar_w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=opts.work_in_stylespace)
        
        with torch.no_grad():
            w_hat = src_w + alpha * net.mapper(tar_au.unsqueeze(1).repeat(1, 18, 1) - src_au.unsqueeze(1).repeat(1, 18, 1), src_w)
            x_hat, w_hat, _ = net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
            output_images.append(x_hat)
            # calculate the au intensity loss
            # first store the target image.
            path = os.path.join(log_dir, f'{str(batch_idx).zfill(5)}.jpg')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torchvision.utils.save_image(torch.cat([x_tar.detach().cpu()]), path,
                                 normalize=True, scale_each=True, range=(-1, 1), nrow=opts.test_batch_size)
            
        image = dlib.load_rgb_image(path)

        pred_tar = au_intensity_esimation(image)


        print("Image {} generated!".format(str(batch_idx)))

        # Save and output images.
        parse_and_log_images(output_images=output_images, batch_idx=batch_idx)
        
        image = dlib.load_rgb_image(path)
            
        pred_gen = au_intensity_esimation(image)
        
        if pred_gen is None or pred_tar is None:
            continue
        
        au_loss += loss(torch.from_numpy(pred_tar), torch.from_numpy(pred_gen)).item()
        
        print("au_loss for image {}: ".format(batch_idx) + str( au_loss / (batch_idx + 1)))
    
    au_loss = au_loss / len(test_dataset)
    print("au_loss is: " + str(au_loss))
    

    return

def parse_and_log_images(output_images,batch_idx,index=None):
    if index is None:
        path = os.path.join(log_dir, f'{str(batch_idx).zfill(5)}.jpg')
    else:
        path = os.path.join(log_dir, f'{str(batch_idx).zfill(5)}.jpg')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(torch.cat([output_images[0].detach().cpu()]), path,
                                 normalize=True, scale_each=True, range=(-1, 1), nrow=opts.test_batch_size)

def au_intensity_esimation(image):

    # Extract AU information
    pred = au_extract(image)

    return pred

# main
interpolation_au()