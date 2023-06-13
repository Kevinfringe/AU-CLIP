import os

import torchvision
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append("..")
sys.path.append("../../")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/models")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/utils")

from mapper.datasets.latents_dataset import CustomizedDataset

from mapper.inference.interpolation_opts import TrainOptions
from mapper.styleclip_mapper import AU2W_Mapper

log_dir = "./animation_results/"

opts = TrainOptions().parse()

def animation_au():
    
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

    for batch_idx, [src_w, src_au, tar_w, tar_au] in enumerate(test_dataloader):
        if batch_idx > 10:
            break

        src_w = src_w.to(device)
        src_au = src_au.to(device)
        tar_w = tar_w.to(device)
        tar_au = tar_au.to(device)
        output_images = []
        
        # Function to calculate increment based on the difference from 1
        def calc_increment(val, steps=4):
            diff = 1 - val
            return diff / steps

        for i in range(5):
            alpha = 0.1 * i

            with torch.no_grad():
                w_hat = src_w + alpha * net.mapper(tar_au.unsqueeze(1).repeat(1, 18, 1) - src_au.unsqueeze(1).repeat(1, 18, 1), src_w)
                x_hat, w_hat, _ = net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
                output_images.append(x_hat)

        # Save output images.
        title = "exp_3"
        parse_and_log_images(output_images=output_images, au=batch_idx, title=title)
            
    return

def parse_and_log_images(output_images, au, title,index=None):
    if index is None:
        path = os.path.join(log_dir, title, f'image{str(au)}.jpg')
    else:
        path = os.path.join(log_dir, title, f'image{str(au)}.jpg')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(torch.cat([output_images[0].detach().cpu(), output_images[1].detach().cpu(), output_images[2].detach().cpu(), output_images[3].detach().cpu()]), path,
                                 normalize=True, scale_each=True, range=(-1, 1), nrow=opts.batch_size)



# main
animation_au()