import os

import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("../")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/models")
sys.path.append("/hy-tmp/StyleCLIP-main_prev/utils")

from mapper.datasets.latents_dataset import CustomizedDataset

from mapper.inference.inference_opts import TrainOptions
from mapper.styleclip_mapper import AU2W_Mapper

log_dir = "./generatated_image_compare_styleclip/"

opts = TrainOptions().parse()

device = 'cuda:0'
# Initialize network
net = AU2W_Mapper(opts).to(device)
# Initialize testset loader.
test_latents = torch.load(opts.latents_test_path)

test_dataset_celeba = CustomizedDataset(latents=test_latents.cpu(),
                                        opts=opts,
                                        aus_path=opts.test_aus,
                                        index_list=None,
                                        )

test_dataset = test_dataset_celeba

def interpolation_au(tar_index):
    
    test_dataset = CustomizedDataset(latents=test_latents.cpu(),
                                        opts=opts,
                                        aus_path=opts.test_aus,
                                        index_list=None,
                                        tar_idx=tar_index)
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
        
        if batch_idx == 4:
            break

        src_w = src_w.to(device)
        src_au = src_au.to(device)
        tar_w = tar_w.to(device)
        tar_au = tar_au.to(device)
        output_images = []
        path = None
        
        x_src, _ = net.decoder([src_w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=opts.work_in_stylespace)
        x_tar, _ = net.decoder([tar_w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=opts.work_in_stylespace)
        
        with torch.no_grad():
            w_hat = src_w + alpha * net.mapper(tar_au.unsqueeze(1).repeat(1, 18, 1) - src_au.unsqueeze(1).repeat(1, 18, 1), src_w)
            x_hat, w_hat, _ = net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
            output_images.append(x_hat)
            # calculate the au intensity loss
            # first store the target image.
            path = os.path.join(log_dir, f'{str(batch_idx).zfill(5)+"_"+str(tar_index).zfill(5)+str(tar_au - src_au)}.jpg')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if batch_idx == 3:
                torchvision.utils.save_image(torch.cat([x_src.detach().cpu(), x_hat.detach().cpu(), x_tar.detach().cpu()]), path,
                                     normalize=True, scale_each=True, range=(-1, 1), nrow=opts.test_batch_size)
            

                print("Image {} generated!".format(str(batch_idx)))


# main
for i in range(len(test_dataset)):
    print("Start to process {}th image.".format(i))
    interpolation_au(i)