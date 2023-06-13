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

log_dir = "./interpolation_results_increment_wholeau/"

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
    alpha = 0.2

    for batch_idx, [src_w, src_au, tar_w, tar_au] in enumerate(test_dataloader):
        if batch_idx > 10:
            break

        src_w = src_w.to(device)
        src_au = src_au.to(device)
        tar_w = tar_w.to(device)
        tar_au = tar_au.to(device)
        output_images = []
        delta_images = []
        
        x_tar, _ = net.decoder([tar_w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=opts.work_in_stylespace)

        # Function to calculate increment based on the difference from 1
        def calc_increment(val, tar, steps=5):
            diff = tar - val
            return diff / steps
        

        # for au_index in range(src_au.shape[1]):
        # Initialize increment tensor with zeros
        increment_au = torch.zeros_like(tar_au)
        #print("The shape of increment_au is: " + str(increment_au.shape))

        for i in range(5):
            # Manipulate a single action unit at a time
            increment = calc_increment(src_au[0], tar_au[0])
            increment_au[0] += increment

            with torch.no_grad():
                w_delta = net.mapper(increment_au.unsqueeze(1).repeat(1, 18, 1), src_w)
                w_hat = src_w + alpha * w_delta
                x_hat, w_hat, _ = net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
                x_delta, w_delta, _ = net.decoder([w_delta], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
                output_images.append(x_hat)
                delta_images.append(x_delta)

            print("The increased au of image {} is: ".format(batch_idx) + str(increment_au))


        # Save and output images.
        title = "image_{}".format(batch_idx)
        parse_and_log_images(output_images=output_images, au=batch_idx, tar_image=x_tar, title=title)

    return

def parse_and_log_images(output_images,au, title, tar_image,index=None):
    if index is None:
        path = os.path.join(log_dir, title, f'image{str(au)}.jpg')
    else:
        path = os.path.join(log_dir, title, f'image{str(au)}.jpg')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(torch.cat([output_images[0].detach().cpu(), output_images[1].detach().cpu(), output_images[2].detach().cpu(), output_images[3].detach().cpu(), output_images[4].detach().cpu(), tar_image.detach().cpu()]), path,
                                 normalize=True, scale_each=True, range=(-1, 1), nrow=opts.batch_size)



# main
interpolation_au()