import os
import torch
import torchvision
import sys

sys.path.append("../../")
from mapper.styleclip_mapper import StyleCLIPMapper
from mapper.datasets.latents_dataset import LatentsDataset
from mapper.options.train_options import TrainOptions

def generate_images(opts, set_type='train'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = StyleCLIPMapper(opts).to(device)
    net.eval()
    with torch.no_grad():
        if set_type == 'train':
            latents_path = opts.latents_train_path
            images_dir = "./train_images"
        else:
            latents_path = opts.latents_test_path
            images_dir = "./test_images"
        latents = torch.load(latents_path)
        print(len(latents))
        for i, latent in enumerate(latents):
            latent = latent.unsqueeze(0).to(device)
            x, _ = net.decoder([latent], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=opts.work_in_stylespace)
            img_path = os.path.join(images_dir, f'{i:05}.jpg')
            torchvision.utils.save_image(x, img_path, normalize=True, scale_each=True, range=(-1, 1))

if __name__ == '__main__':
    opts = TrainOptions().parse()
    
    generate_images(opts, set_type='train')
    generate_images(opts, set_type='test')