from graphAU.dataset import pil_loader
from au_extractor import au_extract

# img_path = './00019.jpg'

# img = pil_loader(img_path)

# au_extract(img)


import torch
import numpy as np
from torchvision.transforms import ToPILImage

# Generate a batch of input images
N = 4  # Number of images in the batch
C = 3  # Number of channels
H = 256  # Height of the images
W = 256  # Width of the images

# Create a random batch of images
input_images = torch.randn(N, C, H, W)

# Initialize the ToPILImage transformation
to_pil = ToPILImage()

# Convert the batch of tensors into individual PIL images
pil_images = [to_pil(input_images[i]) for i in range(N)]

# Pass the individual PIL images to au_extract and collect the output action units
output_au = []
for img in pil_images:
    # Call the au_extract function with the PIL image
    action_units = au_extract(img)

    # Collect the output action units
    output_au.append(action_units)

# Convert the list of output action units to a batched tensor
output_au_batch = torch.stack([torch.from_numpy(np.array(au)) for au in output_au])

# Print the output batch of action units
print("Output Action Units:")
print(output_au_batch)
