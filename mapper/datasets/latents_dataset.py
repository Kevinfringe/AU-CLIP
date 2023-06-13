import torch
import csv
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

del_list_train = "../utils/del_train_index.csv"
del_list_test = "../utils/del_test_index.csv"


class LatentsDataset(Dataset):

	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):

		return self.latents[index]

class StyleSpaceLatentsDataset(Dataset):

	def __init__(self, latents, opts):
		padded_latents = []
		for latent in latents:
			latent = latent.cpu()
			if latent.shape[2] == 512:
				padded_latents.append(latent)
			else:
				padding = torch.zeros((latent.shape[0], 1, 512 - latent.shape[2], 1, 1))
				padded_latent = torch.cat([latent, padding], dim=2)
				padded_latents.append(padded_latent)
		self.latents = torch.cat(padded_latents, dim=2)
		self.opts = opts

	def __len__(self):
		return len(self.latents)

	def __getitem__(self, index):
		return self.latents[index]


class CustomizedDataset(Dataset):
    '''
        Only the latent code and action unit needed to be loaded in.
        The latents code are pre-loaded via torch.load().
    '''
    def __init__(self, latents, opts, aus_path, index_list, tar_idx=None):
        
        self.latents = latents
        self.opts = opts
        self.aus_path = aus_path

        self.index_list = index_list
        self.tar_idx = tar_idx
        
        # below code is for the new version of au loss.
#         if self.index_list is not None:
#             with open(del_list_train, 'r') as file:
#                 reader = csv.reader(file)
#                 row_list = next(reader)

#                 self.del_list = row_list
#         else:
#             with open(del_list_test, 'r') as file:
#                 reader = csv.reader(file)
#                 row_list = next(reader)

#                 self.del_list = row_list
        

    def __len__(self):
        if self.index_list is not None:
            return len(self.index_list)
        else:
            return len(self.latents)

    def __getitem__(self, index):
        if self.index_list is not None:
            input_index = index
            index = (int)(self.index_list[index])
        
        # Only need to comment below if branch if want to reverse to stable version.
#         if str(index) in self.del_list or index in self.del_list:
 
#             if self.index_list is not None:
#                 index = (int)(self.index_list[input_index + 1])
#             else:
#                 index += 1
        
        src_latent_code = self.latents[index]
        
        # 57 has been removed during truncation, so no need for this two lines of code.
        # if index == 57:
        #     index = 58
        
        file_index = str(index).zfill(5)  # Convert the index to a string with leading zeros

        aus_file = f"{self.aus_path}/{file_index}.csv"  # Construct the file path for the CSV file

        # Load action unit values from the CSV file
        with open(aus_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            src_au = next(reader)  # Get the first (and only) row of the CSV file
            src_au = [float(a) for a in src_au]  # Convert string values to floats
            src_au_tensor = torch.tensor(src_au, dtype=torch.float32)  # Convert action units to a PyTorch tensor

        # Randomly pick another index
        # if self.index_list is None:
        random_index = 0
        if self.tar_idx is None:
            random_index = random.randint(0, len(self.latents) - 1)
            while random_index == index or random_index == 57:
                random_index = random.randint(0, len(self.latents) - 1)
        else:
            random_index = self.tar_idx
        # else:
        #     random_index = random.randint(0, len(self.index_list) - 1)
        #     while random_index == index or random_index == 57 or (str(random_index) in self.del_list) or (str(random_index) not in self.index_list):
        #         random_index = random.randint(0, len(self.index_list) - 1)

        tar_latent_code = self.latents[random_index]
        tar_file_index = str(random_index).zfill(5)  # Convert the random index to a string with leading zeros

        tar_aus_file = f"{self.aus_path}/{tar_file_index}.csv"  # Construct the file path for the CSV file

        # Load action unit values from the CSV file of the randomly picked index
        with open(tar_aus_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            tar_au = next(reader)  # Get the first (and only) row of the CSV file
            tar_au = [float(a) for a in tar_au]  # Convert string values to floats
            tar_au_tensor = torch.tensor(tar_au, dtype=torch.float32)  # Convert action units to a PyTorch tensor

        return src_latent_code, src_au_tensor, tar_latent_code, tar_au_tensor

    
# test code for CustomizedDataset
# import sys
# sys.path.append("../../")

# from mapper.options.train_options import TrainOptions

# opts = TrainOptions().parse()
# train_latents = torch.load(opts.latents_train_path)

# aus_path = opts.train_aus

# # Instantiate the dataset
# dataset = CustomizedDataset(train_latents, opts, aus_path)

# # Test accessing a single item from the dataset
# index = 0  # Choose an index to access a specific item
# src_latent_code, src_au_tensor, tar_latent_code, tar_au_tensor = dataset[index]

# # Print the retrieved values
# print("Src Latent code:", src_latent_code)
# print("Src Action units:", src_au_tensor)
# print("Target Latent code:", tar_latent_code)
# print("Target Action units:", tar_au_tensor)


# # Test iterating through the dataset using a DataLoader
# batch_size = 2  # Choose an appropriate batch size
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Iterate through the dataloader
# for batch in dataloader:
#     src_latent_code, src_au_tensor, tar_latent_code, tar_au_tensor = batch
#     # Perform operations on the batched data
#     print("Batch latent code shape:", src_latent_code.shape)
#     print("Batch action units shape:", src_au_tensor.shape)
#     print("Batch latent code shape:", tar_latent_code.shape)
#     print("Batch action units shape:", tar_au_tensor.shape)
    
#     break  # Stop after the first batch for demonstration purposes