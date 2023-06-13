'''
    This part of code is to generate the action unit for images and store into a .csv file with the same name as input image.
'''
import os
import sys
sys.path.append("/hy-tmp/StyleCLIP-main_prev/Action-Units-Heatmaps-master")

import csv
import time
import PIL
from au_extractor_new import au_extract
import glob, time, dlib, matplotlib.pyplot as plt, numpy as np
import AUmaps

train_set_path = "../mapper/training/train_images"
test_set_path = "../mapper/training/test_images"
train_au_path = "../mapper/training/train_aus_new"
test_au_path = "../mapper/training/test_aus_new"

index_list_path = "../mapper/truncated_index.csv"

def au_generate(isTrain=True):
    
    index_list = None
    if isTrain:
        img_path = train_set_path
        au_path = train_au_path
        # Load in the truncated train set index list.
        # Only for the sake of time saving... lol
        with open(index_list_path, 'r') as file:
            reader = csv.reader(file)
            row_list = next(reader)

            index_list = row_list
            
        print("Train set processing starts.")
    else:
        img_path = test_set_path
        au_path = test_au_path
        print("Test set processing starts.")

    # Create the output directory if it does not exist
    os.makedirs(au_path, exist_ok=True)
    
    start_time = time.time()
    num_broken_images = 0
    del_index_list = []
    
    # This branch is for training set.
    if index_list is not None:
        for i in range(889, len(index_list)):
            index = index_list[i]
            
            file_index = str(index).zfill(5)  # Convert the index to a string with leading zeros
            
            file_name = f"{train_set_path}/{file_index}.jpg"  # Construct the file path for the image file
            
            img = dlib.load_rgb_image(file_name)

            # Extract AU information
            pred = au_extract(img)
            if pred is None:
                del_index_list.append(index)
                continue
            print(pred)

            # Create a CSV file with the same name as the input image
            csv_file = os.path.join(au_path, file_index + ".csv")
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(pred)

            print("AU information saved to:", csv_file)
        del_file = os.path.join(os.getcwd(), "del_train_index.csv")
        with open(del_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(del_index_list)
    else:
        # Iterate through all the files under img_path
        for file in os.listdir(img_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                # Load the image
                img_name = os.path.splitext(file)[0]
                print((int)(img_name))
                img_file = os.path.join(img_path, file)
                img = dlib.load_rgb_image(img_file)

                # Extract AU information
                pred = au_extract(img)
                if pred is None:
                    del_index_list.append((int)(img_name))
                    continue

                print(pred)

                # Create a CSV file with the same name as the input image
                csv_file = os.path.join(au_path, img_name + ".csv")
                with open(csv_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(pred)

                print("AU information saved to:", csv_file)
        
        del_file = os.path.join(os.getcwd(), "del_test_index.csv")
        with open(del_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(del_index_list)
            
    elapsed_time = time.time() - start_time
    print("Processing completed.")
    print("Total elapsed time: {:.2f} seconds".format(elapsed_time))
    print("Number of broken images:", num_broken_images)


au_generate(False)