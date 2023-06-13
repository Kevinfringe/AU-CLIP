'''
    This part of code is to generate the action unit for images and store into a .csv file with the same name as input image.
'''
import os
import csv
import time
import PIL
from graphAU.dataset import pil_loader
from au_extractor import au_extract

train_set_path = "../mapper/training/train_images"
test_set_path = "../mapper/training/test_images"
train_au_path = "../mapper/training/train_aus"
test_au_path = "../mapper/training/test_aus"

def au_generate(isTrain=True):
    
    if isTrain:
        img_path = train_set_path
        au_path = train_au_path
        print("Train set processing starts.")
    else:
        img_path = test_set_path
        au_path = test_au_path
        print("Test set processing starts.")

    # Create the output directory if it does not exist
    os.makedirs(au_path, exist_ok=True)
    
    start_time = time.time()
    num_broken_images = 0

    # Iterate through all the files under img_path
    for file in os.listdir(img_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            # Load the image
            img_name = os.path.splitext(file)[0]
            img_file = os.path.join(img_path, file)
            try:
                img = pil_loader(img_file)
            except PIL.UnidentifiedImageError:
                print("Skipping broken image:", img_file)
                os.remove(img_file)
                num_broken_images += 1
                continue

            # Extract AU information
            pred = au_extract(img)

            # Create a CSV file with the same name as the input image
            csv_file = os.path.join(au_path, img_name + ".csv")
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(pred)

            print("AU information saved to:", csv_file)
            
    elapsed_time = time.time() - start_time
    print("Processing completed.")
    print("Total elapsed time: {:.2f} seconds".format(elapsed_time))
    print("Number of broken images:", num_broken_images)


au_generate(isTrain=True)
au_generate(False)