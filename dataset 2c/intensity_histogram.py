import os
import numpy as np
import cv2

# Path details for all image files
path_details = {}

patch_size = 7

for (dirpath, dirnames, filenames) in os.walk("./Train/"):
    path_details = []
    for fname in filenames:
        if fname.split('.')[-1] != 'png':
            continue
        path_details.append((dirpath, dirpath + '/' + fname, fname))

# Extracting 2 dimentional feature vectors
for p in path_details:
    dir_path, file_path, file_name = p
    if file_name.split('.')[-1] != 'png':
        continue
    image = np.array(cv2.imread(file_path))
    features = []

    # Dividing image into patch with size 7 x 7
    for i in range(int(len(image)-patch_size)):
        for j in range(int(len(image[0])-patch_size)):
            patch = np.array([pixel[patch_size+j:patch_size+j+6] for pixel in image[patch_size + i:patch_size+i+6]])

            # Storing mean and variance of intensity of each patch as a 2d vector
            intensity_param = np.array([np.mean(patch), np.var(patch)])
            features.append(intensity_param)
    features = np.array(features).astype(np.float)

    # Writing feature vector data in text file as space separated values
    print("Processing " + dir_path + "features_" + file_name.split('.')[0] + ".txt")
    np.savetxt(dir_path + "features_" + file_name.split('.')[0] + ".txt", features, delimiter = ' ', fmt = '%s')