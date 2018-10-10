import os
import numpy as np
import cv2

# Path details for all image files
path_details = {}

patch_size = 32
bin_size = 32

for (dirpath, dirnames, filenames) in os.walk("./test/"):
    class_title = dirpath.split('/')[-1]
    path_details[class_title] = []
    for fname in filenames:
        if fname.split('.')[-1] != 'jpg':
            continue
        path_details[class_title].append((dirpath, dirpath + '/' + fname, fname))
del path_details['']

# Extracting 24 dimentional feature vectors
for class_title, path_details in path_details.items():
    for p in path_details:
        dir_path, file_path, file_name = p
        if file_name.split('.')[-1] != 'jpg':
            continue
        image = np.array(cv2.imread(file_path))
        features = []

        # Dividing image into patch with size 32 x 32
        for i in range(int(len(image)/patch_size)):
            for j in range(int(len(image[0])/patch_size)):
                patch = np.array([pixel[j*patch_size:(j+1)*patch_size] for pixel in image[i*patch_size:(i+1)*patch_size]])
                
                color_histogram = np.zeros(24).astype(np.int)
                for pix_row in range(len(patch)):
                    for pix_col in range(len(patch[0])):
                        for color_index in range(0, 3):
                            color_value = patch[pix_row][pix_col][color_index]
                            color_histogram[color_index * 8 + int(color_value / bin_size)] += 1
                features.append(color_histogram)
        features = np.array(features).astype(np.int)

        # Writing feature vector data in text file as space separated values
        print("Processing " + dir_path + "/features_" + file_name.split('.')[0] + ".txt")
        np.savetxt(dir_path + "/features_" + file_name.split('.')[0] + ".txt", features, delimiter = ' ', fmt = '%s')
