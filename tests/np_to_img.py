import os
import numpy as np
import matplotlib.pyplot as plt

dir_path = "G:\\PhysionetChallenge2024\\test_rot_data\\unet_outputs"
# dir_path = "G:\\PhysionetChallenge2024\\test_data\patches\\label_patches"

os.listdir(dir_path)

for file in os.listdir(dir_path):
    print(file)
    if file.endswith(".npy"):
        img = np.load(os.path.join(dir_path, file))
    # save image
    with open(os.path.join(dir_path, file.replace('.npy', '.png')), 'wb') as f:
        plt.imsave(f, img, cmap='gray')

