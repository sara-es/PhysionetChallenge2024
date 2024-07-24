import os
import numpy as np
import matplotlib.pyplot as plt

base_path = "G:\\PhysionetChallenge2024\\temp_data\\"
dir_path = base_path + "unet_outputs"
out_path = base_path + "unet_outputs_imgs"
os.makedirs(out_path, exist_ok=True)

os.listdir(dir_path)

for file in os.listdir(dir_path):
    print(file)
    if file.endswith(".npy"):
        img = np.load(os.path.join(dir_path, file))
    # save image
    with open(os.path.join(out_path, file.replace('.npy', '.png')), 'wb') as f:
        plt.imsave(f, img, cmap='gray')

