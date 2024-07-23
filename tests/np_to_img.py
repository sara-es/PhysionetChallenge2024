import os
import numpy as np
import matplotlib.pyplot as plt

base_path = "G:\\PhysionetChallenge2024\\masks"
dir_path = base_path
out_path = base_path + "unet_out_pngs"
os.makedirs(out_path, exist_ok=True)

os.listdir(dir_path)

for file in os.listdir(dir_path):
    print(file)
    if file.endswith(".npy"):
        img = np.load(os.path.join(dir_path, file))
    # save image
    with open(os.path.join(out_path, file.replace('.npy', '.png')), 'wb') as f:
        plt.imsave(f, img, cmap='gray')

