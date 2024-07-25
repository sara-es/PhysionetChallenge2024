import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import matplotlib.pyplot as plt

from digitization import Unet
from utils import team_helper_code
from tqdm import tqdm

patch_dir = "temp_data\\patches"
im_patch_dir = os.path.join(patch_dir, 'image_patches')
label_patch_dir = os.path.join(patch_dir, 'label_patches')
ids = os.listdir(label_patch_dir)
ids_to_predict = [f.split('-')[0] for f in ids]
ids_to_predict = team_helper_code.check_dirs_for_ids(ids_to_predict, im_patch_dir, 
                                                        label_patch_dir, True)

save_pth = "temp_data\\reconstructed_patches"
os.makedirs(save_pth, exist_ok=True)

for i, image_id in tqdm(enumerate(ids_to_predict), desc='Running U-net on images', 
                            disable=False, total=len(ids_to_predict)):
    patch_ids = [f for f in ids if f.split('-')[0] == image_id]
    patch_ids = sorted(patch_ids)

    patches_arr = np.zeros((len(patch_ids), 256, 256))
    #load in patches and populate patches arr
    for j, patch_id in enumerate(patch_ids):
        with open(os.path.join(label_patch_dir, patch_id), 'rb') as f:
            patch = np.load(f)
            patches_arr[j] = patch

    predicted_im = Unet.patching.depatchify(patches_arr, patches_arr.shape[1:])
    with open(os.path.join(save_pth, image_id + '.png'), 'wb') as f:
        plt.imsave(f, predicted_im, cmap='gray')