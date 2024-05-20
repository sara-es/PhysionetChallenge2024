"""
A scipt to visualize different image cleaning methods.

Example usage follows the convention of the other Challenge scripts:
python image_cleaning_visualization.py -i [input_folder] -o [output_folder] -v
"""

import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

from tqdm import tqdm
import argparse
from dataclasses import dataclass
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import imageio.v3 as iio # may be faster to use PIL, check this

from preprocessing import hough_grid_detection, cepstrum_grid_detection, cepstrum_bresenham
import helper_code
from utils import team_helper_code


@dataclass
class CleanedImage():
    method: str
    image: np.ndarray
    gridsize: int


def plot_cleaned_images(original_image, cleaned_images, record_name, output_folder):
    fig = plt.figure(layout="tight")
    n_images = len(cleaned_images) + 1 

    # show images in two columns
    cols = 2
    rows = (n_images + 1) // cols

    # plot original image
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(original_image, cmap='gray')
    ax.set_title('Original Image: ' + record_name)
    ax.axis('off')

    # plot cleaned images
    for i, cleaned_image in enumerate(cleaned_images):
        ax = fig.add_subplot(rows, cols, i+2)
        ax.imshow(cleaned_image.image, cmap='gray')
        ax.set_title(f'{cleaned_image.method} - Gridsize: {cleaned_image.gridsize}')
        ax.axis('off')

    plt.savefig(os.path.join(output_folder, record_name + '_cleaned_images.png'))
    plt.close()


def main(data_folder, output_folder, verbose):
    # Find data files.
    records = helper_code.find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    for i in tqdm(range(len(records))):  
        record = os.path.join(data_folder, records[i])
        record_name = records[i]

        # get filenames/paths of all records to be reconstructed
        image_files = team_helper_code.load_image_paths(record)
        image_file = image_files[0]
        original_image = iio.imread(image_file)

        # plot to view the raw image, and the RGB channels individually
        # note: these might be backwards - I think cv2 uses BGR, not RGB
        blue_im = original_image[:, :, 0].astype(np.float32)  # this channel doesn't show up the grid very much
        green_im = original_image[:,:,1].astype(np.float32)
        red_im = original_image[:, :, 2].astype(np.float32)
        im_bw = (0.299*red_im + 0.114*blue_im + 0.587*green_im) # conversion from RGB -> greyscale
        im_bw[im_bw>80] = 255 # magic number to clean image a little bit

        # rotate image using cepstrum method
        angle, gridsize = cepstrum_grid_detection.get_rotation_angle(im_bw)

        rotated_image = sp.ndimage.rotate(original_image, angle, axes=(1, 0), reshape=True)

        # unet preprocessing

        

        images = [
            CleanedImage('Cepstrum', rotated_image, gridsize),
            # CleanedImage('Cepstrum w/ Bresenham', cep2_image, cep2_gridsize)
        ]

        # plot signal reconstruction
        plot_cleaned_images(original_image, images, record_name, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the results of the digitization model pipeline on the Challenge data.')
    parser.add_argument('-i', '--data_folder', type=str, help='The folder containing the Challenge images.')
    parser.add_argument('-o', '--output_folder', type=str, help='The folder to save the output visualization.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print progress messages.')
    args = parser.parse_args()

    main(args.data_folder, args.output_folder, args.verbose)


