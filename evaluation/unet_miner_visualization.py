"""
A scipt to visualize the results of the digitization model pipeline on the Challenge data.

Example usage follows the convention of the other Challenge scripts:
python clean_miner_visualization.py -i [input_folder] -o [output_folder] -v

Note that the input folder must have image files as well as ground truth signal files.
"""


import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
import traceback, argparse
from tqdm import tqdm
from PIL import Image

import helper_code
from preprocessing import cepstrum_grid_detection
from digitization.ECGminer import digitize_image
from utils import team_helper_code
import clean_miner_visualization 
from skimage.morphology import opening


def plot_signal_reconstruction(label_signal, output_signal, output_fields, mean_snr, trace, image_file, output_folder="", record_name=None, gridsize=None):
    description_string = f"""{record_name} ({output_fields['fs']} Hz)
    Reconstruction SNR: {mean_snr:.2f}
    Gridsize: {gridsize}"""

    # set up mosaic for original image, cleaned image with trace, ground truth signal, and reconstructed signal  
    mosaic = plt.figure(layout="tight", figsize=(18, 13))
    axd = mosaic.subplot_mosaic([
                                    ['original_image', 'ecg_plots'],
                                    ['trace', 'ecg_plots']
                                ])
    # plot original image
    # with Image.open(image_file) as img:
    #     axd['original_image'].axis('off')
    #     axd['original_image'].imshow(img, cmap='gray')

    # to plot u-net output (numpy array)
    axd['original_image'].axis('off')
    axd['original_image'].imshow(image_file, cmap='gray')
        
    # plot trace
    axd['trace'].xaxis.set_visible(False)
    axd['trace'].yaxis.set_visible(False)
    axd['trace'].imshow(trace.data, cmap='gray')    

    # plot ground truth and reconstructed signal
    fig, axs = plt.subplots(output_signal.shape[1], 1, figsize=(9, 12.5), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle(description_string)
    # fig.text(0.5, 0.95, description_string, ha='center')
    for i in range(output_signal.shape[1]):
        axs[i].plot(output_signal[:, i])
        axs[i].plot(label_signal[:, i])
    fig.canvas.draw()
    axd['ecg_plots'].axis('off')
    axd['ecg_plots'].imshow(fig.canvas.renderer.buffer_rgba())
    # save everything in one image
    filename = f"{record_name}_reconstruction.png"
    plt.figure(mosaic)
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()


def main(data_folder, unet_outputs_folder, output_folder, verbose):
    # Find data files.
    records = helper_code.find_records(data_folder)
    if len(records) == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the images if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    mean_snrs = np.zeros(len(records))

    for i in tqdm(range(len(records))):  
        record = os.path.join(data_folder, records[i])
        record_name = records[i]

        # get ground truth signal and metadata
        header_file = helper_code.get_header_file(record)
        header = helper_code.load_text(header_file)
        num_samples = helper_code.get_num_samples(header)

        # get filenames/paths of all records to be reconstructed
        image_files = team_helper_code.load_image_paths(record)
        image_file = image_files[0]
        if len(image_files) > 1:
            if verbose:
                print(f"Multiple images found, using image at {image_file}.")
        
        # get the gridsize from the image   
        # cleaned_image, gridsize = cepstrum_grid_detection.clean_image(image_file)
        gridsize = 37.5


        # use the u-net output as the clean image input to ecg-miner
        numpy_record = record_name.split('_')[0] + '.npy'
        restored_image = np.load(os.path.join(unet_outputs_folder, numpy_record))
        restored_image = np.where(restored_image > 0.3, 1, 0)

        # digitize with ecg-miner
        # Invert the colours if using the U-Net output as the cleaned image
        restored_image = abs(restored_image - 1)*255
        restored_image = opening(restored_image, footprint=[(np.ones((3, 1)), 1), (np.ones((1, 3)), 1)])

        signal, trace = digitize_image.digitize_image(restored_image, gridsize, num_samples)
        signal = np.nan_to_num(signal)

        # get digitization output
        signal = np.asarray(signal*1000, dtype=np.int16)
        
        # run_model.py just rewrites header file to output folder here, so we can skip that step
        # uncomment following line if we want to follow challenge save/load protocols exactly
        # output_signal, output_fields = save_and_load_wfdb(header, signal, output_folder=output_folder, record=record_name)        
        output_signal, output_fields = clean_miner_visualization.format_wfdb_signal(header, signal) # output_record is the filepath the output signal will be saved to
        
        # get ground truth signal
        label_signal, label_fields = helper_code.load_signal(record)

        # match signal lengths: make sure channel orders match and trim output signal to match label signal length
        output_signal, output_fields, label_signal, label_fields = clean_miner_visualization.match_signal_lengths(output_signal, output_fields, label_signal, label_fields)

        # compute SNR vs ground truth
        mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, mean_weighted_absolute_difference_metric = clean_miner_visualization.single_signal_snr(output_signal, output_fields, label_signal, label_fields, record_name, extra_scores=True)
        
        # add metrics to dataframe to save later
        mean_snrs[i] = mean_snr
        #TODO

        # plot signal reconstruction
        plot_signal_reconstruction(label_signal, output_signal, output_fields, mean_snr, trace, restored_image, output_folder, record_name=record_name, gridsize=gridsize)

    print(f"Finished. Overall mean SNR: {np.nanmean(mean_snrs):.2f} over {len(records)} records.")
    # save metrics to csv
    #TODO

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Visualize the results of the digitization model pipeline on the Challenge data.')
#     parser.add_argument('-i', '--data_folder', type=str, help='The folder containing the Challenge images.')
#     parser.add_argument('-o', '--output_folder', type=str, help='The folder to save the output visualization.')
#     parser.add_argument('-v', '--verbose', action='store_true', help='Print progress messages.')
#     args = parser.parse_args()

#     main(args.data_folder, args.output_folder, args.verbose)

data_folder = os.path.join('tiny_testset', 'lr_unet_tests', 'data_images')
output_folder = os.path.join('evaluation', 'viz', 'unet')
unet_outputs_folder = os.path.join('tiny_testset', 'lr_unet_tests', 'unet_outputs')
main(data_folder, unet_outputs_folder, output_folder, verbose=True)