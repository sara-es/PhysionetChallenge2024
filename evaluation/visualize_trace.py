"""
A test script to display the results of the digitization model pipeline on the Challenge data.

Our general digitization process is
1. generate testing images and masks
2. preprocess testing images to fix rotation and estimate grid size/scale
3. generate u-net patches
4. run u-net on patches
5. recover image with signal outline from u-net outputs
6. reconstruct signal and trace from u-net output

Here we'll assume that all of that has been done, and we just need to load the respective files 
and display them all together in a single image. The only exception is the trace, which we'll
re-generate from the u-net output image.

To generate the data first, use evaluation/generate_data.py. 
"""

import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import team_code, helper_code
from evaluation import eval_utils
from utils import model_persistence, team_helper_code


def visualize_trace(test_images_dir, unet_outputs_dir, reconstructed_signal_dir,
                    wfdb_records_dir, visualization_save_dir):
    records = helper_code.find_records(wfdb_records_dir)
    ids = team_helper_code.check_dirs_for_ids(records, test_images_dir, 
                                              unet_outputs_dir, True)
    image_ids = team_helper_code.find_available_images(ids, 
                                                       test_images_dir, verbose=True)
    unet_ids = team_helper_code.find_available_images(ids, 
                                                      unet_outputs_dir, verbose=True)
    if len(image_ids) != len(unet_ids) and len(image_ids) > 0:
        print(image_ids, unet_ids)
        raise ValueError("Number of image and U-Net output files do not match, please make "+\
                         "sure both have been generated and saved correctly.")
    
    image_ids = sorted(list(image_ids))
    unet_ids = sorted(list(unet_ids))

    # load resnet for classification
    # classification_model = model_persistence.load_model('model', 'classification_model')
    # resnet_model = classification_model["model"]
    # dx_classes = classification_model["dx_classes"]
    
    for i in range(len(image_ids)):
        # set up mosaic for original image, u-net output with trace, ground truth signal, 
        # and reconstructed signal
        mosaic = plt.figure(layout="tight", figsize=(18, 13))
        axd = mosaic.subplot_mosaic([
                                        ['original_image', 'ecg_plots'],
                                        ['trace', 'ecg_plots']
                                    ])
        # load image
        # TODO: this assumes image needs no preprocessing, or preprocessing has already been done
        with Image.open(os.path.join(test_images_dir, image_ids[i] + '.png')) as img:
            axd['original_image'].axis('off')
            axd['original_image'].imshow(img, cmap='gray')

        # load u-net output
        unet_image_path = os.path.join(unet_outputs_dir, unet_ids[i] + '.npy')
        with open(unet_image_path, 'rb') as f:
            unet_image = np.load(f)
        # digitize signal from u-net ouput
        record_path = os.path.join(wfdb_records_dir, records[i]) 
        header_txt = helper_code.load_header(record_path)
        reconstructed_signal, trace = team_code.reconstruct_signal(records[i], unet_image, 
                                 header_txt, reconstructed_signal_dir, save_signal=True)
        reconstructed_signal = np.asarray(np.nan_to_num(reconstructed_signal))
        
        # plot trace of signal from u-net output
        axd['trace'].xaxis.set_visible(False)
        axd['trace'].yaxis.set_visible(False)
        axd['trace'].imshow(trace.data, cmap='gray')

        # load ground truth signal
        label_signal, label_fields = helper_code.load_signals(
                                        os.path.join(wfdb_records_dir, records[i]))
        # load reconstructed signal - this is to ensure format is the same as ground truth
        output_signal, output_fields = helper_code.load_signals(
                                        os.path.join(reconstructed_signal_dir, records[i]))
        
        output_signal, output_fields, label_signal, label_fields = \
            eval_utils.match_signal_lengths(reconstructed_signal, output_fields, 
                                            label_signal, label_fields)
        label_signal = eval_utils.trim_label_signal(label_signal, 
                                                    label_fields["sig_name"], 
                                                    int(label_fields["sig_len"])/4, 
                                                    rhythm=['II'])
        
        # TODO calculate reconstruction SNR
        mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, \
            mean_weighted_absolute_difference_metric = eval_utils.single_signal_snr(output_signal,
                        output_fields, label_signal, label_fields, records[i], extra_scores=True)
        gridsize = '37.5 (hardcoded)'

        labels = None
        # optional: classify signal
        # labels = team_code.classify_signals(records[i], reconstructed_signal_dir, resnet_model, 
        #                                     dx_classes, verbose=True)
        
        description_string = f"""{records[i]} ({label_fields['fs']} Hz)
    Reconstruction SNR: {mean_snr:.2f}
    Gridsize: {gridsize}
    {label_fields['comments']}
    Predicted labels: {labels}"""
    
        # plot ground truth and reconstructed signal
        fig, axs = plt.subplots(reconstructed_signal.shape[1], 1, figsize=(9, 12.5), sharex=True)
        fig.subplots_adjust(hspace=0.1)
        fig.suptitle(description_string)
        # fig.text(0.5, 0.95, description_string, ha='center')
        for j in range(reconstructed_signal.shape[1]):
            if j == 0:
                axs[j].plot(reconstructed_signal[:, j], label='Reconstructed Signal')
                axs[j].plot(label_signal[:, j], label='Ground Truth Signal')
                axs[j].legend()
            else:
                axs[j].plot(reconstructed_signal[:, j])
                axs[j].plot(label_signal[:, j])
            axs[j].set_ylabel(f'{label_fields["sig_name"][j]}', rotation='horizontal')
            # axs[j].set_yrotation(0)
        fig.canvas.draw()
        axd['ecg_plots'].axis('off')
        axd['ecg_plots'].imshow(fig.canvas.renderer.buffer_rgba())
        # save everything in one image
        filename = f"{records[i]}_reconstruction.png"
        plt.figure(mosaic)
        plt.savefig(os.path.join(visualization_save_dir, filename))
        plt.close()


if __name__ == "__main__":
    test_images_folder = os.path.join("temp_data", "images")
    unet_outputs_folder = os.path.join("temp_data", "unet_outputs")
    # unet_outputs_folder = os.path.join("temp_data", "masks")
    reconstructed_signal_dir = os.path.join("temp_data", "reconstructed_signals")
    visualization_save_folder = os.path.join("evaluation", "data", "trace_visualizations")

    visualize_trace(test_images_folder, 
                    unet_outputs_folder, 
                    reconstructed_signal_dir,
                    wfdb_records_dir=test_images_folder,
                    visualization_save_dir=visualization_save_folder)