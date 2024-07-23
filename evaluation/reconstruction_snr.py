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
import pandas as pd

import team_code, helper_code
from evaluation import eval_utils
from utils import model_persistence, team_helper_code


def eval_reconstruction(test_images_dir, unet_outputs_dir, reconstructed_signal_dir,
                    wfdb_records_dir, visualization_save_dir):
    records = helper_code.find_records(wfdb_records_dir)
    ids = team_helper_code.check_dirs_for_ids(records, test_images_dir, 
                                              unet_outputs_dir, True)
    image_ids = team_helper_code.find_available_images(ids, 
                                                       test_images_dir, verbose=True)
    unet_ids = team_helper_code.find_available_images(ids, 
                                                      unet_outputs_dir, verbose=True)
    # if len(image_ids) != len(unet_ids) and len(image_ids) > 0:
    #     print(image_ids, unet_ids)
    #     raise ValueError("Number of image and U-Net output files do not match, please make "+\
    #                      "sure both have been generated and saved correctly.")
    
    image_ids = sorted(list(image_ids))
    unet_ids = sorted(list(unet_ids))

    # load resnet for classification
    # classification_model = model_persistence.load_model('model', 'classification_model')
    # resnet_model = classification_model["model"]
    # dx_classes = classification_model["dx_classes"]

    snrs = []  
    # snrs_df = pd.DataFrame(columns=["record", "mask_path", "snr", "frequency", "estimated_gridsize",
    #                                 # "ground_truth_label", "predicted_label", 
    #                                 "snr_median", "mean_ks_metric", 
    #                                 "mean_asci_metric", "mean_weighted_absolute_difference_metric"])
    stats = []
    
    for i in range(len(image_ids)):
        # print(image_ids[i])
        image_info = {}
        image_info["record"] = records[i]

        # load u-net output
        unet_image_path = os.path.join(unet_outputs_dir, unet_ids[i] + '.npy')
        image_info["unet_output_path"] = unet_image_path
        with open(unet_image_path, 'rb') as f:
            unet_image = np.load(f)
        # digitize signal from u-net ouput
        record_path = os.path.join(wfdb_records_dir, records[i]) 
        header_txt = helper_code.load_header(record_path)
        reconstructed_signal, raw_signals, gridsize = team_code.reconstruct_signal(records[i], unet_image, 
                                 header_txt, reconstructed_signal_dir, save_signal=True)

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
        
        # calculate reconstruction SNR
        mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, \
            mean_weighted_absolute_difference_metric = eval_utils.single_signal_snr(output_signal,
                        output_fields, label_signal, label_fields, records[i], extra_scores=True)
        gridsize = f'{gridsize:.2f}'

        image_info["snr"] = mean_snr
        image_info["snr_median"] = mean_snr_median
        image_info["mean_ks_metric"] = mean_ks_metric
        image_info["mean_asci_metric"] = mean_asci_metric
        image_info["mean_weighted_absolute_difference_metric"] = mean_weighted_absolute_difference_metric
        image_info["estimated_gridsize"] = gridsize

        labels = None
        # optional: classify signal
        # labels = team_code.classify_signals(records[i], reconstructed_signal_dir, resnet_model, 
        #                                     dx_classes, verbose=True)

        snrs.append(mean_snr)
        stats.append(image_info)
        
        if mean_snr < 3.5:
            print(f"Low SNR for {records[i]}: {mean_snr:.2f}")

    print(f"Average SNR: {np.mean(snrs):.2f}")
    df = pd.DataFrame(stats)
    os.makedirs(os.path.join("evaluation", "data"), exist_ok=True)
    df.to_csv(os.path.join("evaluation", "data", "snr_old_line_trace_sliding_window_control.csv"), index=False)


if __name__ == "__main__":
    test_images_folder = os.path.join("temp_data", "images")
    # unet_outputs_folder = os.path.join("test_data", "unet_outputs")
    unet_outputs_folder = os.path.join("temp_data", "masks")
    reconstructed_signal_dir = os.path.join("temp_data", "reconstructed_signals")
    os.makedirs(reconstructed_signal_dir, exist_ok=True)
    visualization_save_folder = os.path.join("evaluation", "trace_visualizations")
    os.makedirs(visualization_save_folder, exist_ok=True)

    eval_reconstruction(test_images_folder, 
                    unet_outputs_folder, 
                    reconstructed_signal_dir,
                    wfdb_records_dir=test_images_folder,
                    visualization_save_dir=visualization_save_folder)