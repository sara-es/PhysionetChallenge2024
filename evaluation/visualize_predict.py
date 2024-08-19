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
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json

import team_code, helper_code
from evaluation import eval_utils
from utils import model_persistence, team_helper_code
from digitization.ECGminer.assets.Image import Image as ECGImage
from digitization.ECGminer.assets.Format import Format
from digitization.ECGminer.assets.Lead import Lead
from digitization.ECGminer.assets.Point import Point
from digitization.ECGminer.assets.Rectangle import Rectangle
from digitization import Unet
from preprocessing import classifier, column_rotation
import digitization.YOLOv7

from digitization.Unet import Unet
import torch
from digitization.Unet.ECGunet import BasicResUNet


def get_trace(image, raw_signals, bounds, rhythm_leads=[Lead.II]):
    """
    Get the trace of the signal from the u-net output image.
    """
    """
    Get the trace of the extraction algorithm performed over the ECG image.

    Args:
        ecg (Image): ECG image.
        signals (Iterable[Iterable[Point]]): List with the points of each of
            the signals of each lead.
        ref_pulses (Iterable[Tuple[int, int]]): List with the reference pulses
            of each ECG row.

    Returns:
        Image: ECG image with the trace painted over.
    """
    NROWS, NCOLS = (3,4)
    ORDER = Format.STANDARD
    COLORS = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 200, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 0, 125),
        (0, 125, 0),
        (125, 0, 0),
        (0, 100, 125),
        (125, 125, 0),
        (125, 0, 125),
    ]
    H_SPACE = 20

    # Invert the colours
    trace = abs(image - 1)*255
    # convert greyscale to rgb
    trace = cv2.merge([trace,trace,trace])
    trace = np.uint8(trace)
    trace = ECGImage(trace)
    # replicate crop that happens in preprocessing
    # rect = Rectangle(Point(0, 350), Point(image.shape[1], image.shape[0])) #set it to image size
    trace.crop(bounds)
    trace.to_BGR()

    # Draw signals
    for i, lead in enumerate(ORDER):
        rhythm = lead in rhythm_leads
        r = rhythm_leads.index(lead) + NROWS if rhythm else i % NROWS
        c = 0 if rhythm else i // NROWS
        signal = raw_signals[r]
        obs_num = len(signal) // (1 if rhythm else NCOLS)
        signal = signal[c * obs_num : (c + 1) * obs_num]
        color = COLORS[i % len(COLORS)]
        for p1, p2 in zip(signal, signal[1:]):
            trace.line(p1, p2, color, thickness=2)
    return trace


def visualize_trace(test_images_dir, unet_outputs_dir, reconstructed_signal_dir,
                    digitization_model,
                    wfdb_records_dir, visualization_save_dir, save_images):
    records = helper_code.find_records(wfdb_records_dir)
    ids = team_helper_code.check_dirs_for_ids(records, test_images_dir, 
                                              None, True)
    image_ids = team_helper_code.find_available_images(ids, 
                                                       test_images_dir, verbose=True)
    # unet_ids = team_helper_code.find_available_images(ids, 
    #                                                   unet_outputs_dir, verbose=True)
    # if len(image_ids) != len(unet_ids) and len(image_ids) > 0:
    #     print(image_ids, unet_ids)
    #     raise ValueError("Number of image and U-Net output files do not match, please make "+\
    #                      "sure both have been generated and saved correctly.")
    
    image_ids = sorted(list(image_ids), reverse=True)
    ids = sorted(list(ids), reverse=True)
    # unet_ids = sorted(list(unet_ids))

    # load resnet for classification
    yolo_model = digitization_model['yolov7-ecg-2c']
    unet_generated = digitization_model['unet_generated']
    unet_real = digitization_model['unet_real']
    classification_model = digitization_model['image_classifier']

    # force load unet from checkpoint ###############
    # models = model_persistence.load_models("model", True, 
    #                     models_to_load=['digitization_model'])
    # unet_generated = Unet.utils.load_unet_from_state_dict(models['digitization_model'])
    #############################

    patch_folder = os.path.join('test_data', 'patches', 'image_patches')
    os.makedirs(patch_folder, exist_ok=True)

    snrs = []  
    stats = []
    
    for i in range(len(ids[:200])): # care, seems to crash after plotting ~200 or so - I blame tkinter
        print(image_ids[i])
        image_info = {}
        image_info["record"] = ids[i]
        image_path = os.path.join(test_images_dir, image_ids[i] + '.png')

        # get bounding boxes/ROIs using YOLOv7
        args = digitization.YOLOv7.detect.OptArgs()
        args.device = "0"
        args.source = image_path
        args.project = "test_data\\detect"
        args.nosave = False # set False for testing to save images with ROIs
        rois = digitization.YOLOv7.detect.detect_single(yolo_model, args, verbose=True)

        # resize image if it's very large
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            if image.size[0] > 2300: # index 0 is width
                ratio = 2300/image.size[0]
                new_height = int(ratio*image.size[1])
                image = image.resize((2200, new_height), Image.Resampling.LANCZOS)
            image = np.array(image)

        # patchify image
        Unet.patching.save_patches_single_image(ids[i], image, None, 
                                                patch_size=256,
                                                im_patch_save_path=patch_folder,
                                                lab_patch_save_path=None)

        # run classifier
        is_real_image = False
        # is_real_image = classifier.classify_image(ids[i], patch_folder, 
        #                                             classification_model, True)
        if is_real_image:
            print("Detected real image")
        
        if is_real_image:
            unet_model = unet_real
        else:
            unet_model = unet_generated
        # predict on patches, recover u-net output image
        unet_image = Unet.predict_single_image(ids[i], patch_folder, unet_model,
                                                original_image_size=image.shape[:2])
        
        with open(os.path.join(unet_outputs_dir, ids[i] + '.png'), 'wb') as f:
            plt.imsave(f, unet_image, cmap='gray')

        # digitize signal from u-net ouput
        record_path = os.path.join(wfdb_records_dir, ids[i]) 
        header_txt = helper_code.load_header(record_path)
        # rotate reconstructed u-net output to original orientation
        rotated_mask, rotated_image_path, rot_angle = column_rotation(ids[i], 
                                                        unet_image, image,
                                                        angle_range=(-20, 20), verbose=True)
        
        if rot_angle != 0: # currently just rotate the mask, do no re-predict 
            print(f"Rotation angle detected: {rot_angle}")
        #     try: # sometimes this fails, if there are edge effects
        #         args.source = rotated_image_path
        #         rois = digitization.YOLOv7.detect.detect_single(yolo_model, args, True)
        #         reconstructed_signal, raw_signals, gridsize = team_code.reconstruct_signal(ids[i], rotated_mask, 
        #                                                 rois,
        #                                                 header_txt,
        #                                                 reconstructed_signal_dir, 
        #                                                 save_signal=True,
        #                                              is_real_image=is_real_image)
        #         unet_image = rotated_mask # to save later, optional
        #     except Exception as e: # in that case try it with the original (non-rotated) mask
        #         print(f"Error reconstructing signal after rotating image {image_path}: {e}")
        #         reconstructed_signal, raw_signals, gridsize = team_code.reconstruct_signal(ids[i], unet_image,
        #                                                 rois, 
        #                                                 header_txt,
        #                                                 reconstructed_signal_dir, 
        #                                                 save_signal=True,
        #                                              is_real_image=is_real_image)        
        # else: # no rotation needed
        reconstructed_signal, raw_signals, gridsize = team_code.reconstruct_signal(ids[i], unet_image, 
                                                    rois,
                                                    header_txt,
                                                    reconstructed_signal_dir, 
                                                    save_signal=True,
                                                    is_real_image=is_real_image)
        
        if reconstructed_signal is None:
            print(f"Failed to reconstruct signal {ids[i]}")
            reconstructed_signal = np.zeros((5000, 12))

        # load reconstructed signal - this is to ensure format is the same as ground truth
        output_signal, output_fields = helper_code.load_signals(
                                        os.path.join(reconstructed_signal_dir, ids[i]))
        
        # load ground truth signal
        try:
            label_signal, label_fields = helper_code.load_signals(
                                            os.path.join(wfdb_records_dir, ids[i]))
        except:
            label_signal, label_fields = np.zeros_like(output_signal), None
        
        if label_fields is not None:
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
                            output_fields, label_signal, label_fields, ids[i], extra_scores=True)

            # load image generation info from json
            # config_file = os.path.join(test_images_dir, image_ids[i] + '.json')
            # with open(config_file, 'r') as f:
            #     config = json.load(f)
            true_gridsize = '39' # config['gridsize']
            dc_pulse = 'N/A' # config['dc_pulse']
        else:
            label_signal = np.zeros_like(output_signal)
            mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, \
                mean_weighted_absolute_difference_metric = 0, 0, 0, 0, 0
            true_gridsize = 'N/A'
            dc_pulse = 'N/A'

        gridsize = f'{gridsize:.2f}' if gridsize is not None else 'N/A'
        
        if mean_snr < save_image_threshold:
            # set up mosaic for original image, u-net output with trace, ground truth signal, 
            # and reconstructed signal
            mosaic = plt.figure(layout="tight", figsize=(18, 13))
            axd = mosaic.subplot_mosaic([
                                            ['original_image', 'ecg_plots'],
                                            ['trace', 'ecg_plots']
                                        ])
            # load image from YOLO detection
            yolo_img_path = os.path.join("test_data", "detect", "exp", image_ids[i]+'.png')
            with Image.open(yolo_img_path) as img:
                axd['original_image'].axis('off')
                axd['original_image'].imshow(img, cmap='gray')

            # plot trace of signal from u-net output
            axd['trace'].xaxis.set_visible(False)
            axd['trace'].yaxis.set_visible(False)
            if raw_signals is not None:
                raw_signal = raw_signals['raw_signals']
                bounds = raw_signals['bounding_rect']
                trace = get_trace(unet_image, raw_signal, bounds)
                axd['trace'].imshow(trace.data, cmap='gray')
            else:
                axd['trace'].imshow(unet_image, cmap='gray')

            labels = None
            # optional: classify signal
            # labels = team_code.classify_signals(records[i], reconstructed_signal_dir, resnet_model, 
            #                                     dx_classes, verbose=True)
            
            description_string = f"""{ids[i]} ({output_fields['fs']} Hz)
    Reconstruction SNR: {mean_snr:.2f}
    Gridsize: {gridsize}
    {output_fields['comments'][1:-1]}
    Predicted real? {is_real_image}"""
        
            # plot ground truth and reconstructed signal
            fig, axs = plt.subplots(output_signal.shape[1], 1, figsize=(9, 12.5), sharex=True)
            fig.subplots_adjust(hspace=0.1)
            fig.suptitle(description_string)
            # fig.text(0.5, 0.95, description_string, ha='center')
            for j in range(output_signal.shape[1]):
                if j == 0:
                    axs[j].plot(label_signal[:, j], label='Ground Truth Signal')
                    axs[j].plot(output_signal[:, j], label='Reconstructed Signal', alpha=0.8)
                    axs[j].legend(loc='upper right')
                else:
                    axs[j].plot(label_signal[:, j])
                    axs[j].plot(output_signal[:, j], alpha=0.8)
                axs[j].set_ylabel(f'{label_fields["sig_name"][j]}', rotation='horizontal')
                # axs[j].set_yrotation(0)
            fig.canvas.draw()
            axd['ecg_plots'].axis('off')
            axd['ecg_plots'].imshow(fig.canvas.renderer.buffer_rgba())

            # save everything in one image
            filename = f"{ids[i]}_reconstruction.png"
            plt.figure(mosaic)
            plt.savefig(os.path.join(visualization_save_dir, filename))
            plt.close()

        image_info["snr"] = mean_snr
        image_info["snr_median"] = mean_snr_median
        image_info["mean_ks_metric"] = mean_ks_metric
        image_info["mean_asci_metric"] = mean_asci_metric
        image_info["mean_weighted_absolute_difference_metric"] = mean_weighted_absolute_difference_metric
        image_info["estimated_gridsize"] = gridsize
        image_info["actual_gridsize"] = true_gridsize
        image_info["reference_pulse"] = dc_pulse

        snrs.append(mean_snr)
        stats.append(image_info)
        if mean_snr < 3.5:
            print(f"Low SNR for {ids[i]}: {mean_snr:.2f}")

    print(f"Average SNR: {np.mean(snrs):.2f}")
    df = pd.DataFrame(stats)
    os.makedirs(os.path.join("evaluation", "data"), exist_ok=True)
    df.to_csv(os.path.join("evaluation", "data", "snr.csv"), index=False)


if __name__ == "__main__":
    test_images_folder = os.path.join("test_data", "images")
    # unet_outputs_folder = os.path.join("test_data", "unet_outputs")
    unet_outputs_folder = os.path.join("test_data", "unet_outputs")
    os.makedirs(unet_outputs_folder, exist_ok=True)
    reconstructed_signal_dir = os.path.join("test_data", "reconstructed_signals")
    os.makedirs(reconstructed_signal_dir, exist_ok=True)
    visualization_save_folder = os.path.join("test_data", "trace_visualizations")
    os.makedirs(visualization_save_folder, exist_ok=True)
    save_image_threshold = 30 # snr threshold below which images will be saved for visualization

    model_folder = 'model'
    digitization_model = dict()
    models = model_persistence.load_models(model_folder, True, 
                        models_to_load=['yolov7-ecg-2c', 
                                        'unet_generated',
                                        'unet_real',
                                        'image_classifier', 
                                        ])
    digitization_model['unet_generated'] = Unet.utils.load_unet_from_state_dict(
                                                        models['unet_generated'])
    digitization_model['unet_real'] = Unet.utils.load_unet_from_state_dict(models['unet_real'])
    digitization_model['image_classifier'] = classifier.load_from_state_dict(
                                                    models['image_classifier'])
    digitization_model['yolov7-ecg-2c'] = models['yolov7-ecg-2c']

    visualize_trace(test_images_folder, 
                    unet_outputs_folder, 
                    reconstructed_signal_dir,
                    digitization_model,
                    wfdb_records_dir=test_images_folder,
                    visualization_save_dir=visualization_save_folder,
                    save_images=save_image_threshold)