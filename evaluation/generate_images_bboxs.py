"""
Generates images and json files with bounding boxes for the images.
"""

import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import matplotlib.pyplot as plt
import numpy as np
import team_code, helper_code
from sklearn.utils import shuffle
import generator


def generate_data(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    records = shuffle(records, random_state=42)[8000:16000]
    num_records = len(records)
    train_records = records[:int(num_records*0.8)]
    val_records = records[int(num_records*0.8):int(num_records*0.9)]
    test_records = records[int(num_records*0.9):]

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    test_images_folder = os.path.join("temp_data", "yolo_images", "train")
    os.makedirs(test_images_folder, exist_ok=True)

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.calibration_pulse = 0.5
    img_gen_params.augment = True
    img_gen_params.random_print_header = 0.8
    img_gen_params.random_resolution = True
    img_gen_params.lead_bbox = True
    img_gen_params.lead_name_bbox = True
    img_gen_params.store_config = 1
    img_gen_params.input_directory = data_folder
    img_gen_params.output_directory = test_images_folder

    # generate train images
    if verbose:
        print("Generating images from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, train_records)

    # generate val images
    img_gen_params.output_directory = os.path.join("temp_data", "yolo_images", "val")
    os.makedirs(img_gen_params.output_directory, exist_ok=True)
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, val_records)

    # generate test images
    img_gen_params.output_directory = os.path.join("temp_data", "yolo_images", "test")
    os.makedirs(img_gen_params.output_directory, exist_ok=True)
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, test_records)


if __name__ == "__main__":
    data_folder = os.path.join("ptb-xl", "records500")
    model_folder = os.path.join("model")
    verbose = True

    generate_data(data_folder, model_folder, verbose)