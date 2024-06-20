import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

from sklearn.utils import shuffle
import team_code, helper_code, prepare_image_data
import generator
from utils import team_helper_code


class PrepareImArgs():
    def __init__(self, input_folder, output_folder) -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder


def visualize_preprocessing(images_folder, processed_image_folder, visualization_folder, verbose,
                            n_gen_data=0, data_folder=None):
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(processed_image_folder, exist_ok=True)

    if n_gen_data > 0:
        # Find the data files.
        if verbose:
            print('Finding the Challenge data...')

        records = helper_code.find_records(data_folder)
        num_records = len(records)
        # test on set number of records, set random_state for consistency if needed
        records = shuffle(records, random_state=None)[:n_gen_data] 

        if num_records == 0:
            raise FileNotFoundError('No data were provided.')

        # params for generating images
        img_gen_params = generator.DefaultArgs()
        img_gen_params.random_bw = 0.2
        img_gen_params.wrinkles = True
        img_gen_params.print_header = 0.8
        img_gen_params.augment = True
        img_gen_params.rotate = 5
        # img_gen_params.seed = 42 # uncomment for reproducibility
        img_gen_params.input_directory = data_folder
        img_gen_params.output_directory = images_folder

        # generate images
        if verbose:
            print("Generating images from wfdb files...")
        generator.gen_ecg_images_from_data_batch.run(img_gen_params, records)

        # add image names to header files for generated images
        if verbose:
            print("Adding image filenames to headers...")
        prepare_args = PrepareImArgs(images_folder, images_folder)
        prepare_image_data.run(prepare_args)

    # preprocess images. Make any edits to the function in team_code.py as needed
    records = helper_code.find_records(images_folder)
    team_code.preprocess_images(images_folder, processed_image_folder, verbose, 
                                records_to_process=records)

    # visualize the preprocessed images
    if verbose:
        print("Generating and saving visualizations...")
    # processed_records = helper_code.find_records(processed_image_folder)
    # check that we have both images and processed images for each ID
    ids = team_helper_code.check_dirs_for_ids(records, images_folder, 
                                              processed_image_folder, verbose)
    
    for record in ids:
        original_image_name = team_helper_code.find_available_images(
                            [record], images_folder, verbose)[0]
        original_image_path = os.path.join(images_folder, original_image_name + '.png')
        processed_image_name = team_helper_code.find_available_images(
                            [record], processed_image_folder, verbose)[0]  
        processed_image_path = os.path.join(processed_image_folder, processed_image_name + '.png')
        with open(original_image_path, 'rb') as f:
            original_image = plt.imread(f)
        with open(processed_image_path, 'rb') as f:
            processed_image = plt.imread(f)

        # load saved header file after processing, in case we have saved rotation
        # angle or grid size there
        record_path = os.path.join(processed_image_folder, record) 
        header_txt = helper_code.load_header(record_path)
        try:
            grid_size = team_helper_code.get_gridsize_from_header(header_txt)
            header_str2 = f"Grid size: {grid_size}"
        except:
            header_str2 = "No grid size found in header"

        # show images in two columns
        fig = plt.figure(layout="tight")
        rows, cols = 1, 2

        # plot first image
        ax = fig.add_subplot(rows, cols, 1)
        ax.imshow(original_image, cmap='gray')
        ax.set_title(f"Original image: {record}")
        ax.axis('off')

        # plot second image
        ax = fig.add_subplot(rows, cols, 2)
        ax.imshow(processed_image, cmap='gray')
        ax.set_title(header_str2)
        ax.axis('off')

        plt.savefig(os.path.join(visualization_folder, record + '_comparison.png'))
        plt.close()


if __name__ == "__main__":
    # change folder paths as needed
    image_folder = os.path.join("evaluation", "data", "images")
    processed_image_folder = os.path.join("evaluation", "data", "processed_images")
    visualization_folder = os.path.join("evaluation", "data", "preprocessing_visualizations")
    verbose = True
    num_images_to_generate = 0 # int, set to 0 if data has already been generated to speed up testing time
    data_folder = os.path.join("ptb-xl", "records500") # can set to None if num_images_to_generate = 0
    
    visualize_preprocessing(image_folder, processed_image_folder, visualization_folder, verbose, 
                            n_gen_data=num_images_to_generate, data_folder=data_folder)