import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import joblib, time
from tqdm import tqdm
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
import generator


def generate_test_images(wfdb_records_folder, images_folder):
    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.print_header = True
    img_gen_params.augment = True # TODO reconstruction breaks with any rotation at the moment
    img_gen_params.rotate = 5
    img_gen_params.input_directory = wfdb_records_folder
    img_gen_params.output_directory = images_folder
    

def run_models(record, digitization_model, classification_model, verbose):
    # TODO
    # Load the record and header file

    # preprocess the image file

    # run u-net on preprocessed image

    # reconstruct signal

    # run resnet on reconstructed signal and return predicted class
    pass


if __name__ == "__main__":   
    data_folder = "G:\\PhysionetChallenge2024\\ptb-xl\\combined_records"
    model_folder = "G:\\PhysionetChallenge2024\\model"
    verbose = True 
    digitization_model, classification_model = team_code.load_models(model_folder, verbose)