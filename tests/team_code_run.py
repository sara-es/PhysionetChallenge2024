import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import joblib, time
from tqdm import tqdm
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet


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