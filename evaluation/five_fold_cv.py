import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from utils import default_models


def train_reconstruction(data_folder, 
                         record_ids, 
                         model_folder, 
                         verbose, 
                         models_to_train=default_models.DIGITIZATION_MODELS, 
                         allow_failures=True):
    """
    Parameters:
        data_folder (str): The path to the foldder containing the data.
        record_ids (list): The list of record ids, e.g. ['00001_lr', ...]
        model_folder (str): The path to the folder where the models will be saved.
        verbose (bool): Printouts?
        models_to_train (list, default: "all"): A list of the models to train, used mainly for 
            modular testing. Allows the user to specify which models should be trained. Default 
            behaviour is to train all models listed in default_models. 

    """
    pass
