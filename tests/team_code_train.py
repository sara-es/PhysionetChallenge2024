import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
from utils import model_persistence


def train_models(data_folder, model_folder, verbose):
    """
    Team code version
    To quickly test that a function is working, comment out all irrelevant code. 
    If the necessary data has already been generated (in tiny_testset, for example),
    everything should run independently.
    """
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)[:2000]
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    # test on a smaller number of records for now
    records = shuffle(records)
    num_records = len(records)
    
    digitization_model = team_code.train_digitization_model(data_folder, model_folder, verbose, 
                                records_to_process=records, delete_training_data=False)



if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "ptb-xl", "records500")
    # data_folder = os.path.join("temp_data", "train", "images")
    model_folder = os.path.join(os.getcwd(), "model")
    verbose = True

    train_models(data_folder, model_folder, verbose)
