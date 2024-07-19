import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
from utils import model_persistence


def split_data(data_folder, tts=0.8, max_samples=None):
    """
    Split the data into training and validation sets
    """
    records = helper_code.find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    # optionally test on a smaller number of records
    if max_samples is not None: 
        records = shuffle(records, random_state=42)[:max_samples]
        num_records = len(records)
    
    records = shuffle(records)
    train_records = records[:int(tts*num_records)]
    val_records = records[int(tts*num_records):]

    return train_records, val_records


def eval_resnet(data_folder, model_folder, verbose, max_samples=None):
    pass


def main(data_folder, model_folder, verbose, max_samples=None):
    """
    Team code version
    """
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    train_records, val_records = split_data(data_folder, tts=0.8, max_samples=max_samples)

    # train classification model
    resnet_model, uniq_labels = team_code.train_classification_model(
        data_folder, verbose, records_to_process=train_records)

    # save trained classification model
    unet_model = None
    team_code.save_models(model_folder, unet_model, resnet_model, uniq_labels)


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "ptb-xl", "records500")
    model_folder = os.path.join(os.getcwd(), "model")
    verbose = True
    max_samples = 20 # limit n_samples for fast testing, set to None to use all samples

    main(data_folder, model_folder, verbose, max_samples)
