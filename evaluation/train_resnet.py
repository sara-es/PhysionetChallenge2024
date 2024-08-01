import sys, os, numpy as np, pickle
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
import classification
from classification import seresnet18
from utils import model_persistence
from sklearn.model_selection import train_test_split


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
        records = shuffle(records, random_state=2024)[:max_samples]
        num_records = len(records)
    
    train_records, val_records = train_test_split(records, shuffle=True, train_size=tts, random_state=2024)

    return train_records, val_records

def run_models(record, classification_model, verbose):
    
    ###########################
    ### FROM TEAM_CODE.RUN_MODELS() including only the parts for the resnet model(s)

    # If ´classification_model´ is a list, there is multiple models trained and one dictionary for dx_classes
    if isinstance(classification_model, list):
        dx_classes = [d['dx_classes'] for d in classification_model if 'dx_classes' in d][0]
        classification_model = [d for d in classification_model if 'dx_classes' not in d]
    else:
        dx_classes = classification_model['dx_classes']
        classification_model = classification_model['classification_model']


    # Run the classification model; if you did not train this model, then you can set labels=None.
    labels = team_code.classify_signal(record, data_folder, classification_model, dx_classes, verbose=verbose)

    ########

    return labels

def run(records, classification_model, data_folder, verbose):

    num_records = len(records)

    if num_records==0:
        raise Exception('No data were provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    # Run the team's models on the Challenge data.
    if verbose:
        print('Running the Challenge model(s) on the Challenge data...')

    output_labels = []
    input_labels = []
    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        data_record = os.path.join(data_folder, records[i])
        labels_tmp = run_models(data_record, classification_model, verbose)
        output_labels.append(labels_tmp)

        actual_labels_tmp = helper_code.load_labels(os.path.join(data_folder, records[i]))
        input_labels.append(actual_labels_tmp)

    f_measure, _, _ = helper_code.compute_f_measure(input_labels, output_labels)
    print('F-measure = ', f_measure)

def main(data_folder, model_folder, verbose, max_samples=None):
    """
    Team code version
    """
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')
    records = helper_code.find_records(data_folder)
    num_records = len(records)
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    train_records, val_records = split_data(data_folder)

    # train classification model
    resnet_model, uniq_labels = team_code.train_classification_model(
        data_folder, model_folder, verbose, records_to_process=train_records)

    # save trained classification model(s) and also test to load the model(s)
    try:
        model = model_persistence.load_models("model", True, 
                            models_to_load=['digitization_model'])
        unet_model = model['digitization_model']
    except:
        unet_model = None

    team_code.save_models(model_folder, unet_model, resnet_model, uniq_labels)
    digitization_model, classification_model = team_code.load_models(model_folder, verbose)

    # run model(s)
    run(val_records, classification_model, data_folder, verbose)


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "tiny_testset", "hr_gt")
    model_folder = os.path.join(os.getcwd(), "test_model")
    output_folder = os.path.join(os.getcwd(), "tiny_testset", 'test_outputs')
    verbose = True
    max_samples = 10 # limit n_samples for fast testing, set to None to use all samples

    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    main(data_folder, model_folder, verbose, max_samples)

    """
    with open('test_files.pkl', 'rb') as f:
        records = pickle.load(f)

    models = model_persistence.load_models(model_folder, verbose, models_to_load=['classification_model', 'dx_classes'])
    
    resnet_model = models['classification_model']
    classes = models['dx_classes']
    
    eval_resnet(data_folder, records, resnet_model, classes, verbose)
    """
