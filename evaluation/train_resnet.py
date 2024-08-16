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
        records = shuffle(records, random_state=42)[:max_samples]
        num_records = len(records)
    
    train_records, val_records = train_test_split(records, shuffle=True, train_size=tts, random_state=2024)
   # records = shuffle(records)
   # train_records = records[:int(tts*num_records)]
    #val_records = records[int(tts*num_records):]

    return train_records, val_records


def eval_resnet(data_folder, records, resnet_model, classes, verbose, max_samples=None):

    input_labels = []
    output_labels = []
    for r in records:
        data = [classification.get_testing_data(r, data_folder)] 
        pred_dx, probabilities = seresnet18.predict_proba(resnet_model, data, classes, verbose)
        labels = classes[np.where(pred_dx == 1)]
        # if verbose:
        #     print(f"Classes: {classes}, probabilities: {probabilities}")
        #     print(f"Predicted labels: {labels}")

        actual_labels = helper_code.load_labels(os.path.join(data_folder, r))
        
        if actual_labels and not '' in actual_labels:
            input_labels.append(actual_labels)
            output_labels.append(labels)

    """
    with open('input_labels.pkl', 'wb') as f:
        pickle.dump(input_labels, f)

    with open('output_labels.pkl', 'wb') as f:
        pickle.dump(output_labels, f)
    """

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
    
    records = shuffle(records, random_state=42)
    train_records = records[:int(0.9*num_records)]
    val_records = records[int(0.9*num_records):]

    # train_records = train_records[:200]
    # val_records = val_records[:100]

    # train classification model
    resnet_model, uniq_labels = team_code.train_classification_model(
        data_folder, verbose, records_to_process=train_records)

    # save trained classification model
    try:
        model = model_persistence.load_models("model", True, 
                            models_to_load=['digitization_model'])
        unet_model = model['digitization_model']
    except:
        unet_model = None
    team_code.save_models(model_folder, unet_model, resnet_model, uniq_labels)

    # test model
    eval_resnet(data_folder, val_records, resnet_model, uniq_labels, verbose, max_samples)


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "ptb-xl", "records500")
    model_folder = os.path.join(os.getcwd(), "model")
    os.makedirs(model_folder, exist_ok=True)
    verbose = True
    max_samples = None # limit n_samples for fast testing, set to None to use all samples

    main(data_folder, model_folder, verbose, max_samples)

    """
    with open('test_files.pkl', 'rb') as f:
        records = pickle.load(f)

    models = model_persistence.load_models(model_folder, verbose, models_to_load=['classification_model', 'dx_classes'])
    
    resnet_model = models['classification_model']
    classes = models['dx_classes']
    
    eval_resnet(data_folder, records, resnet_model, classes, verbose)
    """
