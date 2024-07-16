import sys, os, numpy as np
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
import classification
from classification import seresnet18
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


def eval_resnet(data_folder, records, resnet_model, classes, verbose, max_samples=None):
    
    input_labels = []
    output_labels = []
    for r in records:
        data = [classification.get_testing_data(r, data_folder)] 
        pred_dx, probabilities = seresnet18.predict_proba(resnet_model, data, classes, verbose)
        labels = classes[np.where(pred_dx == 1)]
        if verbose:
            print(f"Classes: {classes}, probabilities: {probabilities}")
            print(f"Predicted labels: {labels}")

        actual_labels = helper_code.load_labels(os.path.join(data_folder, r))
        print('Actual labels', actual_labels)
        
        actual_labels_bin = np.zeros(len(classes))
        for l in actual_labels:
            if l in classes:
                index = np.where(classes == l)[0]
                actual_labels_bin[index] = 1
        print(actual_labels_bin, pred_dx)
        input_labels.append(actual_labels_bin)
        output_labels.append(pred_dx)

    f_measure, _, _ = helper_code.compute_f_measure(input_labels, output_labels)
    print('F-measure = ', f_measure)


def main(data_folder, model_folder, verbose, max_samples=None):
    """
    Team code version
    """
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    train_records, val_records = split_data(data_folder, tts=0.8, max_samples=max_samples)
    print(f"Num of training records: {len(train_records)} // Num of val records: {len(val_records)}")

    # train classification model
    resnet_model, uniq_labels = team_code.train_classification_model(
        data_folder, verbose, records_to_process=train_records)

    # save trained classification model
    unet_model = None
    team_code.save_models(model_folder, unet_model, resnet_model, uniq_labels)

    # test model
    eval_resnet(data_folder, val_records, resnet_model, uniq_labels, verbose, max_samples)


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "ptb-xl", "records100")
    model_folder = os.path.join(os.getcwd(), "model")
    verbose = True
    max_samples = 20 # limit n_samples for fast testing, set to None to use all samples

    main(data_folder, model_folder, verbose, max_samples)
