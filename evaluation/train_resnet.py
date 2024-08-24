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
from classification.utils import multiclass_predict_from_logits


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
    y_pred = np.zeros((len(records), len(resnet_model), len(classes)))
    y_true = np.zeros((len(records), len(classes)))
    outputs = []
    targets = []
    for j, r in enumerate(records):
        # check to make sure we have labels
        target = helper_code.load_labels(os.path.join(data_folder, r))
        
        if target and not '' in target:
            y_true[j] = helper_code.compute_one_hot_encoding(target, classes)
            targets.append(target)
        else: continue

        data = [classification.get_testing_data(r, data_folder)]
        for i, val in enumerate(resnet_model):
            _, y_pred[j, i] = seresnet18.predict_proba(
                                                resnet_model[val], data, classes, verbose)
        
        # hacky way to get the mean of all the resnets
        probs = np.mean(y_pred[j], axis=0)
        
        pred_dx = multiclass_predict_from_logits(classes, probs)
        outputs.append(classes[np.where(pred_dx == 1)])

    with open('output_probabilities.pkl', 'wb') as f:
        pickle.dump(y_pred, f)

    with open('target_labels.pkl', 'wb') as f:
        pickle.dump(y_true, f)

    f_measure, _, _ = helper_code.compute_f_measure(targets, outputs)
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
#     resnet_model, uniq_labels = team_code.train_classification_model(
#         data_folder, verbose, records_to_process=train_records)
# :
#     unet_model = None
#     team_code.save_models(model_folder, unet_model, resnet_model, uniq_labels)

    resnet_models = model_persistence.load_models(model_folder, verbose, 
                        models_to_load=[ 
                                        'dx_classes', 
                                        'res0', 'res1', 'res2', 'res3', 'res4'
                                        ])
    uniq_labels = resnet_models.pop('dx_classes')

    # test model
    eval_resnet(data_folder, val_records, resnet_models, uniq_labels, verbose, max_samples)


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
