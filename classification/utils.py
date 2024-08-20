import os
import numpy as np
import helper_code

def get_demographic_features(record):
    
    '''
    age_gender = [age, age_flag, gender_female, gender_male, gender_flag]
    Note: features are a bit strange (-1 when no age, but 0s for sex?)
    '''
    age_gender = np.zeros(5)
    header = helper_code.load_header(record)
    
    age, has_age = helper_code.get_variable(header, '# Age:')
    #height, has_height = helper_code.get_variable(header, 'Height')
    #weight, has_weight = helper_code.get_variable(header, 'Weight')
    sex, has_sex = helper_code.get_variable(header, '# Sex:')

    if has_age:
        age_gender[0] = int(age)/100.
        age_gender[1] = 1
    else:
        age_gender[0] = -1
    
    if has_sex:
        if sex == 'Female':
            age_gender[4] = 1
            age_gender[2] = 1
        elif sex == 'Male':
            age_gender[4] = 1
            age_gender[3] = 1

    return age_gender


def get_training_data(record, data_folder):
    # get headers for labels and demographic info
    record_path = os.path.join(data_folder, record) 
    header_txt = helper_code.load_header(record_path)
    labels = helper_code.load_labels(record_path) # If no labels, returns ['']
    
    if labels:
        if not '' in labels: # only process records with labels for training
            # get demographic info
            age_gender = get_demographic_features(record_path)
            fs = helper_code.get_sampling_frequency(header_txt)
        else:
            return None, None
    else:
        return None, None
    
    data = [record_path, fs, age_gender]

    return data, labels


def get_testing_data(record, data_folder):
    record_path = os.path.join(data_folder, record)
    header_txt = helper_code.load_header(record_path)
    fs = helper_code.get_sampling_frequency(header_txt)
    age_gender = get_demographic_features(record_path)
    data = [record_path, fs, age_gender]
    return data


# Convert torch to numpy and encode the logits
def preprocess_labels(true_labels, pre_logits, threshold):

    true_labels = true_labels.cpu().detach().numpy().astype(np.int32)
    pre_logits = pre_logits.cpu().detach().numpy().astype(np.float32)

    # == Convert logits to binary to compute F-measure ==
    
    pre_binary = np.zeros(pre_logits.shape, dtype=np.int32)

    # Find the index of the maximum value within the logits
    likeliest_dx = np.argmax(pre_logits, axis=1)

    # First, add the most likeliest diagnosis to the predicted label
    pre_binary[np.arange(true_labels.shape[0]), likeliest_dx] = 1

    # Then, add all the others that are above the decision threshold
    other_dx = pre_logits >= threshold

    pre_binary = pre_binary + other_dx
    pre_binary[pre_binary > 1.1] = 1
    pre_binary = np.squeeze(pre_binary) 

    return true_labels, pre_logits, pre_binary


def compute_classification_metrics(actual_labels, pre_logits, unique_labels, threshold):
    """
    Parameters
        actual_labels (torch.Tensor): The true labels, one hot encoded
        pre_logits (torch.Tensor): The predicted logits (probabilities), one hot encoded
        unique_labels (list): unique labels in string format
        threshold (float): The decision threshold for additional (multilabel) predicted labels

    Returns
        f_measure (float)
    """
    actual_labels, pre_logits, pre_binary = preprocess_labels(actual_labels, pre_logits, threshold)

    # Compute the metrics
    f_measure, _, _ = compute_f_measure_from_onehot(actual_labels, pre_binary, unique_labels)

    return f_measure


def multiclass_predict_from_logits(dx_labels, pre_logits, threshold=0.5):
    """
    The same as preprocess_labels, I presume, but for one sample and without known labels.
    """
    # define an empty array to store the predicted labels
    pred_labels = np.zeros(len(dx_labels)) # pre_binary

    # Find the index of the maximum value within the logits
    likeliest_dx = np.argmax(pre_logits)

    # Add all Dx above the decision threshold
    pred_labels[pre_logits >= threshold] = 1

    # Make sure at least one diagnosis is included (the likeliest one)
    # NOTE if we don't include 'Norm' as a diagnosis, we can remove this line
    pred_labels[likeliest_dx] = 1  

    return pred_labels


def threshold_predict_from_logits(classes, pre_logits, threshold=0.5):
    """
    Takes logits from a BINARY classifier and returns the predicted labels, but only if the 
    probability of 'abnormal' is above the threshold.
    Hard-coded 'Abnormal' for now...
    """
    # define an empty array to store the predicted labels
    abnormal_index = np.where(classes == 'Abnormal')[0]
    pred_labels = np.zeros(len(classes))

    # Only return 1 if the probability of 'abnormal' is above the threshold
    if pre_logits[abnormal_index] >= threshold:
        pred_labels[abnormal_index] = 1
    else:
        pred_labels[~abnormal_index] = 1

    return pred_labels


def compute_f_measure_from_onehot(labels, outputs, unique_classes):
    """
    Takes in labels and pre_logits as one-hot encoded arrays.

    Similar to helper_code.compute_f_measure(), except that function expects inputs as 
    (label_dxs, output_dxs) where label_dxs and output_dxs are lists of diagnoses in string format.
    """
    A = helper_code.compute_one_vs_rest_confusion_matrix(labels, outputs, unique_classes)

    num_classes = len(unique_classes)
    per_class_f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            per_class_f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            per_class_f_measure[k] = float('nan')

    if np.any(np.isfinite(per_class_f_measure)):
        macro_f_measure = np.nanmean(per_class_f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, per_class_f_measure, unique_classes
