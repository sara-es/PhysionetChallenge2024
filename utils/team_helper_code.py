import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np

import helper_code


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


def load_image_paths(record):
    path = os.path.split(record)[0]
    image_files = helper_code.get_image_files(record)

    images = list()
    for image_file in image_files:
        image_file_path = os.path.join(path, image_file)
        if os.path.isfile(image_file_path):
            images.append(image_file_path)

    return images