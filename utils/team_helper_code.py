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


def compute_classification_metrics(actual_labels, pre_logits, threshold):

    actual_labels, pre_logits, pre_binary = preprocess_labels(actual_labels, pre_logits, threshold)

    # Compute the metrics
    f_measure, _, _ = helper_code.compute_f_measure(actual_labels, pre_binary)

    return f_measure