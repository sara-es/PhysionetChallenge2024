import numpy as np

import helper_code

# Extract features.
def extract_features(record):
    images = helper_code.load_image(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])