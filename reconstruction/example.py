# This overly simple model uses the mean of these overly simple features as a seed for a random number generator.

import numpy as np

def train(features):
    model = np.mean(features)
    return model