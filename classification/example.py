import numpy as np
from sklearn.ensemble import RandomForestClassifier

import helper_code

def train(features, labels):
    features = np.vstack(features)
    classes = sorted(set.union(*map(set, labels)))
    labels = helper_code.compute_one_hot_encoding(labels, classes)

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state   = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_leaf_nodes=max_leaf_nodes, 
            random_state=random_state
        ).fit(features, labels)
    
    return model