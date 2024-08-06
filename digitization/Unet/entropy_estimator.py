# Nicola Dinsdale 2024
# Code to estimate the entropy from the output of the unet 
# Potentially use as a proxy for segmentation quality

# Entropy estimation 
from scipy.stats import entropy
import numpy as np 

def entropy_est(input_softmax,  reduce=False, mode='average'):
    '''
    input_softmax: model output,  C x H x W , where C is the segmentation channels
    reduce: True or False --> Return a single value per image or a single value across the image
    mode: 'average' or 'max' across the image --> recommend average that shows good correlation with dice score
    '''
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    soft_im = softmax(input_softmax)
    modes = ['average', 'max']
    assert mode in modes

    e = entropy(soft_im, axis=0) # Calculate entropy over channels
    
    if reduce:
        if mode == 'average':
            return e.mean()
        elif mode == 'max':
            return e.max()
        else:
            raise Exception('Mode not implemented')
    else:
        return e
    
    