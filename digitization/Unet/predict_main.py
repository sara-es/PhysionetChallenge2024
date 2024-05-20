# Nicola Dinsdale 2024
# ECG segmentation predict main
########################################################################################################################
# Import dependencies 
import numpy as np
import torch
from torch.autograd import Variable

########################################################################################################################
def normal_predict(args, model, test_loader):
    cuda = torch.cuda.is_available()

    pred = []
    true = []
    orig = []
    
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            orig.append(data.detach().cpu().numpy())
            true.append(target.detach().cpu().numpy())
            x = model(data)
            pred.append(x.detach().cpu().numpy())
    orig = np.array(orig)
    pred = np.array(pred)
    true = np.array(true)
    return pred, true, orig

def dice(ground_truth, prediction):
    # Calculate the 3D dice coefficient of the ground truth and the prediction
    ground_truth = ground_truth > 0.5  # Binarize volume
    prediction = prediction > 0.5  # Binarize volume
    epsilon = 1e-5  # Small value to prevent dividing by zero
    true_positive = np.sum(np.multiply(ground_truth, prediction))
    false_positive = np.sum(np.multiply(ground_truth == 0, prediction))
    false_negative = np.sum(np.multiply(ground_truth, prediction == 0))
    dice3d_coeff = 2*true_positive / \
        (2*true_positive + false_positive + false_negative + epsilon)
    return dice3d_coeff
########################################################################################################################

