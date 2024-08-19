import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing.classifier import datasets, utils
from preprocessing.classifier.ResNet_adapt import ResNet_adapt


def load_from_state_dict(state_dict):
    model = ResNet_adapt(embsize=2)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(state_dict)
    return model


def predict(model, dataloader, have_labels=False):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model.eval()
    pred_store = []
    true_store = []
    with torch.no_grad():
        for data, target in dataloader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data.to(device), target.to(device)
            x = model(data)
            pred_store.append(x.detach().cpu().numpy())
            if have_labels:
                true = target.detach().cpu().numpy()
                true_store.append(true)
    pred = np.array(pred_store)
    true = np.array(true_store)

    return pred, true


def classify_image(image_id, patch_folder, model, verbose):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # get paths of patches
    patch_names = os.listdir(patch_folder)
    patch_paths = [os.path.join(patch_folder, p) for p in patch_names if image_id in p]
    patch_paths = sorted(patch_paths)
    test_dataset = datasets.PatchDataset(patch_paths, None, train=False, transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    class_probs, _ = predict(model, test_dataloader, have_labels=False)

    # get the class with the highest probability
    # class_probs is shape (n_patches, 1, n_classes)
    class_probs = np.squeeze(class_probs)
    class_probs = np.mean(class_probs, axis=0)

    is_real = np.argmax(class_probs) # 1 if real, 0 if generated
    return is_real




    
