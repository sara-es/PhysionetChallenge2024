import sys, os, joblib, pickle
sys.path.append(os.path.join(sys.path[0], '..'))

# Save your trained digitization model.
def save_digitization_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'digitization_model.sav')
    joblib.dump(d, filename, protocol=0)


# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'dx_model.sav')
    joblib.dump(d, filename, protocol=0)


# Functions to save models depending on type
def save_model_pkl(model, name, folder):
    if model is not None:
        try:
            fn = os.path.join(folder, name + '.sav')
            pickle.dump(model, open(fn, 'wb'))
        except:
            print(f'Could not save model to {fn}.')


def save_model_torch(model, name, folder):
    import torch
    if model is not None:
        try:
            fn = os.path.join(folder, name + '.pth')
            torch.save(model, fn)
        except:
            print(f'Could not save torch model to {fn}.')