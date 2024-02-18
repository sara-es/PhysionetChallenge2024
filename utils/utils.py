import sys, os, joblib, pickle
import torch
sys.path.append(os.path.join(sys.path[0], '..'))


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


def save_models(models, model_folder, verbose=True):
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            save_model_torch(model, name, model_folder)
        else:
            save_model_pkl(model, name, model_folder)
        if verbose:
            print(f'{name} model saved.')


def load_models(model_folder, verbose, models_to_load):
    team_models = {}

    if verbose >= 1:
        print(f'Attempting to load from {model_folder}...')

    if os.path.exists(model_folder):
        for name in models_to_load:
            # try to load pkl
            fnp = os.path.join(model_folder, name + '.sav')
            fnt = os.path.join(model_folder, name + '.pth')
            if os.path.exists(fnp):
                try:
                    model = pickle.load(open(fnp, 'rb'))
                    team_models[name] = model
                    if verbose >= 1:
                        print(f"Loaded {name} model.")
                except ValueError:
                    print(f"I couldn't load the pickled model {fnp}.")
            # else try to load torch
            elif os.path.exists(fnt):
                device = torch.device("cpu")
                try:
                    model = torch.load(fnt, map_location=device)
                    team_models[name] = model
                    if verbose >= 1:
                        print(f"Loaded {name} model.")
                except ValueError:
                    print(f"I couldn't load the torch model {fnt}.")
            else:
                print(f"I can't find the model {name}.")
    else:
        print(f"{model_folder} not found or is not a valid path.")

    return team_models