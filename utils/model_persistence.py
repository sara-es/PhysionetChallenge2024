import sys, os, joblib, pickle
import torch
sys.path.append(os.path.join(sys.path[0], '..'))


# Functions to save models depending on type
def save_model_pkl(model, name, folder):
    if model is not None:
        try:
            fn = os.path.join(folder, name + '.sav')
            joblib.dump(model, open(fn, 'wb'))
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


def load_checkpoint_dict(folder, name, verbose=False):
    """
    This is ONLY for model saves with checkpoint in the name. It will not work for normal torch 
    model saves (.pth files).

    Checkpoint saves are dictionaries with keys: 'epoch', 'model_state_dict', 
    'optimizer_state_dict', and 'loss'.
    """
    import torch
    import torch.optim as optim
    from digitization.Unet.ECGunet import BasicResUNet
    cuda = torch.cuda.is_available()

    fn = os.path.join(folder, name)
    # Load the model
    unet = BasicResUNet(3, 2, nbs=[1, 1, 1, 1], init_channels=16, cbam=False)
    optimizer = optim.AdamW(params=unet.parameters())
    if cuda:
        unet = unet.cuda()
    if verbose:
        print('Loading U-net Weights...')
    try:
        checkpoint = torch.load(fn)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_reached = checkpoint['epoch']
        loss_store = [[checkpoint['loss'], None]]
        if verbose:
            print(f'{len(checkpoint["model_state_dict"])}/{len(unet.state_dict())} weights ' +\
                    f'loaded from {fn}. Starting from epoch {epoch_reached}.',
                    flush=True)
    except Exception as e:
        print(e)
        print(f'Could not load U-net checkpoint from {fn}.')
    return unet


def load_models(model_folder, verbose, models_to_load):
    team_models = {}

    if verbose:
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
                    if verbose:
                        print(f"Loaded {name} model.")
                except ValueError:
                    print(f"I couldn't load the pickled model {fnp}.")
            # else try to load torch
            elif os.path.exists(fnt):
                device = torch.device("cpu")
                try:
                    model = torch.load(fnt, map_location=device)
                    team_models[name] = model
                    if verbose:
                        print(f"Loaded {name} model.")
                except ValueError:
                    print(f"I couldn't load the torch model {fnt}.")
            else:
                print(f"I can't find the model {name}.")
    else:
        print(f"{model_folder} not found or is not a valid path.")

    return team_models