import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
import torch
from digitization.Unet.ECGunet import BasicResUNet
import team_code, helper_code
from digitization import Unet


def load_unet_from_dict(LOAD_PATH_UNET, verbose=True):
    cuda = torch.cuda.is_available()
    print(f'Cuda available: {cuda}')

     # Load the model
    unet = BasicResUNet(3, 2, nbs=[1, 1, 1, 1], init_channels=16, cbam=False)
    if cuda:
        unet = unet.cuda()

    if verbose:
        print('Loading U-net Weights...')
    encoder_dict = unet.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_UNET)['model_state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
    if verbose:  
        print('weights loaded unet = ', len(pretrained_dict), '/', len(encoder_dict))
    unet.load_state_dict(torch.load(LOAD_PATH_UNET)['model_state_dict'])


def load_unet(LOAD_PATH_UNET, verbose=True):
    try:
        torch.load(LOAD_PATH_UNET)
    except:
        print(f'Could not load torch model from {LOAD_PATH_UNET}.')
        return

    cuda = torch.cuda.is_available()
    unet = BasicResUNet(3, 2, nbs=[1, 1, 1, 1], init_channels=16, cbam=False)
    if cuda:
        unet = unet.cuda()

    unet_state_dict = torch.load(LOAD_PATH_UNET)
    print(f'U-net: loaded {len(unet_state_dict)}/{len(unet.state_dict())} weights.')
    unet.load_state_dict(unet_state_dict)
    

def save_unet(SAVE_PATH_UNET, unet, verbose=True):
    try:
        torch.save(unet, SAVE_PATH_UNET + '.pth')
    except:
        print(f'Could not save torch model to {SAVE_PATH_UNET}.')


def test_load_save(save_path):
    record_ids = helper_code.find_records(os.path.join('tiny_testset', 'lr_gt'))
    patch_folder = os.path.join(os.getcwd(), 'temp_data', 'patches')
    model_folder = 'model'
    args = Unet.utils.Args()
    args.epochs = 1
    unet_model = team_code.train_unet(record_ids, patch_folder, model_folder, verbose=True, 
                                      args=args, warm_start=True, delete_patches=False)
    save_unet(save_path, unet_model)


if __name__ == '__main__':
    # patchsize = 256
    # LOAD_PATH_UNET = os.path.join('model', 'UNET_256_checkpoint')
    LOAD_PATH_UNET = os.path.join('model', 'pretrained', "UNET_run1_256_aug_checkpoint")
    # load_unet(LOAD_PATH_UNET)
    SAVE_PATH = os.path.join('model', 'UNET_256_checkpoint')
    test_load_save(SAVE_PATH)
    load_unet(SAVE_PATH)
    