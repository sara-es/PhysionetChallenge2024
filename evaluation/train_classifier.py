import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
from preprocessing import classifier
from utils import model_persistence, constants


def train_unet(real_data_folder, model_folder, verbose):
    """
    Team code version. 
    """
    # Find the data files.
    real_images_folder = os.path.join(real_data_folder, 'images')
    real_masks_folder = os.path.join(real_data_folder, 'masks')
    real_patch_folder = os.path.join(real_data_folder, 'patches')
    os.makedirs(real_patch_folder, exist_ok=True)
    # check that real images and masks are available
    if not os.path.exists(real_images_folder) or not os.path.exists(real_masks_folder):
        print(f"Real images or masks not found in {real_data_folder}, unable to train " +\
                "real image classifier or u-net.")
    real_images = os.listdir(real_images_folder)

    # train classifier for real vs. generated data
    classifier.save_patches_batch(real_images_folder, real_masks_folder, 
                                  constants.PATCH_SIZE, real_patch_folder, 
                                  verbose, delete_images=False)
    
    # train u-net
    # args = Unet.utils.Args()
    # args.epochs = 500
    # args.augmentation = True
    # unet_model = team_code.train_unet(train_records, patch_folder, model_folder, verbose, 
    #                      args=args, warm_start=True, max_train_samples=30000, delete_patches=False)

    # # save trained u-net
    # model_persistence.save_model_torch(unet_model, 'digitization_model', model_folder)


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "real_images")
    model_folder = os.path.join(os.getcwd(), "model")
    verbose = True

    train_unet(data_folder, model_folder, verbose)
