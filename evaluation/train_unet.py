import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
from utils import model_persistence


def train_unet(data_folder, model_folder, verbose, num_images_to_generate=0):
    """
    Team code version. If num_images_to_generate=0, assumes data has already been generated.
    """
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')
    records = helper_code.find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    # Get the file paths of signals
    train_records = shuffle(records, random_state=42)
    train_records = shuffle(train_records, random_state=42)
    num_records = len(records)

    print(train_records[:5])

    images_folder = os.path.join("temp_data", "images")
    masks_folder = os.path.join("temp_data", "masks")
    patch_folder = os.path.join("temp_data", "patches")
    unet_output_folder = os.path.join("temp_data", "unet_outputs")
    reconstructed_signals_folder = os.path.join("temp_data", "test_outputs")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(unet_output_folder, exist_ok=True)
    os.makedirs(reconstructed_signals_folder, exist_ok=True)

    print(data_folder)

    if num_images_to_generate > 0:
        train_records = train_records[:num_images_to_generate]
        # generate images and masks for training u-net; generate patches
        print(f"Generating {num_images_to_generate} images and masks...")
        team_code.generate_training_images(data_folder, images_folder, 
                                            masks_folder, patch_folder, 
                                            verbose, records_to_process=train_records)
    
    # train u-net
    args = Unet.utils.Args()
    args.epochs = 500
    args.augmentation = True
    unet_model = team_code.train_unet(train_records, patch_folder, model_folder, verbose, 
                         args=args, warm_start=True, max_train_samples=30000, delete_patches=False)

    # save trained u-net
    model_persistence.save_model_torch(unet_model, 'digitization_model', model_folder)


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "ptb-xl", "records500")
    model_folder = os.path.join(os.getcwd(), "model")
    verbose = True
    num_images_to_generate = 0 # int, set to 0 if data has already been generated to speed up testing time

    train_unet(data_folder, model_folder, verbose, num_images_to_generate)
