import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
from utils import model_persistence
from evaluation import generate_data


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
    tts = 1
    records = shuffle(records, random_state=42)
    num_records = len(records)
    train_records = records[:int(tts*num_records)]
    val_records = records[int(tts*num_records):]

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
        # generate images and masks for training u-net; generate patches
        print(f"Generating {num_images_to_generate} images and masks...")
        team_code.generate_unet_training_data(data_folder, images_folder, 
                                            masks_folder, patch_folder, 
                                            verbose, records_to_process=train_records)
    
    # train u-net
    args = Unet.utils.Args()
    args.train_val_prop = 0.8
    args.epochs = 50
    unet_model = team_code.train_unet(train_records, patch_folder, model_folder, verbose, 
                         args=args, warm_start=True, max_train_samples=False, delete_patches=False)

    # save trained u-net
    model_persistence.save_model_torch(unet_model, 'digitization_model', model_folder)

    # testing: generate new images, patch them, then run u-net, then
    # reconstruct signals from u-net outputs, then save reconstructed signals
    # generate_data.generate_and_predict_unet_batch(data_folder, images_folder, masks_folder, patch_folder,
    #                               unet_output_folder, unet_model, reconstructed_signals_folder,
    #                               verbose, records_to_process=val_records, delete_images=False)

    # train classification model
    # resnet_model, uniq_labels = team_code.train_classification_model(
    #     data_folder, verbose, records_to_process=val_records
    #     )

    # TODO: display some results



if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "ptb-xl", "records500")
    model_folder = os.path.join(os.getcwd(), "model")
    verbose = True
    num_images_to_generate = 0 # int, set to 0 if data has already been generated to speed up testing time

    train_unet(data_folder, model_folder, verbose, num_images_to_generate)
