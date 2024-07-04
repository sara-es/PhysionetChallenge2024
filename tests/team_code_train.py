import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
from utils import model_persistence


def train_models(data_folder, model_folder, verbose):
    """
    Team code version
    To quickly test that a function is working, comment out all irrelevant code. 
    If the necessary data has already been generated (in tiny_testset, for example),
    everything should run independently.
    """
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    # test on a tiny number of records for now
    # records = records[:2000]
    # num_records = len(records)
    
    # Get the file paths of signals
    tts = 0.4
    records = shuffle(records, random_state=42)
    train_records = records[:int(tts*num_records)]
    val_records = records[int(tts*num_records):]

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

    # generate images and masks for training u-net; generate patches 
    team_code.generate_unet_training_data(data_folder, images_folder, 
                                          masks_folder, patch_folder, 
                                          verbose, records_to_process=train_records)
    
    # if images and masks are already generated, use only records that are present
    # records = helper_code.find_records(images_folder)
    # tts = 1
    # records = shuffle(records)
    # train_records = records[:int(tts*num_records)]
    # val_records = records[int(tts*num_records):]

    # train u-net
    args = Unet.utils.Args()
    args.train_val_prop = 0.8
    args.epochs = 50
    unet_model = team_code.train_unet(train_records, patch_folder, model_folder, verbose, 
                         args=args, warm_start=False, max_train_samples=False, delete_patches=True)

    # save trained u-net
    model_persistence.save_model_torch(unet_model, 'digitization_model', model_folder)

    # generate new images, patch them, then run u-net, then
    # reconstruct signals from u-net outputs, then save reconstructed signals
    team_code.generate_and_predict_unet_batch(data_folder, images_folder, masks_folder, patch_folder,
                                  unet_output_folder, unet_model, reconstructed_signals_folder,
                                  verbose, records_to_process=val_records, delete_images=True)

    # train classification model
    resnet_model, uniq_labels = team_code.train_classification_model(
        data_folder, verbose, records_to_process=val_records
        )

    # save trained classification model
    team_code.save_models(model_folder, unet_model, resnet_model, uniq_labels)

    # optionally display some results



if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "ptb-xl", "records500")
    model_folder = os.path.join(os.getcwd(), "model")
    verbose = True

    train_models(data_folder, model_folder, verbose)
