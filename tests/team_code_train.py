import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet


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
    
    # Get the file paths of signals
    tts = 0.6
    records = shuffle(records)
    train_records = records[:int(tts*num_records)]
    val_records = records[int(tts*num_records):]

    # test on a tiny number of records for now
    num_records = len(records)
    train_records = records[:10]
    val_records = records[10:20]
    print(val_records)

    images_folder = os.path.join("temp_data", "train_images")
    masks_folder = os.path.join("temp_data", "train_masks")
    patch_folder = os.path.join("temp_data", "patches")
    unet_output_folder = os.path.join("temp_data", "unet_outputs")
    reconstructed_signals_folder = os.path.join("tiny_testset", "test_outputs")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(unet_output_folder, exist_ok=True)
    os.makedirs(reconstructed_signals_folder, exist_ok=True)

    # print(train_records)

    # generate images and masks for training u-net; generate patches 
    team_code.generate_unet_training_data(data_folder, images_folder, 
                                          masks_folder, patch_folder, 
                                          verbose, records_to_process=train_records)
    
    # train u-net
    args = Unet.utils.Args()
    args.train_val_prop = 0.8
    args.epochs = 1
    unet_model = team_code.train_unet(train_records, patch_folder, model_folder, verbose, 
                         args=args, warm_start=True)

    # save trained u-net
    # included in train_unet step

    # generate new images, patch them, then run u-net, then
    # reconstruct signals from u-net outputs, then save reconstructed signals
    team_code.generate_and_predict_unet_batch(data_folder, images_folder, masks_folder, patch_folder,
                                  unet_output_folder, model_folder, reconstructed_signals_folder,
                                  verbose, records_to_process=val_records, delete_images=False)

    # team_code.reconstruct_signal_from_unet_output()

    # train classification model
    resnet_model, uniq_labels = team_code.train_classification_model(
        data_folder, verbose, records_to_process=val_records
        )

    # save trained classification model
    team_code.save_models(model_folder, unet_model, resnet_model, uniq_labels)

    # optionally display some results



if __name__ == "__main__":
    # data_folder = "G:\\PhysionetChallenge2024\\ptb-xl\\combined_records"
    data_folder = "G:\\PhysionetChallenge2024\\tiny_testset\\lr_gt"
    model_folder = "G:\\PhysionetChallenge2024\\model"
    verbose = True

    train_models(data_folder, model_folder, verbose)
