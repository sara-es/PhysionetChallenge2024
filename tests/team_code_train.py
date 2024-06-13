import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

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
    # tts = 0.6
    # records = shuffle(records)
    # train_records = records[:int(tts*num_records)]
    # val_records = records[int(tts*num_records):]

    # use tiny testset for testing
    data_folder = os.path.join("tiny_testset", "lr_unet_tests", "data_images")
    records = helper_code.find_records(data_folder)
    num_records = len(records)
    train_records = records[:10]
    val_records = records[10:]
    images_folder = os.path.join("tiny_testset", "lr_unet_tests", "data_images")
    masks_folder = os.path.join("tiny_testset", "lr_unet_tests", "binary_masks")
    patch_folder = os.path.join("tests", "data", "patches")
    unet_output_folder = os.path.join("tiny_testset", "lr_unet_tests", "unet_outputs")
    reconstructed_signals_folder = os.path.join("tiny_testset", "test_outputs")

    # print(train_records)

    # generate images and masks for training u-net; generate patches
    # TODO generate_unet_training_data
    images_folder = os.path.join("ptb-xl", "train_images")
    masks_folder = os.path.join("ptb-xl", "train_masks")
    
    team_code.generate_unet_training_data(data_folder, images_folder, 
                                          masks_folder, patch_folder, 
                                          verbose, records_to_process=records)

    # train u-net
    args = Unet.utils.Args()
    args.train_val_prop = 0.8
    args.epochs = 1
    team_code.train_unet(train_records, patch_folder, model_folder, verbose, 
                         args=args, warm_start=True)

    # save trained u-net
    # included in train_unet step

    # generate new images, patch them, then run u-net, then
    # reconstruct signals from u-net outputs, then save reconstructed signals
    team_code.generate_and_predict_unet_batch(data_folder, images_folder, masks_folder, patch_folder,
                                  unet_output_folder, model_folder, reconstructed_signals_folder,
                                  verbose, records_to_process=None, delete_images=False)

    # team_code.reconstruct_signal_from_unet_output()

    # train classification model
    resnet_model, uniq_labels = team_code.train_classifier(
        data_folder, verbose, records_to_process=None
        )

    # save trained classification model

    # optionally display some results



if __name__ == "__main__":
    data_folder = "G:\\PhysionetChallenge2024\\ptb-xl\\combined_records"
    model_folder = "G:\\PhysionetChallenge2024\\model"
    verbose = True

    train_models(data_folder, model_folder, verbose)
