#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most 
# parts of the required functions, change or remove non-required functions, and add your own 
# functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib, os, sys, time
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

import helper_code
from utils import team_helper_code, constants, model_persistence
from digitization import Unet, ECGminer
from classification import seresnet18
import generator, preprocessing, classification
import generator.gen_ecg_images_from_data_batch
from evaluation import eval_utils

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of 
# the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code,
# but do *not* change the arguments of this function. If you do not train one of the models, then
# you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    start_time = time.time()
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = helper_code.find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Train the digitization model. If you are not training a digitization model, then you can
    # remove this part of the code.
    if verbose:
        print('Training the digitization model...')

    digitization_model, reconstructed_signals_folder = train_digitization_model(
        data_folder, model_folder, verbose, records_to_process=records, delete_training_data=False)
    
    if verbose:
        time1 = time.time()
        print(f'Done. Time to train digitization model and generate classifier training data: ' + \
              f'{time1 - start_time:.2f} seconds.')
    
    # Extract the features and labels from the data.
    if verbose:
        print('Training the classification model...')

    classification_model, classes = train_classification_model(reconstructed_signals_folder, 
                                                               verbose, records_to_process=None)

    if verbose:
        time2 = time.time()
        print(f'Done. Time to train digitization model and generate classifier training data: ' + \
              f'{time2 - time1:.2f} seconds.')
        
    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        print('Done. Total time to train models: ' + f'{time.time() - start_time:.2f} seconds.')
        print()


# Load your trained models. This function is *required*. You should edit this function to add your
# code, but do *not* change the arguments of this function. If you do not train one of the models,
# then you can return None for the model.
def load_models(model_folder, verbose):
    models = model_persistence.load_models(model_folder, verbose, 
                        models_to_load=['digitization_model', 'classification_model', 'dx_classes'])
    digitization_model = models['digitization_model']
    classification_model = models['classification_model', 'dx_classes']
    return digitization_model, classification_model


# Run your trained digitization model. This function is *required*. You should edit this function
# to add your code, but do *not* change the arguments of this function. If you did not train one of
# the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Load the digitization model.
    unet_model = Unet.utils.load_unet_from_state_dict(digitization_model)

    # Run the digitization model; if you did not train this model, then you can set signal=None.
    signal, reconstructed_signal_dir = unet_predict_single_image(record, unet_model, verbose, 
                                                                 delete_patches=True)
    
    # Load the classification model and classes.
    resnet_model = classification_model['classification_model']
    dx_classes = classification_model['dx_classes']
    
    # Run the classification model; if you did not train this model, then you can set labels=None.
    labels = classify_signals(record, reconstructed_signal_dir, resnet_model, 
                                dx_classes, verbose=verbose)
    
    # delete any temporary files
    for f in os.listdir(reconstructed_signal_dir):
        os.remove(os.path.join(reconstructed_signal_dir, f))
    
    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    if digitization_model is not None:
        model_persistence.save_model_torch(digitization_model, 'digitization_model', model_folder)

    if classification_model is not None:
        d = {'classification_model': classification_model, 'dx_classes': classes}
        model_persistence.save_models(d, model_folder, verbose=False)
        

def train_digitization_model(data_folder, model_folder, verbose, records_to_process=None,
                             delete_training_data=True):
    """
    Our general digitization process is
    1. generate testing images and masks
    2. preprocess testing images to estimate grid size/scale
    3. generate u-net patches
    4. run u-net on patches
    5. recover full image with signal outline from u-net outputs
    6. reconstruct signal and trace from u-net output

    At each step we save the outputs to disk to save on memory; here, we assume by default that 
    they should be deleted when no longer needed, but if you want to keep them for debugging or
    visualization, set delete_trainin_data to False. 
    """
    # hard code some folder paths for now
    images_folder = os.path.join(os.getcwd(), 'temp_data', 'images')
    masks_folder = os.path.join(os.getcwd(), 'temp_data', 'masks')
    patch_folder = os.path.join(os.getcwd(), 'temp_data', 'patches')
    unet_output_folder = os.path.join(os.getcwd(), 'temp_data', 'unet_outputs')
    reconstructed_signals_folder = os.path.join(os.getcwd(), 'temp_data', 
                                                'reconstructed_signals')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(unet_output_folder, exist_ok=True)
    os.makedirs(reconstructed_signals_folder, exist_ok=True)

    # TODO can do a split here if we want to have unet train and predict on different records
    if not records_to_process:
        records_to_process = helper_code.find_records(data_folder)

    # generate images and masks for training u-net; generate patches
    generate_unet_training_data(data_folder, images_folder, 
                                masks_folder, patch_folder, 
                                verbose, records_to_process=records_to_process)
    if verbose:
        print(f'Done.')
    
    # train U-net
    args = Unet.utils.Args()
    args.train_val_prop = 1.0 # we want to train on all available data
    args.epochs = 1 # TODO: increase this for actual training
    unet_model = train_unet(records_to_process, patch_folder, model_folder, verbose, args=args, 
                            warm_start=True)
    if verbose:
        print(f'Done.')

    if verbose:
        print('Generating training data for classification...')
    # generate training data for resnet: we want to use the reconstruction predictions (should
    # not be a problem to re-use the same base data, if we generate images with another seed?)
    # The following function generates new images, patches them, runs u-net, reconstructs signals
    # from u-net outputs, then finally saves the reconstructed signals in wfdb format.
    generate_and_predict_unet_batch(data_folder, images_folder, masks_folder, patch_folder,
                                  unet_output_folder, unet_model, reconstructed_signals_folder,
                                  verbose, records_to_process=records_to_process, 
                                  delete_images=delete_training_data)
    
    return unet_model, reconstructed_signals_folder
        

def generate_unet_training_data(wfdb_records_folder, images_folder, masks_folder, patch_folder,
                                verbose, patch_size=constants.PATCH_SIZE, records_to_process=None):
    """
    Call generate_images_from_wfdb to generate images and masks; then patchify the images and masks
    for training the U-Net model. Save the patches in patches_folder. 
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(wfdb_records_folder)

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.print_header = True
    img_gen_params.input_directory = wfdb_records_folder
    img_gen_params.output_directory = images_folder

    # set params for generating masks
    mask_gen_params = generator.MaskArgs()
    mask_gen_params.input_directory = wfdb_records_folder
    mask_gen_params.output_directory = masks_folder

    # generate images and masks
    if verbose:
        print("Generating images from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process)
    if verbose:
        print("Generating masks from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(mask_gen_params, records_to_process)

    # generate patches
    Unet.patching.save_patches_batch(records_to_process, images_folder, masks_folder, patch_size,
                                     patch_folder, verbose, max_samples=False)


def train_unet(record_ids, patch_folder, model_folder, verbose, 
               args=None, max_train_samples=5000, warm_start=True, delete_patches=True):
    """
    Train the U-Net model from patches and save the resulting model. 
    Note that no validation is done by default - during the challenge we will want to train
    on all available data. Manually set args.train_val_prop to a value between 0 and 1 to
    enforce validation.
    """
    if not args: # use default args if none are provided
        args = Unet.utils.Args()
    
    patchsize = constants.PATCH_SIZE
    # path for model checkpoints, used with early stopping or to resume training later
    CHK_PATH_UNET = os.path.join(model_folder, 'UNET_' + str(patchsize))
    # for saving the loss values, used with early stopping
    LOSS_PATH = os.path.join(model_folder, 'UNET_' + str(patchsize) + '_losses')
    # if we're loading a pretrained model - hardcoded for now
    LOAD_PATH_UNET = None
    if warm_start:
        chkpt_path = os.path.join('digitization', 'model_checkpoints', 
                                      'UNET_'+ str(patchsize) + '_checkpoint')
        if not os.path.exists(chkpt_path):
            print(f"Warm start requested but no checkpoint found at {LOAD_PATH_UNET}, " +\
                  "training U-net from scratch.")
        else:
            LOAD_PATH_UNET = chkpt_path

    image_patch_folder = os.path.join(patch_folder, 'image_patches')
    mask_patch_folder = os.path.join(patch_folder, 'label_patches')

    unet_model = Unet.train_unet(record_ids, image_patch_folder, mask_patch_folder,
            args, CHK_PATH_UNET, LOSS_PATH, LOAD_PATH_UNET, verbose,
            max_samples=max_train_samples,
            )
    
    if delete_patches: 
        for im in os.listdir(image_patch_folder):
            os.remove(os.path.join(image_patch_folder, im))
        for im in os.listdir(mask_patch_folder):
            os.remove(os.path.join(mask_patch_folder, im))

    return unet_model


def reconstruct_signal(record, unet_image, header_txt, 
                       reconstructed_signals_folder, save_signal=True):
    """

    """

    # TODO: get gridsize from header file
    # alternately can pass original image in as an argument to this function and extract
    # gridsize from the image here
    ###### FIXME hardcoded gridsize for now ######
    gridsize = 37.5

    # reconstruct signals from u-net outputs
    signal_length = helper_code.get_num_samples(header_txt)
    reconstructed_signal, trace = ECGminer.digitize_image_unet(unet_image, gridsize, 
                                                               sig_len=signal_length)
    reconstructed_signal = np.asarray(np.nan_to_num(reconstructed_signal))

    # save reconstructed signal and copied header file in the same folder
    if save_signal:
        output_record_path = os.path.join(reconstructed_signals_folder, record)
        helper_code.save_header(output_record_path, header_txt)
        comments = [l for l in header_txt.split('\n') if l.startswith('#')]
        helper_code.save_signals(output_record_path, reconstructed_signal, comments)

    # TODO: adapt self.postprocessor.postprocess to work for different layouts

    return reconstructed_signal, trace


def generate_and_predict_unet_batch(wfdb_records_folder, images_folder, mask_folder, patch_folder,
                                  unet_output_folder, unet_model, reconstructed_signals_folder,
                                  verbose, records_to_process=None, delete_images=True):
    """
    An all-in-one to generate images from records, run them through the U-Net model, and 
    reconstruct the patches to a full image. Assumes we are generating these images, so we have 
    masks (labels), and can return a DICE score for evaluation.

    TODO: can either move a lot of these folder names to constants, or hard code them from a base 
    directory since they're all temporary files anyway
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(wfdb_records_folder)

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.random_bw = 0.2
    img_gen_params.wrinkles = True
    img_gen_params.print_header = True
    img_gen_params.input_directory = wfdb_records_folder
    img_gen_params.output_directory = images_folder

    # generate images 
    if verbose:
        print("Generating images from wfdb files...")
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process)

    # generate patches
    Unet.patching.save_patches_batch(records_to_process, images_folder, mask_folder, constants.PATCH_SIZE, 
                                     patch_folder, verbose, max_samples=False)
    dice_list = Unet.batch_predict_full_images(records_to_process, patch_folder, unet_model, 
                                   unet_output_folder, verbose, save_all=True)

    # reconstruct_signals
    reconstructed_signals = []
    snr_list = np.zeros(len(records_to_process))
    for i, record in tqdm(enumerate(records_to_process), 
                       desc='Reconstructing signals from U-net outputs', disable=not verbose):
        # load u-net outputs
        record_id = team_helper_code.find_available_images(
                            [record], unet_output_folder, verbose)[0] # returns list
        unet_image_path = os.path.join(unet_output_folder, record_id + '.npy')
        with open(unet_image_path, 'rb') as f:
            unet_image = np.load(f)

        # reconstruct signal
        # load header file to save with reconstructed signal
        record_path = os.path.join(wfdb_records_folder, record) 
        label_signal, label_fields = helper_code.load_signals(record_path)
        header_txt = helper_code.load_header(record_path)
        rec_signal, _ = reconstruct_signal(record_id, unet_image, header_txt, 
                       reconstructed_signals_folder)
        reconstructed_signals.append(rec_signal)  

        snr_list[i], _, _, _, _ = eval_utils.single_signal_snr(rec_signal, label_fields, 
                                    label_signal, label_fields, record, extra_scores=False)

    if verbose:
        print(f"Average DICE score: {np.mean(dice_list)}")
        print(f"Average SNR: {np.mean(snr_list)}")

    # delete patches (we have the full images/masks in images_folder)
    im_patch_dir = os.path.join(patch_folder, 'image_patches')
    label_patch_dir = os.path.join(patch_folder, 'label_patches')
    for im in os.listdir(im_patch_dir):
        os.remove(os.path.join(im_patch_dir, im))
    for im in os.listdir(label_patch_dir):
        os.remove(os.path.join(label_patch_dir, im))

    # optional: delete training images and masks, and u-net outputs
    if delete_images:
        for im in os.listdir(images_folder):
            os.remove(os.path.join(images_folder, im))
        for im in os.listdir(mask_folder):
            os.remove(os.path.join(mask_folder, im))
        for im in os.listdir(unet_output_folder):
            os.remove(os.path.join(unet_output_folder, im))


def train_classification_model(reconstructed_records_folder, verbose, 
                     records_to_process=None):
    """
    Extracts features and labels from headers, then one-hot encodes labels and trains the
    SE-ResNet model.
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(reconstructed_records_folder)

    all_data = []
    labels = []
    for record in tqdm(records_to_process, desc='Loading classifier training data', 
                       disable=not verbose):
        data, label = classification.get_training_data(record, 
                                                       reconstructed_records_folder)
        if label is None: # don't use data without labels for training
            continue

        all_data.append(data)
        labels.append(label)
    
    # Make sure we actually have data and labels
    if len(all_data) == 0:
        raise ValueError("No records with labels found in records to process.")

    # One-hot-encode labels 
    mlb = MultiLabelBinarizer()
    multilabels = mlb.fit_transform(labels)
    uniq_labels = mlb.classes_

    # TODO: check if frequency is used/if it's important
    if verbose:
        print("Training SE-ResNet classification model...")
    resnet_model = seresnet18.train_model(
                                all_data, multilabels, uniq_labels, verbose, epochs=5, 
                                validate=False)
    
    if verbose:
        print("Finished training classification model.")
    
    return resnet_model, uniq_labels


def preprocess_images(raw_images_folder, processed_images_folder, verbose, 
                      records_to_process=None):
    """
    Preprocess images found in raw_images_folder and save them in processed_images_folder.
    Optionally provide a list of a subset of records to process (records_to_process).

    Preprocessing steps currently implemented:
        - resize images to a standard size
    
    TODO:
        - determine grid size of the image and save it to either the original header file 
        (will need to pass in the header file path, wfdb_records_folder) or a new file
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(raw_images_folder)
    
    for i in tqdm(range(len(records_to_process)), desc='Preprocessing images', disable=not verbose):
        record = records_to_process[i] 
        #raw_image_path = os.path.join(raw_images_folder, record + '.png')
        
        # load raw image
        record_path = os.path.join(raw_images_folder, record)
        # TODO below commented line returns a PIL image and I had trouble working with it - may want to check this?
        # image = helper_code.load_images(record_image)[0] 
        record_image_name = team_helper_code.find_available_images(
                            [record], raw_images_folder, verbose)[0] # returns list
        with open(os.path.join(raw_images_folder, record_image_name + ".png"), 'rb') as f:
            image = plt.imread(f)

        # resize image if needed
        # TODO this breaks
        # image = preprocessing.resize_image(image)
        
        # get and save the gridsize
        grayscale_image = preprocessing.cepstrum_grid_detection.image_to_grayscale_array(image)
        
        # TODO: fix get_rotation_angle - it breaks for tiny_test/hr_gt/01017_hr
        rot_angle, gridsize = preprocessing.cepstrum_grid_detection.get_rotation_angle(grayscale_image)
        team_helper_code.save_gridsize(record_path, gridsize)
        
        # TODO: set image to the rotated image

        # save processed image
        processed_image = os.path.join(processed_images_folder, record_image_name + '.png')
        with open(processed_image, 'wb') as f:
            plt.imsave(f, image, cmap='gray')
        # image.save(processed_image,"PNG") # check this works? Note: it does not work

        # save header file with gridsize to processed_images_folder
        header_txt = helper_code.load_header(record_path)
        output_record_path = os.path.join(processed_images_folder, record)
        helper_code.save_header(output_record_path, header_txt)


def unet_predict_single_image(record_path, model, verbose, delete_patches=True):
    """
    params
        record_id: str, for saving/loading
        
    """
    # get image from image_path
    image = helper_code.load_images(record_path)[0]
    record_id = os.path.split(record_path)[-1].split('.')[0]

    # hard code some folder paths for now
    patch_folder = os.path.join('temp_data', 'patches', 'test_image_patches')
    reconstructed_signals_folder = os.path.join('temp_data', 'reconstructed_signals')
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(reconstructed_signals_folder, exist_ok=True)

    # preprocess image
    # TODO

    # patchify image
    image = np.asarray(image) # convert to numpy array
    # TODO will need to save/load patch size and original image size for persistence
    Unet.patching.save_patches_single_image(record_id, image, None, 
                                            patch_size=constants.PATCH_SIZE,
                                            im_patch_save_path=patch_folder,
                                            lab_patch_save_path=None)

    # predict on patches, recover u-net output image
    # TODO need to pass original image size as argument here
    predicted_image = Unet.predict_single_image(record_id, patch_folder, model)

    # reconstruct signal from u-net output image
    # load header file to save with reconstructed signal
    header_txt = helper_code.load_header(record_path)
    reconstructed_signal, trace = reconstruct_signal(record_id, predicted_image, 
                                                     header_txt,
                                                     reconstructed_signals_folder, 
                                                     save_signal=True)

    # optional: delete patches
    if delete_patches:
        for im in os.listdir(patch_folder):
            os.remove(os.path.join(patch_folder, im))

    # return reconstructed signal
    return reconstructed_signal, reconstructed_signals_folder


def classify_signals(record_path, data_folder, resnet_model, classes, verbose):
    # wrap in list to match training data format
    record_id = os.path.split(record_path)[-1].split('.')[0]
    data = [classification.get_testing_data(record_id, data_folder)] 
    pred_dx, probabilities = seresnet18.predict_proba(
                                        resnet_model, data, classes, verbose)
    labels = classes[np.where(pred_dx == 1)]
    if verbose:
        print(f"Classes: {classes}, probabilities: {probabilities}")
        print(f"Predicted labels: {labels}")

    return labels

    

