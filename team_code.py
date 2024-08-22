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
import scipy as sp
import shutil
from sklearn.utils import shuffle
from PIL import Image

import digitization.YOLOv7
import digitization.YOLOv7.prepare_labels
import helper_code
import preprocessing.classifier
from utils import team_helper_code, constants, model_persistence
from digitization import Unet, ECGminer
from classification import seresnet18
from classification.utils import multiclass_predict_from_logits
import generator, preprocessing, digitization, classification
from preprocessing import classifier
import generator.gen_ecg_images_from_data_batch
from evaluation import eval_utils
from digitization.ECGminer.assets.DigitizationError import SignalExtractionError

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

    real_data_folder = os.path.join(os.getcwd(), 'real_images')
    if not os.path.exists(real_data_folder):
        real_data_folder = None
    unet_generated, unet_real, image_classifier = train_digitization_model(data_folder, 
                                                        model_folder, verbose, 
                                                        records_to_process=records, 
                                                        delete_training_data=constants.DELETE_DATA,
                                                        real_data_folder=real_data_folder)
    
    digitization_model = dict()
    digitization_model['unet_generated'] = unet_generated
    digitization_model['unet_real'] = unet_real
    digitization_model['image_classifier'] = image_classifier

    save_models(model_folder, digitization_model, None, None)

    if verbose:
        time1 = time.time()
        print(f'Done. Time to train digitization model: ' + \
              f'{time1 - start_time:.2f} seconds.')
    
    # Extract the features and labels from the data.
    if verbose:
        print('Training the classification model...')

    signal_classifier, classes = train_classification_model(data_folder, verbose, 
                                                               records_to_process=None)

    if verbose:
        time2 = time.time()
        print(f'Done. Time to train classification model: ' + \
              f'{time2 - time1:.2f} seconds.')
        
    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_models(model_folder, digitization_model, signal_classifier, classes)

    if verbose:
        print('Done. Total time to train models: ' + f'{time.time() - start_time:.2f} seconds.')
        print()


# Load your trained models. This function is *required*. You should edit this function to add your
# code, but do *not* change the arguments of this function. If you do not train one of the models,
# then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_model = dict()
    models = model_persistence.load_models(model_folder, verbose, 
                        models_to_load=[constants.YOLO_CONFIG, 
                                        'unet_generated',
                                        'unet_real',
                                        'image_classifier', 
                                        'dx_classes', 
                                        'res0', 'res1', 'res2', 'res3', 'res4'])
    digitization_model['unet_generated'] = Unet.utils.load_unet_from_state_dict(
                                                        models['unet_generated'])
    digitization_model['unet_real'] = Unet.utils.load_unet_from_state_dict(models['unet_real'])
    digitization_model['image_classifier'] = classifier.load_from_state_dict(
                                                    models['image_classifier'])
    digitization_model['yolov7'] = models[constants.YOLO_CONFIG]
    classification_model = dict((m, models[m]) for m in ['res0', 'res1', 'res2', 'res3', 'res4', 
                                                         'dx_classes'])
    return digitization_model, classification_model


# Run your trained digitization model. This function is *required*. You should edit this function
# to add your code, but do *not* change the arguments of this function. If you did not train one of
# the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal=None.
    signal, reconstructed_signal_dir, is_real_image = unet_reconstruct_single_image(record, 
                                                        digitization_model, 
                                                        verbose, 
                                                        delete_patches=constants.DELETE_DATA)
    
    # Load the classification model and classes.
    resnet_models = classification_model.copy()
    dx_classes = resnet_models.pop('dx_classes') # this gets the classes out of the dict
    
    # Run the classification model; if you did not train this model, then you can set labels=None.
    if signal is None: # if digitization failed, don't try to classify
        labels = None
    else: 
        labels = classify_signals(record, reconstructed_signal_dir, resnet_models, 
                                dx_classes, verbose=verbose)
    
    # delete any temporary files
    for f in os.listdir(reconstructed_signal_dir):
        os.remove(os.path.join(reconstructed_signal_dir, f))

    if is_real_image and constants.RETURN_SIGNAL_IF_REAL == False:
        signal = None
    
    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    if digitization_model is not None:
        model_persistence.save_models(digitization_model, model_folder)

    if classification_model is not None:
        model_persistence.save_models(classification_model, model_folder)
        model_persistence.save_model_pkl(classes, 'dx_classes', model_folder)
        

def train_digitization_model(data_folder, model_folder, verbose, records_to_process=None,
                             delete_training_data=True, 
                             max_size_training_set=constants.MAX_GENERATE_IMAGES,
                             real_data_folder=None):
    """
    Our general digitization process is
    1. generate synthetic images and masks
    2. preprocess 
    3. generate u-net patches
    4. run u-net on patches
    5. recover full image with signal outline from u-net outputs
    6. reconstruct signal and trace from u-net output

    At each step we save the outputs to disk to save on memory; here, we assume by default that 
    they should be deleted when no longer needed, but if you want to keep them for debugging or
    visualization, set delete_training_data to False. 
    """
    try: # clear any data from previous runs
        shutil.rmtree(os.path.join('temp_data', 'train'))
        shutil.rmtree(os.path.join('temp_data', 'val'))
    except FileNotFoundError:
        pass
    # GENERATED images, bounding boxes, masks, patches, and u-net outputs
    # hard code some folder paths for now
    gen_images_folder = os.path.join(os.getcwd(), 'temp_data', 'train', 'images')
    bb_labels_folder = os.path.join(os.getcwd(), 'temp_data', 'train', 'labels')
    gen_masks_folder = os.path.join(os.getcwd(), 'temp_data', 'train', 'masks')
    gen_patch_folder = os.path.join(os.getcwd(), 'temp_data', 'train', 'patches')
    unet_output_folder = os.path.join(os.getcwd(), 'temp_data', 'train', 'unet_outputs')

    os.makedirs(gen_images_folder, exist_ok=True)
    os.makedirs(bb_labels_folder, exist_ok=True)
    os.makedirs(gen_masks_folder, exist_ok=True)
    os.makedirs(gen_patch_folder, exist_ok=True)
    os.makedirs(unet_output_folder, exist_ok=True)

    if not records_to_process:
        records_to_process = helper_code.find_records(data_folder)
    if max_size_training_set is not None:
        records_to_process = shuffle(records_to_process)[:max_size_training_set]

    # generate images, bounding boxes, and masks for training YOLO and u-net
    # note that YOLO labels assume two classes: short and long leads
    generate_training_images(data_folder, gen_images_folder, 
                             gen_masks_folder, bb_labels_folder, 
                             verbose, records_to_process=records_to_process)
    
    # train YOLOv7 (only one epoch w/ low lr - assume pre-trained model is good enough)
    train_yolo(records_to_process, gen_images_folder, bb_labels_folder, model_folder,
               verbose, delete_training_data=delete_training_data)
    
    if verbose:
        print("Preparing to train semantic segmentation models...")

    # Generate patches for u-net. Note: this deletes source images and masks to save space
    # if delete_training_data is True
    # max_samples is the *approximate* number of patches that will be generated
    Unet.patching.save_patches_batch(records_to_process, gen_images_folder, gen_masks_folder, 
                                     constants.PATCH_SIZE, gen_patch_folder, verbose, 
                                     delete_images=delete_training_data, max_samples=40000)

    # Generate patches for real images if available
    if real_data_folder is not None:
        real_images_folder = os.path.join(real_data_folder, 'images')
        real_masks_folder = os.path.join(real_data_folder, 'masks')
        real_patch_folder = os.path.join('temp_data', 'train', 'real_patches')
        os.makedirs(real_patch_folder, exist_ok=True)
        # check that real images and masks are available
        if not os.path.exists(real_images_folder) or not os.path.exists(real_masks_folder):
            print(f"Real images or masks not found in {real_data_folder}, unable to train " +\
                    "real image classifier or u-net.")
        real_records = os.listdir(real_images_folder)
        real_records = [r.split('.')[0] for r in real_records]
        Unet.patching.save_patches_batch(real_records, real_images_folder, real_masks_folder, 
                                         constants.PATCH_SIZE, real_patch_folder, verbose, 
                                         delete_images=False, require_masks=False)
        # train classifier for real vs. generated data
        if verbose:
            print("Training real vs. generated image classifier...")
        image_classifier = classifier.train_image_classifier(real_patch_folder, gen_patch_folder, 
                                        model_folder, constants.PATCH_SIZE, verbose)

    args = Unet.utils.Args()
    args.train_val_prop = 0.8
    args.epochs = constants.UNET_EPOCHS
    if real_images_folder is not None:
        # train U-net: real data
        if verbose:
            print("Training U-net for real data...")
        unet_real = train_unet(real_records, real_patch_folder, model_folder, verbose, args=args,
                            warm_start=constants.WARM_START, ckpt_name='unet_real')

    # train U-net: generated data
    if verbose: 
        print("Training U-net for generated data...")
    unet_generated = train_unet(records_to_process, gen_patch_folder, model_folder, verbose,
                             args=args, warm_start=constants.WARM_START, ckpt_name='unet_gen')
    
    # optional: delete any leftover training data
    if delete_training_data:
        for im in os.listdir(gen_images_folder):
            os.remove(os.path.join(gen_images_folder, im))
        for im in os.listdir(gen_masks_folder):
            os.remove(os.path.join(gen_masks_folder, im))
        for im in os.listdir(bb_labels_folder):
            os.remove(os.path.join(bb_labels_folder, im))
        for folder in os.listdir(gen_patch_folder):
            for im in os.listdir(os.path.join(gen_patch_folder, folder)):
                os.remove(os.path.join(gen_patch_folder, folder, im))
        for im in os.listdir(unet_output_folder):
            os.remove(os.path.join(unet_output_folder, im))

    if verbose:
        print(f'Done.')
    
    return unet_generated, unet_real, image_classifier


def generate_training_images(wfdb_records_folder, images_folder, masks_folder, bb_labels_folder,
                                verbose, records_to_process=None):
    """
    Call generate_images_from_wfdb to generate images, lead bounding box information, and masks.
    Save the bounding box labels in YOLO format.
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(wfdb_records_folder)

    # params for generating images
    img_gen_params = generator.DefaultArgs()
    img_gen_params.input_directory = wfdb_records_folder
    img_gen_params.output_directory = images_folder
    img_gen_params.lead_bbox = True
    img_gen_params.lead_name_bbox = True
    img_gen_params.print_header = True
    img_gen_params.store_config = 2

    # set params for generating masks
    mask_gen_params = generator.MaskArgs()
    mask_gen_params.input_directory = wfdb_records_folder
    mask_gen_params.output_directory = masks_folder

    # generate images - params done manually because the generator doesn't implement seed correctly
    split = int(len(records_to_process)/4) # 25% no calibration pulse, 50% no noise/wrinkles
    records_to_process = shuffle(records_to_process)

    if verbose:
        print("Generating images from wfdb files (set 1/4)...")
    img_gen_params.calibration_pulse = 0
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[:split])

    if verbose:
        print("Generating images from wfdb files (set 2/4)...")
    img_gen_params.calibration_pulse = 1
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[split:int(split*2)])
    
    if verbose:
        print("Generating images from wfdb files (set 3/4)...")
    img_gen_params.calibration_pulse = 1
    img_gen_params.wrinkles = True
    img_gen_params.augment = True
    img_gen_params.crop = 0.0
    img_gen_params.rotate = 0
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[int(split*2):int(split*3)])
    
    if verbose:
        print("Generating images from wfdb files (set 4/4)...")  
    img_gen_params.random_bw = 0.2
    generator.gen_ecg_images_from_data_batch.run(img_gen_params, records_to_process[int(split*3):])

    # generate bounding box labels and save to a separate folder
    img_files = team_helper_code.find_files(images_folder, extension_str='.png')
    if verbose:
        print("Preparing bounding box labels...")
    digitization.YOLOv7.prepare_labels.prepare_label_files(img_files, images_folder, bb_labels_folder,
                                                           verbose)

    # generate masks
    if verbose:
        print("Generating masks from wfdb files (set 1/2)...")
    mask_gen_params.calibration_pulse = 0
    generator.gen_ecg_images_from_data_batch.run(mask_gen_params, records_to_process[:split])

    if verbose:
        print("Generating masks from wfdb files (set 2/2)...")
    mask_gen_params.calibration_pulse = 1
    generator.gen_ecg_images_from_data_batch.run(mask_gen_params, records_to_process[split:])
    
    if verbose:
        print(f'Done.')


def train_yolo(record_ids, train_data_folder, bb_labels_folder, model_folder, verbose, 
               args=None, delete_training_data=True, warm_start=constants.WARM_START):
    """
    A quick and dirty setup of yolo training config files, and model training call.
    """
    config = constants.YOLO_CONFIG
    if verbose:
        print("Preparing YOLOv7 training data...")
    if not args:
        args = digitization.YOLOv7.train.OptArgs()
        args.device = "0"
        args.cfg = os.path.join("digitization", "YOLOv7", "cfg", "training", config + ".yaml")
        args.data = os.path.join("digitization", "YOLOv7", "data", config + ".yaml")
        args.name = config
        args.epochs = constants.YOLO_EPOCHS
        args.weights = os.path.join("digitization", "model_checkpoints", "yolov7_training.pt")
        args.hyp = os.path.join("digitization", "YOLOv7", "data", "hyp.scratch.custom.yaml") #default

    # use the best weights from previous fine-tuning as starting point
    if warm_start and os.path.exists(os.path.join("digitization", "model_checkpoints", config + ".pt")):
        args.resume = True
        args.weights = os.path.join("digitization", "model_checkpoints", config + ".pt")

    # yolo requires we also have val data
    os.makedirs(os.path.join("temp_data", "val", "images"), exist_ok=True)
    os.makedirs(os.path.join("temp_data", "val", "labels"), exist_ok=True)
    # move some data to val
    val_record_ids = record_ids[:int(len(record_ids)/10)]
    for record in val_record_ids:
        record_id = record.split(os.sep)[-1]
        image_path = os.path.join(train_data_folder, record_id + "-0.png")
        label_path = os.path.join(bb_labels_folder, record_id + "-0.txt")
        try:
            shutil.move(image_path, os.path.join("temp_data", "val", "images"))
            shutil.move(label_path, os.path.join("temp_data", "val", "labels"))
        except:
            pass

    # Train the model
    if verbose:
        print("Training YOLOv7 model...")
    digitization.YOLOv7.train.main(args, verbose)

    # Find best weights and save them to model_folder
    if verbose:
        print("...Done. Saving best weights...")
    best_weights_path = os.path.join("temp_data", "train", config, "weights", "best.pt")
    os.makedirs(model_folder, exist_ok=True)
    try:
        os.remove(os.path.join(model_folder, config + ".pt")) # in case model already exists
    except FileNotFoundError:
        pass
    shutil.move(best_weights_path, os.path.join(model_folder))
    os.rename(os.path.join(model_folder, "best.pt"), os.path.join(model_folder, config + ".pt"))

    # move data back from val to train
    for record in val_record_ids:
        record_id = record.split(os.sep)[-1]
        image_path = os.path.join("temp_data", "val", "images", record_id + "-0.png")
        label_path = os.path.join("temp_data", "val", "labels", record_id + "-0.txt")
        shutil.move(image_path, train_data_folder)
        shutil.move(label_path, bb_labels_folder)

    if delete_training_data:
        shutil.rmtree(os.path.join("temp_data", "train", config))

    if verbose:
        print("...Done.")


def train_unet(record_ids, patch_folder, model_folder, verbose, 
               args=None, max_train_samples=None, warm_start=True, delete_patches=True,
               ckpt_name='unet'):
    """
    Train the U-Net model from patches and save the resulting model. 

    Params:
        record_ids: list of str, record IDs to train on
        patch_folder: str, path to folder with image and mask patches
        model_folder: str, path to folder to save model checkpoints
        verbose: bool
        args: Unet.utils.Args, default None
        max_train_samples: int, default 40000 (approximately 600 images). Number of PATCHES 
          (not records) to use for training and validation. Set to False to use all available 
          patches.
    """
    if verbose:
        print("Training U-net model...")
    if not args: # use default args if none are provided
        args = Unet.utils.Args()
    
    patchsize = constants.PATCH_SIZE
    # path for model checkpoints, used with early stopping or to resume training later
    CHK_PATH_UNET = os.path.join(model_folder, ckpt_name + "_" + str(patchsize))
    # for saving the loss values, used with early stopping
    LOSS_PATH = os.path.join(model_folder, ckpt_name + "_" + str(patchsize) + '_losses')
    # if we're loading a pretrained model - hardcoded for now
    LOAD_PATH_UNET = None
    if warm_start:
        chkpt_path = os.path.join('digitization', 'model_checkpoints', 
                                      ckpt_name + "_"+ str(patchsize) + '_checkpoint')
        args.patience = 5 # decrease patience if using a pretrained model
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
    if verbose:
        print("...Done.")

    return unet_model


def reconstruct_signal(record, unet_image, rois, header_txt, 
                       reconstructed_signals_folder, save_signal=True, is_real_image=False):
    """
    reconstruct signals from u-net outputs

    Returns:
        reconstructed_signal: pandas dataframe, reconstructed signal
        raw_signals: np.array, raw signals in pixel coords
        gridsize: float, scaling factor for the signals in pixel units
    """
    signal_length = helper_code.get_num_samples(header_txt)
    fs = helper_code.get_sampling_frequency(header_txt)
    max_duration = int(signal_length/fs)
    # max duration on images cannot exceed 10s as per Challenge team
    max_duration = 10 if max_duration > 10 else max_duration 
    try:
        reconstructed_signal, raw_signals, gridsize  = ECGminer.digitize_image_unet(unet_image, 
                                        rois, sig_len=signal_length, max_duration=max_duration,
                                        is_generated_image=not is_real_image)
    except SignalExtractionError as e:
        print(f"Error in digitizing signal: {e}")
        return None, None, None
    reconstructed_signal = np.asarray(reconstructed_signal) # note still may have NaNs

    # save reconstructed signal and copied header file in the same folder
    if save_signal:
        output_record_path = os.path.join(reconstructed_signals_folder, record)
        helper_code.save_header(output_record_path, header_txt)
        comments = [l for l in header_txt.split('\n') if l.startswith('#')]
        signal = np.nan_to_num(reconstructed_signal) # cast any NaNs to 0
        helper_code.save_signals(output_record_path, signal, comments)

    # return raw_signals and gridsize for external evaluation
    return reconstructed_signal, raw_signals, gridsize


def train_classification_model(records_folder, verbose, records_to_process=None):
    """
    Extracts features and labels from headers, then one-hot encodes labels and trains the
    SE-ResNet model.
    """
    if not records_to_process:
        records_to_process = helper_code.find_records(records_folder)

    all_data = []
    labels = []
    for record in tqdm(records_to_process, desc='Loading classifier training data', 
                       disable=not verbose):
        data, label = classification.get_training_data(record, records_folder)
        if label is None or '' in label: # don't use data without labels for training
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

    if verbose:
        print("Training SE-ResNet classification model...")

    resnet_model = {}
    n_models = constants.RESNET_ENSEMBLE
    num_epochs = constants.RESNET_EPOCHS
    for i in range(n_models):
        resnet_model[f'res{i}'] = seresnet18.train_model(
                                    all_data, multilabels, uniq_labels, verbose, epochs=num_epochs, 
                                    validate=False)
    if verbose:
        print("Finished training classification model.")
    
    return resnet_model, uniq_labels  


def unet_reconstruct_single_image(record, digitization_model, verbose, delete_patches=True):
    """
    params
        record: str, relative path from data folder and record ID, 
            e.g. 'ptbl-xl/records500/01017_hr'
        model: U-net state dict
        verbose: bool
        delete_patches: bool, whether to delete patches after processing    
    """
    # get image from image_path
    image_path = team_helper_code.load_image_paths(record)[0]
    record_id = os.path.split(record)[-1].split('.')[0]

    # load models
    yolo_model = digitization_model['yolov7']
    unet_generated = digitization_model['unet_generated']
    unet_real = digitization_model['unet_real']
    classification_model = digitization_model['image_classifier']

    # hard code some folder paths for now
    patch_folder = os.path.join('temp_data', 'test', 'patches', 'image_patches')
    reconstructed_signals_folder = os.path.join('temp_data', 'test', 'reconstructed_signals')
    os.makedirs(patch_folder, exist_ok=True)
    os.makedirs(reconstructed_signals_folder, exist_ok=True)
    os.makedirs(os.path.join("temp_data", "test", "images"), exist_ok=True)

    # load header file to save with reconstructed signal
    header_txt = helper_code.load_header(record)

    # get bounding boxes/ROIs using YOLOv7
    args = digitization.YOLOv7.detect.OptArgs()
    args.device = "0"
    args.source = image_path
    args.nosave = True # set False for testing to save images with ROIs
    rois = digitization.YOLOv7.detect.detect_single(yolo_model, args, verbose)

    # resize image if it's very large
    with open(image_path, 'rb') as f:
        image = Image.open(f)
        if image.size[0] > 2300: # index 0 is width
            ratio = 2300/image.size[0]
            new_height = int(ratio*image.size[1])
            image = image.resize((2200, new_height), Image.Resampling.LANCZOS)
        image = np.array(image)

    # patchify image
    Unet.patching.save_patches_single_image(record_id, image, None, 
                                            patch_size=constants.PATCH_SIZE,
                                            im_patch_save_path=patch_folder,
                                            lab_patch_save_path=None)

    # run classifier
    is_real_image = preprocessing.classifier.classify_image(record_id, patch_folder, 
                                                            classification_model, verbose)
    
    if is_real_image:
        unet_model = unet_real
    else:
        unet_model = unet_generated
    # predict on patches, recover u-net output image
    predicted_mask, entropy = Unet.predict_single_image(record_id, patch_folder, unet_model,
                                                original_image_size=image.shape[:2])
    
    # rotate reconstructed u-net output to original orientation
    rotated_mask, rotated_image_path, rot_angle = preprocessing.column_rotation(record_id, 
                                                    predicted_mask, image,
                                                    angle_range=(-20, 20), verbose=verbose)
    
    # save rotated mask for debugging
    # with open(os.path.join("temp_data", "test", "unet_outputs", record_id + '.png'), 'wb') as f:
    #     plt.imsave(f, rotated_mask, cmap='gray')
    
    if rot_angle != 0 and constants.ATTEMPT_ROTATION: # currently just rotate the mask, do no re-predict   
        print(f"Rotation angle detected: {rot_angle}")
        args.source = rotated_image_path
        rois = digitization.YOLOv7.detect.detect_single(yolo_model, args, True)
        reconstructed_signal, raw_signals, gridsize = reconstruct_signal(record_id, rotated_mask, 
                                                    rois,
                                                    header_txt,
                                                    reconstructed_signals_folder, 
                                                    save_signal=True,
                                                    is_real_image=is_real_image)
        if reconstructed_signal is None: # in that case try it with the original (non-rotated) mask
            print(f"Error reconstructing signal after rotating image {image_path}, "+\
                  "trying with original mask.")
            reconstructed_signal, raw_signals, gridsize = reconstruct_signal(record_id, unet_image,
                                                    rois, 
                                                    header_txt,
                                                    reconstructed_signals_folder, 
                                                    save_signal=True,
                                                    is_real_image=is_real_image)   
        else: unet_image = rotated_mask # to save later, optional          
    else: # no rotation needed
        reconstructed_signal, raw_signals, _ = reconstruct_signal(record_id, predicted_mask, 
                                                     rois,
                                                     header_txt,
                                                     reconstructed_signals_folder, 
                                                     save_signal=True,
                                                     is_real_image=is_real_image)

    # optional: delete patches
    if delete_patches:
        for im in os.listdir(patch_folder):
            os.remove(os.path.join(patch_folder, im))

    # return reconstructed signal
    return reconstructed_signal, reconstructed_signals_folder, is_real_image


def classify_signals(record_path, data_folder, resnet_model, classes, verbose):
    record_id = os.path.split(record_path)[-1].split('.')[0]
    data = [classification.get_testing_data(record_id, data_folder)]
    # deal with the ensemble
    if type(resnet_model) == dict:
        for i, val in enumerate(resnet_model):
            _, probabilities = seresnet18.predict_proba(
                                                resnet_model[val], data, classes, verbose)
            if i == 0:
                probs = probabilities
            else:
                probs = probs + probabilities
        
        # hacky way to get the mean of all the resnets
        probs = probs / len(resnet_model)
        
        pred_dx = multiclass_predict_from_logits(classes, probs)
        labels = classes[np.where(pred_dx == 1)]
    
    # single model output
    else:
        # wrap in list to match training data format
 
        pred_dx, probabilities = seresnet18.predict_proba(
                                            resnet_model, data, classes, verbose)
        labels = classes[np.where(pred_dx == 1)]
    
    if verbose:
        print(f"Classes: {classes}, probabilities: {probabilities}")
        print(f"Predicted labels: {labels}")

    return labels
