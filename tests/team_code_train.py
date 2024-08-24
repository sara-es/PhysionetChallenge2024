import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
from utils import model_persistence



import helper_code
from utils import constants, model_persistence
from digitization import Unet
from classification import seresnet18
from classification.utils import multiclass_predict_from_logits
import generator, preprocessing, digitization, classification
from preprocessing import classifier
import generator.gen_ecg_images_from_data_batch
from evaluation import eval_utils
from digitization.ECGminer.assets.DigitizationError import SignalExtractionError


def train_models(data_folder, model_folder, verbose, max_size_training_set=4000, delete_training_data=False,
                 real_data_folder='real_images'):
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
    
    digitization_model = dict()
    
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

    if max_size_training_set is not None:
        records_to_process = shuffle(records, random_state=42)[:max_size_training_set]

    # generate images, bounding boxes, and masks for training YOLO and u-net
    # note that YOLO labels assume two classes: short and long leads
    team_code.generate_training_images(data_folder, gen_images_folder, 
                             gen_masks_folder, bb_labels_folder, 
                             verbose, records_to_process=records_to_process)
    
    # train YOLOv7 
    team_code.train_yolo(records_to_process, gen_images_folder, bb_labels_folder, model_folder,
               verbose, delete_training_data=delete_training_data)
    
    print("Finished training YOLO model.")
    
    return
    
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
        digitization_model['image_classifier'] = classifier.train_image_classifier(real_patch_folder, 
                                        gen_patch_folder, 
                                        model_folder, constants.PATCH_SIZE, verbose)
        
        # SAVE MODEL
        team_code.save_models(model_folder, digitization_model, None, None)

    args = Unet.utils.Args()
    args.train_val_prop = 0.8
    args.epochs = constants.UNET_EPOCHS
    if real_images_folder is not None:
        # train U-net: real data
        if verbose:
            print("Training U-net for real data...")
        digitization_model['unet_real'] = team_code.train_unet(real_records, real_patch_folder, model_folder, verbose, args=args,
                            warm_start=constants.WARM_START, ckpt_name='unet_real')
        team_code.save_models(model_folder, digitization_model, None, None)

    # train U-net: generated data
    if verbose: 
        print("Training U-net for generated data...")
    digitization_model['unet_generated'] = team_code.train_unet(records_to_process, gen_patch_folder, model_folder, verbose,
                             args=args, warm_start=constants.WARM_START, ckpt_name='unet_gen')
    team_code.save_models(model_folder, digitization_model, None, None)



if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "ptb-xl", "records500")
    # data_folder = os.path.join("temp_data", "train", "images")
    model_folder = os.path.join(os.getcwd(), "model")
    verbose = True

    train_models(data_folder, model_folder, verbose)
