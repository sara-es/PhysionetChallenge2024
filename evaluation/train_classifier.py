import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import joblib
import team_code, helper_code
from sklearn.utils import shuffle
from digitization import Unet
from preprocessing import classifier
from utils import model_persistence, constants


def train_unet(real_data_folder, model_folder, verbose, max_size_training_set=150):
    """
    train_digitization_model from team_code, but only the classifier bits
    """
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

    data_folder = os.path.join(os.getcwd(), 'ptb-xl', 'records500')
    records_to_process = helper_code.find_records(data_folder)
    if max_size_training_set is not None:
        records_to_process = shuffle(records_to_process, random_state=42)[:max_size_training_set]

    # generate images, bounding boxes, and masks for training YOLO and u-net
    # note that YOLO labels assume two classes: short and long leads
    # team_code.generate_training_images(data_folder, gen_images_folder, 
    #                          gen_masks_folder, bb_labels_folder, 
    #                          verbose, records_to_process=records_to_process)
    
    # # Generate patches for u-net. Note: this deletes source images and masks to save space
    # # if delete_training_data is True
    # Unet.patching.save_patches_batch(records_to_process, gen_images_folder, gen_masks_folder, 
    #                                  constants.PATCH_SIZE, gen_patch_folder, verbose, 
    #                                  delete_images=False)

    # Find the data files.
    real_images_folder = os.path.join(real_data_folder, 'images')
    real_masks_folder = os.path.join(real_data_folder, 'masks')
    real_patch_folder = os.path.join('temp_data', 'train', 'real_patches')
    os.makedirs(real_patch_folder, exist_ok=True)
    # check that real images and masks are available
    if not os.path.exists(real_images_folder) or not os.path.exists(real_masks_folder):
        print(f"Real images or masks not found in {real_data_folder}, unable to train " +\
                "real image classifier or u-net.")
    # gen_img_patch_dir = os.path.join(gen_patch_folder, 'image_patches')
    real_records = os.listdir(real_images_folder)
    real_records = [r.split('.')[0] for r in real_records]
    
    # Unet.patching.save_patches_batch(real_records, real_images_folder, real_masks_folder, 
    #                                  constants.PATCH_SIZE, real_patch_folder, verbose, 
    #                                  delete_images=False, require_masks=False)
    # train classifier for real vs. generated data
    unet_classifier = classifier.train_image_classifier(real_patch_folder, gen_patch_folder, model_folder, 
                                      constants.PATCH_SIZE, verbose)

    # train u-net
    # args = Unet.utils.Args()
    # args.epochs = 500
    # args.augmentation = True
    # unet_model = team_code.train_unet(train_records, patch_folder, model_folder, verbose, 
    #                      args=args, warm_start=True, max_train_samples=30000, delete_patches=False)

    # save trained u-net
    model_persistence.save_model_torch(unet_classifier, 'image_classifier', model_folder)


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "real_images")
    model_folder = os.path.join(os.getcwd(), "model")
    verbose = True

    train_unet(data_folder, model_folder, verbose)
