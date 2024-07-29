import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import matplotlib.pyplot as plt
from utils import model_persistence
from digitization.Unet import Unet
import team_code

# # image = plt.imread("G:\\PhysionetChallenge2024\\tiny_testset\\real_images\\ecg00051.png")
# image = plt.imread("temp_data\\images\\01017_lr-0.png")
# record_id = "ecg00051"
# patch_size = 256
# image_size = image.shape
# im_patch_save_path = "G:\\PhysionetChallenge2024\\temp_data\\patches"
# model_folder = "G:\\PhysionetChallenge2024\\model"

# Unet.patching.save_patches_single_image(record_id, image, None, 
#                                         patch_size, im_patch_save_path, 
#                                         None)

model_folder = "G:\\PhysionetChallenge2024\\model"
models = model_persistence.load_models(model_folder, verbose=True, 
                        models_to_load=['digitization_model', 'classification_model', 'dx_classes'])
digitization_model = models['digitization_model']
unet_model = Unet.utils.load_unet_from_state_dict(digitization_model)

unet_model = model_persistence.load_checkpoint_dict("model", 
                                                        "UNET_256_checkpoint", True)

record = "tiny_testset/hr_hidden/21544_hr"
# record = "test_data/images/00001_hr"
# records_to_process = [record]

signal, reconstructed_signal_dir = team_code.unet_reconstruct_single_image(record, unet_model, verbose=True, 
                                                                 delete_patches=False)

# plt.imshow(predicted_image)
# plt.show()