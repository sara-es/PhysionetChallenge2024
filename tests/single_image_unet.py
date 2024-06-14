import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
import matplotlib.pyplot as plt

from digitization.Unet import Unet

image = plt.imread("G:\\PhysionetChallenge2024\\tiny_testset\\real_images\\ecg00051.png")
record_id = "ecg00051"
patch_size = 256
image_size = image.shape
im_patch_save_path = "G:\\PhysionetChallenge2024\\tests\\data\\patches\\image_patches"
model_folder = "G:\\PhysionetChallenge2024\\model"

Unet.patching.save_patches_single_image(record_id, image, None, 
                                        patch_size, im_patch_save_path, 
                                        None)

model = Unet.utils.load_unet_from_state_dict(model_folder)

predicted_image = Unet.predict_single_image(record_id, im_patch_save_path, model, 
                                            original_image_size=image_size)

plt.imshow(predicted_image)
plt.show()