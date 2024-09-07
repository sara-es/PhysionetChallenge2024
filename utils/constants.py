# model training
PATCH_SIZE = 256 # Size of the u-net patches in pixels - assumes square patches
YOLO_N_CLASSES = 1 # number of classes for YOLO: 2 distinguishes between long and short leads
YOLO_CONFIG = 'yolov7-ecg-' + str(YOLO_N_CLASSES) + 'c' # one or two classes: 'yolov7-ecg-1c' or 'yolov7-ecg-2c'
WARM_START = False # if True, resume previous training 
DELETE_DATA = True # delete all generated data after training or inference
MAX_GENERATE_IMAGES = 3000 # maximum number of images to generate
RESNET_ENSEMBLE = 3 # number of ResNet models to ensemble, max 5
ALLOW_MULTIPROCESSING = True # speeds up training, but may cause issues in docker

# training epochs
YOLO_EPOCHS = 300 # number of epochs for YOLO training
RESNET_EPOCHS = 150 # number of epochs for ResNet training
UNET_EPOCHS = 500 # number of epochs for U-Net training - has early stopping, so can be high

# digitization / vectorization
YOLO_BOUNDING = True # use YOLO for cropping ECG image before line tracing
YOLO_ROIS = True # use YOLO bounding boxes to get ROIs. Needs YOLO_CONFIG = 'yolov7-ecg-1c'
CHECK_CABRERA = False # flag to check for Cabrera format
ATTEMPT_ROTATION = True # attempt to detect rotation of the image

# classification
MULTILABEL_THRESHOLD = 0.35 # threshold for multiclass classification

# SNR hacks
RETURN_SIGNAL_IF_REAL = True # return SNR if real ECG