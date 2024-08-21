# model training
PATCH_SIZE = 256 # Size of the u-net patches in pixels - assumes square patches
YOLO_CONFIG = 'yolov7-ecg-1c' # one or two classes: 'yolov7-ecg-1c' or 'yolov7-ecg-2c'
WARM_START = False # if True, use pre-trained model weights to start training
DELETE_DATA = True # delete all generated data after training or inference
MAX_GENERATE_IMAGES = 3000 # maximum number of images to generate

# digitization / vectorization
YOLO_BOUNDING = True # use YOLO for cropping ECG image before line tracing
YOLO_ROIS = True # use YOLO bounding boxes to get ROIs. Needs YOLO_CONFIG = 'yolov7-ecg-1c'
CHECK_CABRERA = False # flag to check for Cabrera format

# SNR hacks
RETURN_SIGNAL_IF_REAL = True # return SNR if real ECG