import numpy as np
import cv2
from digitization.ECGminer.assets.Image import Image
from digitization.ECGminer.assets.Point import Point
from digitization.ECGminer.assets.Rectangle import Rectangle
from utils import constants

def get_ECG_rows(yolo_boxes, sens=12):
    yolo_boxes = np.array(yolo_boxes)
    # n.b. to generalise this, we need to work out the number of short lead rows - we could do this by merging rows in the x direction.
    if yolo_boxes.size == 0: # no bounding boxes; guess 3 rows
        return 3, [0, 1, 0, 1]
    # get total bounding box
    left = min(yolo_boxes[:, 1] - yolo_boxes[:, 3]/2)
    right = max(yolo_boxes[:, 1] + yolo_boxes[:, 3]/2)
    top = min(yolo_boxes[:, 2] - yolo_boxes[:, 4]/2)
    bottom = max(yolo_boxes[:, 2] + yolo_boxes[:, 4]/2)
   
    # add buffer to top (~5% of height)
    top = max(0, top - 0.05)

    # note bounding box is in percentage of image size
    bounding_box = [left, right, top, bottom]
    
    if max(yolo_boxes[:,0]) > 0: # using 2 classes or more yolo version
        # long_leads is subset of bounding boxes with long leads:
        long_leads = yolo_boxes[yolo_boxes[:,0]==1]
        vrange = bottom-top
        thresh = vrange / sens #assume there are 6 rows maximum, and we want to find half of that distance as a threshold

        #get the number of long leads
        if long_leads.shape[0] > 1:
            # merge any boxes that are on similar rows
                long_roi = long_leads[:,2]
                long_roi = np.sort(long_roi)
                diffs = np.diff(long_roi)
                diffs = diffs < thresh
                num_long_leads = sum(diffs==False) + 1
        else:
            num_long_leads = sum(yolo_boxes[:,0])
    
        num_rows = 3 + num_long_leads #simplifying challenge assumption that the short leads are in 3x4 format.
        # n.b. to generalise this, we need to work out the number of short lead rows - we could do this by merging rows in the x direction.
    else: # using 1 class yolo version
        num_rows = len(yolo_boxes)
    
    return int(num_rows), bounding_box


def binarize_image(image):
    # Invert the colours
    img_arr = abs(np.array(image) - 1)*255
    img_arr = img_arr.astype(np.uint8)
    # binarize using Otsu's method
    thresh, image_arr = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # convert to ECGminer Image object
    image = cv2.merge([image_arr, image_arr, image_arr])
    image = np.uint8(image)
    image = Image(image)
    return image


def crop_image(image, box):
    # threshold and binarize
    image = binarize_image(image) # returns Image object
    height = image.height
    width = image.width
    #replace preprocessor with our own cropper
    #convert [left, right, top, bottom] into pixels
    pad = 20 #bounding box is pretty good, so let's add a just a small amount of padding
    box[0] = int(np.round(box[0] * width)) - pad
    box[1] = int(np.round(box[1] * width)) + pad
    box[2] = int(np.round(box[2] * height)) - pad
    box[3] = int(np.round(box[3] * height)) + pad
    
    # sanity check just in case bounding box was near edge of image
    if box[0] < 0 : box[0] = 0
    if box[1] > width : box[1] = width
    if box[2] < 0 : box[2] = 0
    if box[3] < height : box[3] = height

    if constants.YOLO_BOUNDING:
        rect = Rectangle(Point(box[0], box[2]), Point(box[1], box[3])) # cropped image from yolo
    else:
        rect = Rectangle(Point(0, 350), Point(width, height))

    image.crop(rect)
    image.to_GRAY()

    return image, rect