# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:01:48 2024

@author: hssdwo
"""
from digitization.ECGminer.digitize_image import get_ECG_rows
import numpy as np
file = "C:/Users/hssdwo/Downloads/yolo_boxes/00841_hr.npy"

# load yolo output
data = np.load(file)

yolo_boxes = np.array(data)

# artificially add another long lead
yolo_boxes = np.vstack([yolo_boxes, yolo_boxes[0]])

# get total bounding box
left = min(yolo_boxes[:, 1] - yolo_boxes[:, 3]/2)
right = max(yolo_boxes[:, 1] + yolo_boxes[:, 3]/2)
top = min(yolo_boxes[:, 2] - yolo_boxes[:, 4]/2)
bottom = max(yolo_boxes[:, 2] + yolo_boxes[:, 4]/2)


# long_leads is subset of bounding boxes with long leads:
long_leads = yolo_boxes[yolo_boxes[:,0]==1]
vrange = bottom-top
thresh = vrange / (6*2) #assume there are 6 rows maximum, and we want to find half of that distance as a threshold

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



# note bounding box is in percentage of image size
bounding_box = [left, right, top, bottom]