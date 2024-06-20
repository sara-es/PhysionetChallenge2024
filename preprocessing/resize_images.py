# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:06:12 2024

function to rescale images to be roughly the same resolution as the training images used to train the u-net

@author: hssdwo
"""
from PIL import Image


def resize_images(images, base_width = 4000, base_height = 1700, thresh = 0.8):
    for idx, image in enumerate(images):
        images[idx] = resize_image(image, base_width, base_height, thresh)
    
    return images

def resize_image(image, base_width = 4000, base_height = 1700, thresh = 0.8):
    """
    Same as previous resize_images but for only one image
    TODO this was previous image.size[0] but .size returns an int. you can either do image.shape[0]
    (which is a tuple,) or image.size (which is an int, total number of pixels in the image)
    """
    if (image.shape[0] < (thresh * base_width)) | (image.shape[0] > ((2-thresh) * base_width)):
        wpercent = (base_width / float(image.shape[0]))
        hsize = int((float(image.shape[1]) * float(wpercent)))
        image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)
    elif (image.shape[0] < (thresh * base_height)) | (image.shape[0] < ((2-thresh) * base_height)):
        wpercent = (base_height / float(image.shape[1]))
        wsize = int((float(image.shape[0]) * float(wpercent)))
        image = image.resize((wsize, base_height), Image.Resampling.LANCZOS)
        
    return image