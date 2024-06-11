# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:06:12 2024

function to rescale images to be roughly the same resolution as the training images used to train the u-net

@author: hssdwo
"""
from PIL import Image


def resize_images(images, base_width = 4000, base_height = 1700, thresh = 0.8):
    for idx, image in enumerate(images):
        if (image.size[0] < (thresh * base_width)) | (image.size[0] > ((2-thresh) * base_width)):
            wpercent = (base_width / float(image.size[0]))
            hsize = int((float(image.size[1]) * float(wpercent)))
            images[idx] = image.resize((base_width, hsize), Image.Resampling.LANCZOS)
        elif (image.size[0] < (thresh * base_height)) | (image.size[0] < ((2-thresh) * base_height))
            wpercent = (base_height / float(image.size[1]))
            wsize = int((float(image.size[0]) * float(wpercent)))
            images[idx] = image.resize((wsize, base_height), Image.Resampling.LANCZOS)
        
    return images