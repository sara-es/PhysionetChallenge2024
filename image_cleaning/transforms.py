"""
Transforms and scaling functions for image processing
"""
import numpy as np
from skimage.morphology import closing


def close_filter(image, fp):
    # morphological filter on red channel? (closing?)
    aa = footprint=[(np.ones((fp, 1)), 1), (np.ones((1, fp)), 1)]

    test = closing(image, footprint=[(np.ones((fp, 1)), 1), (np.ones((1, fp)), 1)])
    output_im = image - test
    return output_im


def std_rescale(image, contrast=1):
    """
    Standardizes to between [-contrast, contrast] regardless of range of input
    """
    ptp_ratio = contrast / np.ptp(image)
    shift = (np.max(image) + np.min(image)) / contrast  # works with negative values
    return ptp_ratio * (image - shift)


def norm_rescale(image, contrast=1):
    """
    Normalizes to zero mean and scales to have one of (abs(max), abs(min)) = contrast
    """
    scale = np.max(np.abs([np.max(image), np.min(image)]))
    return contrast * (image - np.mean(image)) / scale


def zero_one_rescale(image):
    """
    Rescales to between 0 and 1
    """
    return (image - np.min(image)) / np.ptp(image)


def sigmoid(x):
    """
    logistic sigmoid function
    """
    return 1. / (1. + np.exp(-x))


def sigmoid_gen(x, k, x_0):
    """
    sigmoid function with slope (k) and midpoint (x_0) adjustments
    """
    return 1. / (1. + np.exp(-k * (x - x_0)))


def bresenham_line(x1, x2, y1, y2):
    """
    Bresenham's line algorithm
    Currently has same behavior as skimage.draw.line (returns (2, N) instead of (N, 2)
    For single sets of pixel coordinates atm - probably could be vectorized
    """

    # if slope is greater than 1, swap x and y coordinates
    rotate = abs(y2 - y1) > abs(x2 - x1)
    if rotate:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    reverse_x = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        reverse_x = True

    dx = x2 - x1
    dy = abs(y2 - y1)
    error = dx / 2
    ystep = 1 if y1 < y2 else -1

    y = y1
    line = np.zeros((dx + 1, 2))
    for i, x in enumerate(range(x1, x2)):
        line[i] = (np.array([y, x]) if rotate else np.array([x, y]))
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    if reverse_x:
        line.reverse()

    return line
