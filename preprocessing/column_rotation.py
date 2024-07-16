import numpy as np
import scipy as sp
import skimage

def column_rotation(record_id, image_mask, angle_range=(-45, 45), verbose=True):
    """
    Uses the u-net output to find the rotation of the image, rotates the image and returns the
    rotated image and the rotation angle. No changes are currently made to the image itself.
    """
    # angle range to search - can limit this to speed up the search
    min_angle = angle_range[0]
    max_angle = angle_range[1]
    rot_angle = 1000 # set to high number to start - will be within angle_range

    # closing filter on the image to remove noise
    test_im = skimage.morphology.closing(image_mask, footprint=[(np.ones((5, 1)), 1), (np.ones((1, 5)), 1)])

    n_active_cols = 5000

    for angle in range(min_angle, max_angle): 
        # rotate image by angle, make sure no extra lines at borders
        rot_image = sp.ndimage.rotate(test_im, angle, axes=(1, 0), reshape=False)
        rot_image[:, 0] = 0
        rot_image[:, -1] = 0
        rot_image[0, :] = 0
        rot_image[-1, :] = 0
        col_hist = np.sum(rot_image, axis = 0) #sum each column 

        # find the starting and end column - columns with black pixels within the active region
        idxs = np.sum(col_hist > 0)
        if idxs < n_active_cols:
            n_active_cols = idxs
            rot_angle = angle

    rotated_image = sp.ndimage.rotate(image_mask, rot_angle, axes=(1, 0), reshape=True)
    return rotated_image, rot_angle