import scipy as sp
from preprocessing import transforms


def single_channel_sigmoid(red_im, angle):
    """
    Simple function to remove shadows - room for much improvement.
    The input image is called red_im, but really this can be any single channel image,
    ideally without grid lines. 
    """
    output_im = transforms.close_filter(red_im, 8)  # this removes the shadows
    output_im0 = transforms.close_filter(red_im, 2)  # this removes the filter artifacts

    sigmoid_norm1 = 255 * transforms.sigmoid(transforms.norm_rescale(output_im - 0.95 * output_im0, contrast=8))
    sigmoid_std1 = 255 * transforms.sigmoid(transforms.std_rescale(output_im - 0.95 * output_im0, contrast=8))

    # feel like we can combine these somehow to be useful?
    combo1 = -(sigmoid_norm1 - sigmoid_std1)  # this is really a hack - room for much improvement

    greyscale_out = transforms.zero_one_rescale(
        sp.ndimage.rotate(combo1, angle, axes=(1, 0), reshape=True, cval=combo1.mean()))
    cleaned_image = transforms.sigmoid_gen(greyscale_out, 10/255, 100/255)
    cleaned_image = 255 * transforms.zero_one_rescale(cleaned_image)
    return cleaned_image