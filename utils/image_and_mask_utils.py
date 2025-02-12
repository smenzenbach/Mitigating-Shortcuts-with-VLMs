from math import e
import cv2
import numpy as np
from PIL import Image


def dilate_mask(
    mask,
    iterations=1,
    ratio=60,
):
    """
    Dilates a binary mask using a kernel based on the mask size with the number of iterations.

    Parameters:
    ----------
    mask : PIL.Image.Image or np.ndarray
        The binary mask to dilate.
    iterations : int
        The number of iterations to apply dilation.
    ratio : int
        The ratio of the kernel size to the mask size.

    Returns:
    -------
    PIL.Image.Image or np.ndarray
        The dilated mask in the same format as the input.
    """
    is_pil = isinstance(mask, Image.Image)
    mask = np.array(mask, dtype=np.uint8)
    kernel_size_w = max(1, mask.shape[1] // ratio)
    kernel_size_h = max(1, mask.shape[0] // ratio)
    kernel_size = (kernel_size_w, kernel_size_h)
    kernel = np.ones(kernel_size, np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    # ensure its binary
    dilated_mask = np.where(dilated_mask < 128, 0, 255)
    if is_pil:
        return Image.fromarray(dilated_mask).convert("L")
    else:
        return dilated_mask
