import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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
        return Image.fromarray(dilated_mask.astype(np.uint8)).convert("L")
    else:
        return dilated_mask


def get_empty_mask(image_path):
    img = Image.open(image_path).convert("RGB")
    return Image.new("L", img.size, 0)


def get_full_mask(image_path):
    img = Image.open(image_path).convert("RGB")
    return Image.new("L", img.size, 255)


def mask_on_image(image, segmentation_map):
    """
    Overlay a segmentation map on an image.

    Parameters:
    ----------
    image : PIL.Image.Image or np.ndarray
        The base image.
    segmentation_map : PIL.Image.Image or np.ndarray
        The segmentation map to overlay on the image.

    Returns:
    -------
    PIL.Image.Image or np.ndarray
        The image with the segmentation map overlayed in the same format as the image.
    """
    if isinstance(image, Image.Image):
        pil = True
        # Convert images to numpy arrays
        image_np = np.array(image)
    else:
        pil = False
        image_np = image

    if isinstance(segmentation_map, Image.Image):
        segmentation_map_np = np.array(segmentation_map)
    else:
        segmentation_map_np = segmentation_map

    # Ensure the segmentation map is in grayscale
    if len(segmentation_map_np.shape) == 3 and segmentation_map_np.shape[2] == 3:
        segmentation_map_np = cv2.cvtColor(segmentation_map_np, cv2.COLOR_RGB2GRAY)

    # Ensure the images have the same shape
    if image_np.shape[:2] != segmentation_map_np.shape[:2]:
        raise ValueError(
            "The base image and segmentation map must have the same dimensions."
        )

    # Binarize the segmentation map
    segmentation_map_np = np.where(segmentation_map_np > 128, 255, 0).astype(np.uint8)

    # Create a colored overlay for the segmentation map
    colored_overlay = np.zeros((*segmentation_map_np.shape, 3), dtype=np.uint8)
    colored_overlay[segmentation_map_np == 255] = [
        255,
        0,
        255,
    ]  # Pink color for the segmentation

    # Ensure the base image has 3 channels (convert grayscale to RGB if needed)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Combine the images with some transparency for the overlay
    alpha = 0.6  # Transparency factor
    combined_image = cv2.addWeighted(image_np, 1, colored_overlay, alpha, 0)

    if pil:
        # Convert the result back to a PIL image
        return Image.fromarray(combined_image)
    else:
        return combined_image


def plot_grid(figures, label=None, ncols=3, title=None, save_path=None, show=True):
    """
    Plot a grid of images with their corresponding labels.

    Parameters:
    ----------
    figures : list
        A list of images to plot.
    label : list, Optional
        A list of labels for each image.
    ncols : int, Optional
        The number of columns in the grid.
    title : str, Optional
        The title of the plot.
    save_path : str, Optional
        The path to save the plot.
    """
    # Define the number of columns for the grid
    num_rows = (
        len(figures) + ncols - 1
    ) // ncols  # Calculate the number of rows needed

    # Create a figure and a set of subplots
    if ncols > 4:
        fig, axs = plt.subplots(num_rows, ncols, figsize=(15, 2 * num_rows))
    else:
        fig, axs = plt.subplots(num_rows, ncols, figsize=(15, 5 * num_rows))

    # Flatten the axes array for easy iteration
    axs = axs.flatten()

    # Plot each figure in the grid
    for i, figure in enumerate(figures):
        axs[i].imshow(figure)
        axs[i].axis("off")  # Turn off the axis
        if label is not None:
            axs[i].set_title(label[i])

    # Turn off any remaining empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    if title is not None:
        plt.suptitle(title)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        # Show the grid of figures
        plt.tight_layout()
        plt.show()

    # Close the figure to free up memory
    plt.close(fig)
