from PIL import Image
from utils import dilate_mask


def inpainting_preprocessing(image_path, mask_path, iterations=4):
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").point(lambda x: 255 if x > 128 else 0)
    dilated_mask = dilate_mask(mask, iterations=iterations)
    dilated_mask = dilated_mask.resize(img.size)
    return img, dilated_mask


def get_empty_mask(image_path):
    img = Image.open(image_path).convert("RGB")
    return Image.new("L", img.size, 0)


def get_full_mask(image_path):
    img = Image.open(image_path).convert("RGB")
    return Image.new("L", img.size, 255)
