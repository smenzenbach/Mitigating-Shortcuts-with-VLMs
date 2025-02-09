import os
import toml
import yaml
import pandas as pd
import numpy as np
import h5py
from PIL import Image
from torchvision.models import VGG16_Weights

from utils import get_config_id_from_path, create_file_names_list, dilate_mask

# get dir of repo (parent dir)
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Dataset:
    def __init__(self, config_path):
        # load config and get config id
        self.config = toml.load(config_path)
        self.config_id = get_config_id_from_path(config_path)
        # get dataset name and yaml for raw and gt paths
        self.dataset_name = self.config["dataset"]["name"]
        with open(
            f"{repo_dir}/datasets.yaml", "r"
        ) as file:  # adapt path later if repo is ready (without worksopace/home)
            dataset_yaml = yaml.safe_load(file)
        # set raw and gt paths
        self.raw_path = dataset_yaml[self.dataset_name]["raw_path"]
        self.gt_wr_seg_path = dataset_yaml[self.dataset_name]["gt_wr_seg_path"]
        self.gt_rr_seg_path = dataset_yaml[self.dataset_name]["gt_rr_seg_path"]
        # get gt metadata
        gt_label_path = dataset_yaml[self.dataset_name]["gt_label_path"]
        self.metadata = pd.read_csv(
            gt_label_path
        )  # columns: filename, class_label, confounder_label

        # set data path of pipeline output
        self.seg_path = (
            f"{repo_dir}/pipeline_output/{self.dataset_name}/{self.config_id}/seg"
        )
        self.inpainted_path = (
            f"{repo_dir}/pipeline_output/{self.dataset_name}/{self.config_id}/inpainted"
        )

    def load_image(self, img_path, image_size=None):
        """Loads an image using PIL and converts it to a NumPy array."""
        img = Image.open(img_path).convert("RGB")
        if image_size is not None:
            img = img.resize(self.image_size, Image.LANCZOS)  # High-quality resize
        return np.array(img, dtype=np.float32)

    def load_mask(self, mask_path, mask_size=None):
        """Loads a mask using PIL and converts it to a NumPy array."""
        mask = Image.open(mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
        # Dilate the mask as specified in the config
        kernel_size = self.config["inpainting"]["dilation"]["kernel_size"]
        iterations = self.config["inpainting"]["dilation"]["iterations"]
        if (
            "width" in self.config["inpainting"]["dilation"]
            and "height" in self.config["inpainting"]["dilation"]
        ):
            dilation_size = (
                self.config["inpainting"]["dilation"]["width"],
                self.config["inpainting"]["dilation"]["height"],
            )
        else:
            dilation_size = (299, 299)
        mask = dilate_mask(
            mask, kernel_size=kernel_size, iterations=iterations, size=dilation_size
        )
        # resize the mask if specified
        if mask_size is not None:
            mask = mask.resize(mask_size, Image.LANCZOS)
        return np.array(mask, dtype=np.float32)

    def preprocess_data(self, images, model_name="vgg16"):
        if model_name == "vgg16":
            transform = VGG16_Weights.IMAGENET1K_V1.transforms()
        images = transform(images)
        return images

    def load_data(self, model_name=None):
        """Loads the raw images, inpainted images, ground truth masks, masks, class labels, and confounder labels. If model name is given the data gets preprocessed accordingly."""
        # initialize lists
        (
            raw_images,
            images,
            gt_wr_masks,
            gt_rr_masks,
            masks,
            class_label,
            confounder_label,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # get desired size according to model (if model name is given)
        if model_name:
            if model_name == "vgg16":
                size = (224, 224)  # or (299, 299)?
        else:
            size = None

        # get list of inpainted files (to determine when to use raw or inpainted image)
        list_inpainted_files = create_file_names_list(self.inpainted_path)

        for _, row in self.metadata.iterrows():
            # get paths
            raw_img_path = os.path.join(self.raw_path, row["filename"])
            if row["filename"] not in list_inpainted_files:
                img_path = os.path.join(self.inpainted_path, row["filename"])
            else:
                img_path = raw_img_path
            gt_wr_mask_path = os.path.join(self.gt_wr_mask_path, row["filename"])
            gt_rr_mask_path = os.path.join(self.gt_rr_mask_path, row["filename"])
            mask_path = os.path.join(self.seg_path, row["filename"])

            # load images and masks
            raw_image = self.load_image(
                raw_img_path
            )  # resize in preprocess_data (if model_name)
            image = self.load_image(
                img_path
            )  # resize in preprocess_data (if model_name)
            gt_wr_mask = self.load_image(gt_wr_mask_path, mask_size=size)
            gt_rr_mask = self.load_image(gt_rr_mask_path, mask_size=size)
            mask = self.load_mask(mask_path, mask_size=size)

            if model_name:
                raw_image = self.preprocess_data(raw_image, model_name)
                image = self.preprocess_data(image, model_name)

            # append to lists
            raw_images.append(raw_image)
            images.append(image)
            gt_wr_masks.append(gt_wr_mask)
            gt_rr_masks.append(gt_rr_mask)
            masks.append(mask)
            class_label.append(row["class_label"])
            confounder_label.append(row["confounder_label"])

        return (
            np.array(raw_images),
            np.array(images),
            np.array(gt_wr_masks),
            np.array(gt_rr_masks),
            np.array(masks),
            np.array(class_label),
            np.array(confounder_label),
        )

    def create_h5_files(self, model_name=None):
        def save_to_h5(self, path, key, data):
            """Saves the dataset to an HDF5 file."""
            with h5py.File(path, "w") as h5f:
                h5f.create_dataset(key, data=data)

        # load data
        (
            raw_images,
            images,
            gt_wr_masks,
            gt_rr_masks,
            masks,
            class_label,
            confounder_label,
        ) = self.load_data(model_name=model_name)
