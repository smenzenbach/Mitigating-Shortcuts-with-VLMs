import os
from sympy import N
import toml
import yaml
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import VGG16_Weights
from collections import Counter
from tqdm import tqdm


from utils import create_file_names_list, dilate_mask


# get dir of repo (parent dir)
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Dataset:
    def __init__(self, dataset_name, model_name="vgg16"):
        # get dataset name and yaml for raw and gt paths
        self.dataset_name = dataset_name
        with open(
            f"{repo_dir}/data/datasets.yaml", "r"
        ) as file:  # adapt path later if repo is ready (without worksopace/home)
            dataset_yaml = yaml.safe_load(file)
        # set raw and gt paths
        self.raw_dir = dataset_yaml[self.dataset_name]["raw_dir"]
        self.wr_masks_dir = dataset_yaml[self.dataset_name]["wr_masks_dir"]
        self.rr_masks_dir = dataset_yaml[self.dataset_name]["rr_masks_dir"]
        # get gt metadata
        gt_label_path = dataset_yaml[self.dataset_name]["gt_label_path"]
        self.metadata = pd.read_csv(
            gt_label_path
        )  # columns: filename, class_label, confounder_label
        self.model_name = model_name

    def load_image(self, img_path, image_size=None):
        """Loads an image using PIL and converts it to a NumPy array."""
        img = Image.open(img_path).convert("RGB")
        if image_size is not None:
            img = img.resize(self.image_size, Image.LANCZOS)  # High-quality resize
        return img

    def load_mask(self, mask_path, mask_size=None, dilate=False):
        """Loads a mask using PIL and converts it to a NumPy array."""
        mask = Image.open(mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
        if dilate:
            mask = dilate_mask(mask)
        # resize the mask if specified
        if mask_size is not None:
            mask = mask.resize(mask_size, Image.LANCZOS)
        return np.array(mask, dtype=np.float32)

    def preprocess_data(self, images):
        if self.model_name == "vgg16":
            transform = VGG16_Weights.IMAGENET1K_V1.transforms()
        images = transform(images)
        return images

    def load_data(
        self,
        augmented_images_dir=None,
        augmented_gt_label_path=None,
        generated_masks_dir=None,
    ):
        """
        Loads the original images, augmented images, ground truth wrong reason and right reason masks, generated masks (e.g. through our pipeline), class labels, and confounder labels. If model name is given the data gets preprocessed accordingly.

        Parameters:
        ----------
        augmented_images_dir : str
            Path to the augmented images directory.
        augmented_gt_label_path : str
            Path to the augmented ground truth label csv file. Not needed if augmented images are exchanged with original counterpart instead of added to original dataset.
        generated_masks_dir : str
            Path to the generated masks directory.
        """

        # get desired size according to model (if model name is given)
        if self.model_name == "vgg16":
            size = (224, 224)
        else:
            size = (299, 299)

        # get list of inpainted files (to determine when to use raw or inpainted image) and list of mask files that are generated (e.g. through our pipeline)
        if augmented_images_dir is not None:
            list_augmented_files = create_file_names_list(augmented_images_dir)
        if generated_masks_dir is not None:
            list_mask_files = create_file_names_list(generated_masks_dir)

        # merge csv files self.metadata and aug_metadata
        if augmented_gt_label_path is not None:
            aug_metadata = pd.read_csv(augmented_gt_label_path)
            metadata = pd.concat([self.metadata, aug_metadata], ignore_index=True)
        else:
            metadata = self.metadata

        # for testing:
        reduced_set = False
        if reduced_set:
            metadata = metadata.sample(n=600, random_state=1)
        # for testing end

        # get number of data entries
        data_count_train = len(metadata[metadata["split"] == 0])
        data_count_val = len(metadata[metadata["split"] == 1])
        data_count_test = len(metadata[metadata["split"] == 2])
        data_count = data_count_train + data_count_val + data_count_test
        print(
            f"Number of total data entries: {data_count}; train: {data_count_train}, val: {data_count_val}, test: {data_count_test}"
        )

        # initialize dict for train, val, test split with arrays for data
        data_dict = {
            "train": {
                "images": np.empty((data_count_train, 3, size[0], size[1])),
                "wr_masks": np.zeros((data_count_train, 1, size[0], size[1])),
                "rr_masks": np.zeros((data_count_train, 1, size[0], size[1])),
                "gen_masks": np.zeros((data_count_train, 1, size[0], size[1])),
                "class_label": np.zeros(data_count_train).astype(np.int64),
                "confounder_label": np.zeros(data_count_train).astype(np.int64),
            },
            "val": {
                "images": np.empty((data_count_val, 3, size[0], size[1])),
                "wr_masks": np.zeros((data_count_val, 1, size[0], size[1])),
                "rr_masks": np.zeros((data_count_val, 1, size[0], size[1])),
                "gen_masks": np.zeros((data_count_val, 1, size[0], size[1])),
                "class_label": np.zeros(data_count_val).astype(np.int64),
                "confounder_label": np.zeros(data_count_val).astype(np.int64),
            },
            "test": {
                "images": np.empty((data_count_test, 3, size[0], size[1])),
                "wr_masks": np.zeros((data_count_test, 1, size[0], size[1])),
                "rr_masks": np.zeros((data_count_test, 1, size[0], size[1])),
                "gen_masks": np.zeros((data_count_test, 1, size[0], size[1])),
                "class_label": np.zeros(data_count_test).astype(np.int64),
                "confounder_label": np.zeros(data_count_test).astype(np.int64),
            },
        }

        # initialize indices (incremented before adding data)
        train_ind = -1
        val_ind = -1
        test_ind = -1
        for _, row in tqdm(metadata.iterrows()):
            # get paths
            if (
                augmented_images_dir is not None  # if augmented images are given
                and row["filename"]
                in list_augmented_files  # and file is in augmented images dir
                and row["split"] == 0  # and file is in train split
            ):  # then use augmented image
                img_path = os.path.join(augmented_images_dir, row["filename"])
            else:  # else use raw image
                img_path = os.path.join(self.raw_dir, row["filename"])
            wr_mask_path = os.path.join(self.wr_masks_dir, row["filename"])
            rr_mask_path = os.path.join(self.rr_masks_dir, row["filename"])
            if generated_masks_dir is not None and row["filename"] in list_mask_files:
                # if generated mask exists, use it
                mask_path = os.path.join(generated_masks_dir, row["filename"])
            else:
                mask_path = None

            # load images and masks
            image = self.preprocess_data(self.load_image(img_path))
            if os.path.exists(wr_mask_path):
                wr_mask = self.load_mask(wr_mask_path, mask_size=size)
            else:
                wr_mask = None
            if os.path.exists(rr_mask_path):
                rr_mask = self.load_mask(rr_mask_path, mask_size=size)
            else:
                rr_mask = None
            if mask_path is not None:
                mask = self.load_mask(mask_path, mask_size=size, dilate=True)

            # get key for split
            if row["split"] == 0:
                key = "train"
                train_ind += 1
                i = train_ind
            elif row["split"] == 1:
                key = "val"
                val_ind += 1
                i = val_ind
            else:
                key = "test"
                test_ind += 1
                i = test_ind

            # add data to arrays
            data_dict[key]["images"][i] = image
            if wr_mask is not None:  # if mask exists, set wr_mask
                data_dict[key]["wr_masks"][i] = wr_mask
            if rr_mask is not None:  # if mask exists, set rr_mask
                data_dict[key]["rr_masks"][i] = rr_mask
            if mask_path is not None:  # if None, gen_mask stays 0
                data_dict[key]["gen_masks"][i] = mask
            data_dict[key]["class_label"][i] = int(row["class_label"])
            data_dict[key]["confounder_label"][i] = int(row["confounder_label"])

        return data_dict

    def load_dataloaders_and_loss_weights(
        self,
        augmented_images_dir=None,
        augmented_gt_label_path=None,
        generated_masks_dir=None,
        batch_size=16,
    ):
        """Loads the data, builds the datasets, and returns the dataloaders and loss weights for the training and test sets.

        The structure of the data in the dataloader is as follows:
        data[0]: images
        data[1]: class label
        data[2]: confounder label
        data[3]: wrong reason mask
        data[4]: right reason mask
        data[5]: generated mask (augmentation loss)

        Parameters:
        ----------
        augmented_images_dir : str
            Path to the augmented images directory.
        augmented_gt_label_path : str
            Path to the augmented ground truth label csv file. Not needed if augmented images are exchanged with original counterpart instead of added to original dataset.
        generated_masks_dir : str
            Path to the generated masks directory.
        """

        def calculate_loss_weights(counter):
            class0 = counter.get(0, 0)  # Get count of class 0, default to 0 if missing
            # class1 = counter.get(1, 0)  # Get count of class 1, default to 0 if missing
            class0_ratio = class0 / len(dataset)
            class1_ratio = 1 - class0_ratio
            if class0_ratio == 0:
                return torch.tensor([1.0, 0.0]).float()
            if class1_ratio == 0:
                return torch.tensor([0.0, 1.0]).float()
            class0_weight = 1 / class0_ratio
            class1_weight = 1 / class1_ratio
            weights = np.asarray([class0_weight, class1_weight])
            weights /= weights.sum()
            weights = torch.tensor(weights).float()
            return weights

        # load data
        print("Loading data...")
        data_dict = self.load_data(
            augmented_images_dir, augmented_gt_label_path, generated_masks_dir
        )

        # get dataloaders and loss weights
        dataloaders = {}
        loss_weights = {}
        for split_key in ["train", "val", "test"]:
            print(f"\nBuilding dataset for {split_key} set...")
            data = data_dict[split_key]
            dataset = TensorDataset(
                torch.from_numpy(data["images"]).float(),
                torch.from_numpy(data["class_label"]),
                torch.from_numpy(data["confounder_label"]),
                torch.from_numpy(data["wr_masks"]).float(),
                torch.from_numpy(data["rr_masks"]).float(),
                torch.from_numpy(data["gen_masks"]).float(),
            )

            classes = [x[1].item() for x in dataset]
            confounder_dist = [x[2].item() for x in dataset]

            dataloaders[split_key] = DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            loss_weights[split_key] = calculate_loss_weights(Counter(classes))

            print(f"Number of samples in {split_key} set: {len(dataset)}")
            print(f"{split_key.upper()} class dist: {Counter(classes)}")
            print(f"{split_key.upper()} confounder dist: {Counter(confounder_dist)}")
            print(f"{split_key.upper()} weights: {str(loss_weights[split_key])}\n")

        return dataloaders, loss_weights
