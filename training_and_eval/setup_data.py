import os
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

    def load_data(self, augmented_images_dir=None, generated_masks_dir=None):
        """
        Loads the original images, augmented images, ground truth wrong reason and right reason masks, generated masks (e.g. through our pipeline), class labels, and confounder labels. If model name is given the data gets preprocessed accordingly.

        Parameters:
        ----------
        augmented_images_dir : str
            Path to the augmented images directory.
        generated_masks_dir : str
            Path to the generated masks directory.
        """

        # get desired size according to model (if model name is given)
        if self.model_name == "vgg16":
            size = (224, 224)
        else:
            size = (299, 299)

        # get number of data entries
        data_count = len(self.metadata)
        print(f"Number of data entries: {data_count}")

        # initialize arrays
        original_images = np.empty((data_count, 3, size[0], size[1]))
        augmented_images = np.empty((data_count, 3, size[0], size[1]))
        wr_masks = np.zeros((data_count, 1, size[0], size[1]))
        rr_masks = np.zeros((data_count, 1, size[0], size[1]))
        gen_masks = np.zeros((data_count, 1, size[0], size[1]))
        class_label = np.zeros(data_count).astype(np.int64)
        confounder_label = np.zeros(data_count).astype(np.int64)

        # get list of inpainted files (to determine when to use raw or inpainted image) and list of mask files that are generated (e.g. through our pipeline)
        if augmented_images_dir is not None:
            list_augmented_files = create_file_names_list(augmented_images_dir)
        if generated_masks_dir is not None:
            list_mask_files = create_file_names_list(generated_masks_dir)

        for i, row in tqdm(self.metadata.iterrows()):
            # get paths
            raw_img_path = os.path.join(self.raw_dir, row["filename"])
            if (
                augmented_images_dir is not None
                and row["filename"] in list_augmented_files
            ):  # if augmented image exists, use it
                augmented_img_path = os.path.join(augmented_images_dir, row["filename"])
            else:
                augmented_img_path = raw_img_path
            wr_mask_path = os.path.join(self.wr_masks_dir, row["filename"])
            rr_mask_path = os.path.join(self.rr_masks_dir, row["filename"])
            if generated_masks_dir is not None and row["filename"] in list_mask_files:
                # if generated mask exists, use it
                mask_path = os.path.join(generated_masks_dir, row["filename"])
            else:
                mask_path = None

            # load images and masks
            original_image = self.preprocess_data(self.load_image(raw_img_path))
            augmented_image = self.preprocess_data(self.load_image(augmented_img_path))
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

            # add data to arrays
            original_images[i] = original_image
            augmented_images[i] = augmented_image
            if wr_mask is not None:  # if mask exists, set wr_mask
                wr_masks[i] = wr_mask
            if rr_mask is not None:  # if mask exists, set rr_mask
                rr_masks[i] = rr_mask
            if mask_path is not None:  # if None, gen_mask stays 0
                gen_masks[i] = mask
            class_label[i] = int(row["class_label"])
            confounder_label[i] = int(row["confounder_label"])

        return (
            original_images,
            augmented_images,
            wr_masks,
            rr_masks,
            gen_masks,
            class_label,
            confounder_label,
        )

    def load_dataloaders_and_loss_weights(
        self,
        augmented_images_dir,
        generated_masks_dir,
        training_mode,
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
        generated_masks_dir : str
            Path to the generated masks directory.
        training_mode : str
            Training mode for the model. Options: Vanilla, Inpainting.
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
        (
            original_images,
            augmented_images,
            wr_masks,
            rr_masks,
            gen_masks,
            class_label,
            confounder_label,
        ) = self.load_data(augmented_images_dir, generated_masks_dir)

        # Build datasets
        print("Building datasets...")
        if training_mode == "Vanilla":
            dataset = TensorDataset(
                torch.from_numpy(original_images).float(),
                torch.from_numpy(class_label),
                torch.from_numpy(confounder_label),
                torch.from_numpy(wr_masks).float(),
                torch.from_numpy(rr_masks).float(),
                torch.from_numpy(gen_masks).float(),
            )

        elif training_mode == "Inpainting":
            dataset = TensorDataset(
                torch.from_numpy(augmented_images).float(),
                torch.from_numpy(class_label),
                torch.from_numpy(confounder_label),
                torch.from_numpy(wr_masks).float(),
                torch.from_numpy(rr_masks).float(),
                torch.from_numpy(gen_masks).float(),
            )

        # Get split for train and test set
        num_total = len(dataset)
        num_train = int(0.8 * num_total)
        num_test = int(0.1 * num_total)
        num_val = num_total - num_train - num_test

        # Split data and crate dataloaders
        print("Splitting data and creating dataloaders...")
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [num_train, num_val, num_test],
            generator=torch.Generator().manual_seed(0),
        )

        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        test_indices = test_dataset.indices

        print("Sizes of datasets:")
        print(
            f"TRAIN: {len(train_indices)}, VAL: {len(val_indices)}, TEST: {len(test_indices)}"
        )

        dataloaders = {}
        loss_weights = {}

        # train

        train_classes = [x[1].item() for x in train_dataset]
        train_confounder_dis = [x[2].item() for x in train_dataset]

        dataloaders["train"] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        loss_weights["train"] = calculate_loss_weights(Counter(train_classes))

        print(f"\nTRAIN class dist: {Counter(train_classes)}")
        print(f"TRAIN confounder dist: {Counter(train_confounder_dis)}")
        print(f"TRAIN loss weights: {str(loss_weights['train'])}")

        # val
        val_classes = [x[1].item() for x in val_dataset]
        val_confounder_dis = [x[2].item() for x in val_dataset]

        dataloaders["val"] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )
        loss_weights["val"] = calculate_loss_weights(Counter(train_classes))

        print(f"\nVAL class dist: {Counter(val_classes)}")
        print(f"VAL confounder dist: {Counter(val_confounder_dis)}")
        print(f"VAL loss weights: {str(loss_weights['val'])}")

        # test
        test_classes = [x[1].item() for x in test_dataset]
        test_confounder_dis = [x[2].item() for x in test_dataset]

        dataloaders["test"] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )
        loss_weights["test"] = calculate_loss_weights(Counter(train_classes))

        print(f"\nTEST class dist: {Counter(test_classes)}")
        print(f"TEST confounder dist: {Counter(test_confounder_dis)}")
        print(f"TEST loss weights: {str(loss_weights['test'])}")

        return dataloaders, loss_weights
