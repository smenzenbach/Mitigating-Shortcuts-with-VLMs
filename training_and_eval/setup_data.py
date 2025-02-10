import os
import toml
import yaml
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import ConcatDataset
from torchvision.models import VGG16_Weights
from collections import Counter


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
        # get confounded_class_label
        self.confounded_class_label = dataset_yaml[self.dataset_name]["confounded_class_label"]
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
        self.h5_files_path = f"{repo_dir}/pipeline_output/{self.dataset_name}/{self.config_id}/pipeline_h5_files"
        self.h5_raw_files = f"{repo_dir}/pipeline_output/{self.dataset_name}/h5_files"

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
            size = (299, 299)

        # get list of inpainted files (to determine when to use raw or inpainted image)
        list_inpainted_files = create_file_names_list(self.inpainted_path)
        list_wr_mask_files = create_file_names_list(self.gt_wr_seg_path)
        list_rr_mask_files = create_file_names_list(self.gt_rr_seg_path)
        list_mask_files = create_file_names_list(self.seg_path)

        data_count = len(self.metadata.size)# todo: entries of pd data structure

        raw_images = np.empty((data_count, 3, size[0], size[1]))
        images = np.empty((data_count, 3, size[0], size[1])) 
        gt_wr_mask = np.zeros((data_count, 1, 299, 299))
        gt_rr_mask = np.zeros((data_count, 1, 299, 299))
        masks = np.zeros((data_count, 1, 299, 299))
        class_label = np.zeros(data_count).astype(np.int64)
        confounder_label = np.zeros(data_count).astype(np.int64)



        for i, row in self.metadata.iterrows():
            # get paths
            raw_img_path = os.path.join(self.raw_path, row["filename"])
            if row["filename"] not in list_inpainted_files:
                img_path = os.path.join(self.inpainted_path, row["filename"])
            else:
                img_path = raw_img_path
            gt_wr_mask_path = os.path.join(self.gt_wr_seg_path, row["filename"])
            gt_rr_mask_path = os.path.join(self.gt_rr_seg_path, row["filename"])
            mask_path = os.path.join(self.seg_path, row["filename"])

            # load images and masks
            raw_image = self.load_image(
                raw_img_path
            )  # resize in preprocess_data (if model_name)
            image = self.load_image(
                img_path
            )  # resize in preprocess_data (if model_name)
            gt_wr_mask = self.load_mask(gt_wr_mask_path, mask_size=size)
            gt_rr_mask = self.load_mask(gt_rr_mask_path, mask_size=size)
            mask = self.load_mask(mask_path, mask_size=size)

            if model_name:
                raw_image = self.preprocess_data(raw_image, model_name)
                image = self.preprocess_data(image, model_name)

            # append to lists
            raw_images[i] = raw_image
            images[i] = image
            gt_wr_masks[i] = gt_wr_mask
            gt_rr_masks[i] = gt_rr_mask
            masks[i] = mask
            class_label[i] = row["class_label"]
            confounder_label[i] = row["confounder_label"]

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
        def save_to_h5(path, key, data):
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

        # save files
        raw_images_path = os.path.join(self.h5_raw_files, "raw_images.h5")
        if not os.path.exists(raw_images_path):
            save_to_h5(raw_images_path, "raw_images", raw_images)
        
        images_path = os.path.join(self.h5_files_path, "images.h5")
        save_to_h5(images_path, "images", images)

        wr_masks_path = os.path.join(self.h5_raw_files, "wr_masks.h5")
        if not os.path.exists(wr_masks_path):
            save_to_h5(wr_masks_path, "wr_masks",gt_wr_masks)

        rr_masks_path = os.path.join(self.h5_raw_files, "rr_masks.h5")
        if not os.path.exists(rr_masks_path):
            save_to_h5(rr_masks_path, "rr_masks",gt_rr_masks)

        masks_path = os.path.join(self.h5_files_path, "masks.h5")
        save_to_h5(masks_path, "masks", masks)

        class_label_path = os.path.join(self.h5_raw_files, "class_label.h5")
        if not os.path.exists(class_label_path):
            save_to_h5(class_label_path, "class_label", class_label)

        confounder_label_path = os.path.join(self.h5_raw_files, "confounder_label.h5")
        if not os.path.exists(confounder_label_path):
            save_to_h5(confounder_label_path, "confounder_label", confounder_label)


    def load_dataloaders_and_loss_weights(self, use_pipeline_data, batch_size=16):
        def load_h5_files(path, key, number_samples=None):
            with h5py.File(path, "r") as h5f:
                if number_samples is None:
                    return h5f[key][:]
                else:
                    return h5f[key][:number_samples]
                
        def calculate_loss_weights(dataset):
            confounded_class = 0
            non_confounded_class = 0
            for i in dataset.indices:
                if dataset[i][1] == self.confounded_class_label:
                    confounded_class += 1
                else:
                    non_confounded_class += 1
            confounded_class_ratio = confounded_class / len(dataset)
            non_confounded_class_ratio = 1 - confounded_class_ratio
            if confounded_class_ratio == 0:
                return torch.tensor([1.0, 0.0]).float()
            if non_confounded_class_ratio == 0:
                return torch.tensor([0.0, 1.0]).float()
            confounded_class_weight = 1 / confounded_class_ratio
            non_confounded_class_weight = 1 / non_confounded_class_ratio
            if self.confounded_class_label == 1:
                weights = np.asarray([non_confounded_class_weight, confounded_class_weight])
            else:
                weights = np.asarray([confounded_class_weight, non_confounded_class_weight])
            weights /= weights.sum()
            weights = torch.tensor(weights).float()
            print("For train dataset:")
            print(f"  TRAIN class dist: confounded class: {confounded_class}, non-confounded class: {non_confounded_class}")
            print(f"  Loss weights: {str(weights)}")

            return weights

                
        # Load data from h5 files
        raw_images_path = os.path.join(self.h5_raw_files, "raw_images.h5")
        raw_images = load_h5_files(raw_images_path, "raw_images")

        if use_pipeline_data:
            images_path = os.path.join(self.h5_files_path, "images.h5")
            images = load_h5_files(images_path, "images")

        wr_masks_path = os.path.join(self.h5_raw_files, "wr_masks.h5")
        gt_wr_masks = load_h5_files(wr_masks_path, "wr_masks",gt_wr_masks)

        rr_masks_path = os.path.join(self.h5_raw_files, "rr_masks.h5")
        gt_rr_masks = load_h5_files(rr_masks_path, "rr_masks")

        if use_pipeline_data:
            masks_path = os.path.join(self.h5_files_path, "masks.h5")
            masks = load_h5_files(masks_path, "masks")

        class_label_path = os.path.join(self.h5_raw_files, "class_label.h5")
        class_label = load_h5_files(class_label_path, "class_label")

        confounder_label_path = os.path.join(self.h5_raw_files, "confounder_label.h5")
        confounder_label = load_h5_files(confounder_label_path, "confounder_label")

        # Build datasets
        raw_set = TensorDataset(
            torch.from_numpy(raw_images).float(),
            torch.from_numpy(class_label),
            torch.from_numpy(gt_wr_masks).float(),
            torch.from_numpy(confounder_label),
        )

        if use_pipeline_data:
            pipeline_set = TensorDataset(
                torch.from_numpy(images).float(),
                torch.from_numpy(class_label),
                torch.from_numpy(masks).float(),
                torch.from_numpy(confounder_label),
            )

        rr_set = TensorDataset(
            torch.from_numpy(images).float(),
            torch.from_numpy(class_label),
            torch.from_numpy(gt_rr_masks).float(),
            torch.from_numpy(confounder_label),
        )


        # Get split for train and test set
        num_total = len(raw_set)
        num_train = int(0.8 * num_total)
        num_test = num_total - num_train

        # Build datasets
        raw_train_dataset, test_dataset_ = torch.utils.data.random_split(
            raw_set,
            [num_train, num_test],
            generator=torch.Generator().manual_seed(0),
        )

        train_indices = raw_train_dataset.indices
        test_C_indices = [
            idx
            for idx in test_dataset_.indices
            if raw_set[idx][3] == 1 or raw_set[idx][1] != self.confounded_class_label
        ] # confouned images of confounded class and all non-confounded class images
        test_NC_indices = [
            idx
            for idx in test_dataset_.indices
            if raw_set[idx][3] == 0 or raw_set[idx][1] != self.confounded_class_label
        ] # non-confouned images of confounded class and all non-confounded class images

        if use_pipeline_data:
            pipeline_train_dataset = torch.utils.data.Subset(pipeline_set, train_indices)
        test_C_dataset = torch.utils.data.Subset(raw_set, test_C_indices)
        test_NC_dataset = torch.utils.data.Subset(raw_set, test_NC_indices)

        rr_dataset =  torch.utils.data.Subset(rr_set, test_C_indices)

        print("Sizes of datasets:")
        print(
            f"TRAIN: {len(train_indices)}, TEST: {len(test_C_indices)}, TEST_NO_CONFOUNDER: {len(test_NC_indices)}"
        )

        train_classes = [x[1].item() for x in raw_train_dataset]
        train_confounder_dis = [x[3].item() for x in raw_train_dataset]

        test_C_classes = [x[1].item() for x in test_C_dataset]
        test_NC_classes = [x[1].item() for x in test_NC_dataset]


        print(f"TRAIN class dist: {Counter(train_classes)}")
        print(f"TRAIN mask dist: {Counter(train_confounder_dis)}") 
        print(f"TEST class dist: {Counter(test_C_classes)}")
        print(f"TEST_NO_CONFOUNDER class dist: {Counter(test_NC_classes)}")

        dataloaders = {}
        loss_weights = {}
        
        # train
        if use_pipeline_data:
            dataloaders["train"] = DataLoader(
                pipeline_train_dataset, batch_size=batch_size, shuffle=True
                )
        else:
            dataloaders["train"] = DataLoader(
                raw_train_dataset, batch_size=batch_size, shuffle=True
                )
        
        loss_weights["train"] = calculate_loss_weights(raw_train_dataset)

        # test confounded
        dataloaders["test_C"] = DataLoader(
                test_C_dataset, batch_size=batch_size, shuffle=True
                )
        
        loss_weights["test_C"] = calculate_loss_weights(raw_train_dataset)

        # test non-confounded
        dataloaders["test_NC"] = DataLoader(
                test_NC_dataset, batch_size=batch_size, shuffle=True
                )
        
        loss_weights["test_NC"] = calculate_loss_weights(raw_train_dataset)
        
        # right-reason 
        dataloaders["rr"] = DataLoader(
                rr_dataset, batch_size=batch_size, shuffle=True
                )
        
        loss_weights["rr"] = calculate_loss_weights(raw_train_dataset)

        return dataloaders, loss_weights