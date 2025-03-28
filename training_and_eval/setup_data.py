import os
import yaml
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, ResNet50_Weights
from collections import Counter
from tqdm import tqdm


from utils import create_file_names_list, dilate_mask, MyLogger

# get dir of repo (parent dir)
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

coco_file_exceptions = [
    '000000301102.jpg', '000000389986.jpg', '000000279331.jpg', '000000129186.jpg', '000000111062.jpg', 
    '000000552656.jpg', '000000018319.jpg', '000000141946.jpg', '000000080194.jpg', '000000525815.jpg', 
    '000000063270.jpg', '000000383220.jpg', '000000157107.jpg', '000000029875.jpg', '000000243783.jpg', 
    '000000528550.jpg', '000000258304.jpg', '000000460279.jpg', '000000411857.jpg', '000000434981.jpg', 
    '000000385114.jpg', '000000217289.jpg', '000000506209.jpg', '000000348516.jpg', '000000103966.jpg', 
    '000000491430.jpg', '000000424105.jpg', '000000560180.jpg', '000000138299.jpg', '000000365999.jpg', 
    '000000265748.jpg', '000000056896.jpg', '000000214856.jpg'
]
# for coco_ds
coco_file_expceptions = []


class Dataset:
    def __init__(self, dataset_name, model_name="vgg16", logger=None):
        # get dataset name and yaml for raw and gt paths
        self.dataset_name = dataset_name
        with open(
            f"{repo_dir}/data/datasets_info.yaml", "r"
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
        # set worst-group label
        self.worst_group_class_label = dataset_yaml[self.dataset_name][
            "worst_group_class_label"
        ]
        self.h5_dir = f"{repo_dir}/training_and_eval/h5_files/{self.dataset_name}"
        os.makedirs(self.h5_dir, exist_ok=True)
        # set logger
        if logger is not None:
            self.logger = logger
        else:
            logger_dir = f"{repo_dir}/training_and_eval/logs"
            os.makedirs(logger_dir, exist_ok=True)
            logger = MyLogger(f"{logger_dir}/setup_data_{self.dataset_name}.log").logger
        self.logger.info(
            f"Dataset {self.dataset_name} initialized for model {self.model_name}.\n"
        )

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
        elif self.model_name == "resnet50":
            transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
        images = transform(images)
        return images

    def load_data_generator(
        self,
        split_key,
        augmented_images_dir=None,
        augmented_gt_label_path=None,
        generated_masks_dir=None,
        return_length=False,
    ):
        """
        Loads the original images, augmented images, ground truth wrong reason and right reason masks, generated masks (e.g. through our pipeline), class labels, and confounder labels. If model name is given the data gets preprocessed accordingly.

        Parameters:
        ----------
        split_key : str
            Key for the split (train, val, test).
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
        elif self.model_name == "resnet50":
            size = (224, 224)
        else:
            size = (299, 299)

        # get list of inpainted files (to determine when to use raw or inpainted image) and list of mask files that are generated (e.g. through our pipeline)
        if augmented_images_dir is not None:
            list_augmented_files = create_file_names_list(augmented_images_dir)
            self.logger.info(f"Found {len(list_augmented_files)} augmented images.")
        if generated_masks_dir is not None:
            list_mask_files = create_file_names_list(generated_masks_dir)
            self.logger.info(f"Found {len(list_mask_files)} generated masks.")

        # merge csv files self.metadata and aug_metadata
        if augmented_gt_label_path is not None:
            aug_metadata = pd.read_csv(augmented_gt_label_path)
            metadata = pd.concat([self.metadata, aug_metadata], ignore_index=True)
        else:
            metadata = self.metadata

        # for testing:
        reduced_set = False
        if reduced_set:
            metadata = metadata.sample(n=10000, random_state=1)
        # for testing end

        # get number of data entries
        data_count_train = len(metadata[metadata["split"] == 0])
        data_count_val = len(metadata[metadata["split"] == 1])
        data_count_test = len(metadata[metadata["split"] == 2])
        data_count = data_count_train + data_count_val + data_count_test
        print(
            f"Number of total data entries: {data_count}; train: {data_count_train}, val: {data_count_val}, test: {data_count_test}"
        )

        split_dict = {"train": 0, "val": 1, "test": 2}

        # get number of data entries for split
        split_count = len(
            metadata[metadata["split"] == split_dict.get(split_key, split_key)]
        )

        # create a generator function for data
        def data_generator():
            split = split_dict.get(
                split_key, split_key
            )  # if split is string, get the corresponding int, else keep int
            for _, row in metadata[metadata["split"] == split].iterrows():
                if (
                    augmented_images_dir is not None
                    and row["filename"] in list_augmented_files
                    and row["split"] == 0
                ):
                    img_path = os.path.join(augmented_images_dir, row["filename"])
                else:
                    if row["filename"] in coco_file_exceptions:
                        # replace val with train in self.raw_dir
                        train_dir = self.raw_dir.replace("val", "train")
                        img_path = os.path.join(train_dir, row["filename"])
                    else:
                        img_path = os.path.join(self.raw_dir, row["filename"])
                    if not os.path.exists(img_path):
                        # Check if the file is in a subdirectory of raw_dir
                        # TODO: this is too slow for large datasets how can I handle sub_dirs?
                        found = False
                        for root, _, files in os.walk(self.raw_dir):
                            if row["filename"] in files:
                                img_path = os.path.join(root, row["filename"])
                                found = True
                                break
                        if not found:
                            raise FileNotFoundError(
                                f"File {row['filename']} not found in {self.raw_dir} or its subdirectories."
                            )
                wr_mask_path = os.path.join(self.wr_masks_dir, row["filename"])
                rr_mask_path = os.path.join(self.rr_masks_dir, row["filename"])
                if (
                    generated_masks_dir is not None
                    and row["filename"] in list_mask_files
                ):
                    mask_path = os.path.join(generated_masks_dir, row["filename"])
                else:
                    mask_path = None

                # load images and masks
                image = self.preprocess_data(self.load_image(img_path))
                wr_mask = (
                    self.load_mask(wr_mask_path, mask_size=size)
                    if os.path.exists(wr_mask_path)
                    else np.zeros(size)
                )
                rr_mask = (
                    self.load_mask(rr_mask_path, mask_size=size)
                    if os.path.exists(rr_mask_path)
                    else np.zeros(size)
                )
                mask = (
                    self.load_mask(mask_path, mask_size=size, dilate=True)
                    if mask_path is not None
                    else np.zeros(size)
                )

                yield {
                    "image": image,
                    "wr_mask": wr_mask,
                    "rr_mask": rr_mask,
                    "gen_mask": mask,
                    "class_label": int(row["class_label"]),
                    "confounder_label": int(row["confounder_label"]),
                }

        if return_length:
            return data_generator(), split_count
        return data_generator()

    def create_h5_file(
        self,
        identifier,
        split_key,
        augmented_images_dir=None,
        augmented_gt_label_path=None,
        generated_masks_dir=None,
        batch_size=200,
    ):
        """
        Loads the data and creates h5_files for train, val, and test splits.
        Ensures batches are appended to existing datasets without overwriting.
        """

        self.logger.info(
            f"Creating h5 file for '{split_key}' split for identifier '{identifier}'."
        )

        # Ensure output directory exists
        os.makedirs(os.path.join(self.h5_dir, identifier), exist_ok=True)

        h5_file_path = os.path.join(self.h5_dir, identifier, f"{split_key}_data.h5")
        with h5py.File(h5_file_path, "a") as f:
            # We will lazily create the datasets once we have seen the first sample.
            datasets_initialized = False

            # Temporary buffers for accumulating one batch worth of data
            images_batch = []
            class_batch = []
            confounder_batch = []
            wr_masks_batch = []
            rr_masks_batch = []
            gen_masks_batch = []

            # Get a generator for the current split
            gen, split_count = self.load_data_generator(
                split_key,
                augmented_images_dir,
                augmented_gt_label_path,
                generated_masks_dir,
                return_length=True,
            )

            if gen is None:
                self.logger.warning(f"No data found for split: {split_key}")

            self.logger.info(f"Processing {split_count} samples for split: {split_key}")

            for sample in tqdm(
                gen, desc=f"Processing {split_key} data", total=split_count
            ):
                # Append sample data to buffers
                images_batch.append(sample["image"])
                class_batch.append(sample["class_label"])
                confounder_batch.append(sample["confounder_label"])
                wr_masks_batch.append(sample["wr_mask"])
                rr_masks_batch.append(sample["rr_mask"])
                gen_masks_batch.append(sample["gen_mask"])

                # If we've collected a full batch, write to HDF5
                if len(images_batch) == batch_size:
                    # Initialize datasets on the first batch
                    if not datasets_initialized:
                        # Determine shapes from the first sample in the batch
                        image_shape = np.array(sample["image"]).shape  # e.g., (3, H, W)
                        # Assuming masks are 2D; we add a channel dimension to get (1, H, W)
                        mask_shape = np.array(sample["wr_mask"]).shape
                        # Create datasets with initial shape (0, ...)
                        f.create_dataset(
                            "images",
                            shape=(0, *image_shape),
                            maxshape=(None, *image_shape),
                            chunks=True,
                            dtype="float32",
                        )
                        f.create_dataset(
                            "class_label",
                            shape=(0,),
                            maxshape=(None,),
                            chunks=True,
                            dtype="int64",
                        )
                        f.create_dataset(
                            "confounder_label",
                            shape=(0,),
                            maxshape=(None,),
                            chunks=True,
                            dtype="int64",
                        )
                        f.create_dataset(
                            "wr_masks",
                            shape=(0, 1, *mask_shape),
                            maxshape=(None, 1, *mask_shape),
                            chunks=True,
                            dtype="float32",
                        )
                        f.create_dataset(
                            "rr_masks",
                            shape=(0, 1, *mask_shape),
                            maxshape=(None, 1, *mask_shape),
                            chunks=True,
                            dtype="float32",
                        )
                        f.create_dataset(
                            "gen_masks",
                            shape=(0, 1, *mask_shape),
                            maxshape=(None, 1, *mask_shape),
                            chunks=True,
                            dtype="float32",
                        )
                        datasets_initialized = True

                    # Convert buffers to numpy arrays
                    images_batch_np = np.array(images_batch, dtype="float32")
                    class_batch_np = np.array(class_batch, dtype="int64")
                    confounder_batch_np = np.array(confounder_batch, dtype="int64")
                    wr_masks_batch_np = np.array(wr_masks_batch, dtype="float32")
                    rr_masks_batch_np = np.array(rr_masks_batch, dtype="float32")
                    gen_masks_batch_np = np.array(gen_masks_batch, dtype="float32")

                    # Get current length of datasets to know where to append new data
                    cur_len = f["images"].shape[0]
                    new_len = cur_len + images_batch_np.shape[0]

                    # Resize and append data for each dataset
                    f["images"].resize((new_len, *f["images"].shape[1:]))
                    f["images"][cur_len:new_len] = images_batch_np

                    f["class_label"].resize((new_len,))
                    f["class_label"][cur_len:new_len] = class_batch_np

                    f["confounder_label"].resize((new_len,))
                    f["confounder_label"][cur_len:new_len] = confounder_batch_np

                    f["wr_masks"].resize((new_len, *f["wr_masks"].shape[1:]))
                    # Ensure masks have the shape (batch, 1, H, W)
                    f["wr_masks"][cur_len:new_len] = wr_masks_batch_np.reshape(
                        -1, 1, *wr_masks_batch_np.shape[1:]
                    )

                    f["rr_masks"].resize((new_len, *f["rr_masks"].shape[1:]))
                    f["rr_masks"][cur_len:new_len] = rr_masks_batch_np.reshape(
                        -1, 1, *rr_masks_batch_np.shape[1:]
                    )

                    f["gen_masks"].resize((new_len, *f["gen_masks"].shape[1:]))
                    f["gen_masks"][cur_len:new_len] = gen_masks_batch_np.reshape(
                        -1, 1, *gen_masks_batch_np.shape[1:]
                    )

                    # Clear buffers for the next batch
                    images_batch, class_batch = [], []
                    confounder_batch, wr_masks_batch = [], []
                    rr_masks_batch, gen_masks_batch = [], []

            # After finishing the loop, check if there is any remaining data to write
            if images_batch:
                if not datasets_initialized:
                    raise ValueError(
                        "Dataset was not initialized, but data was yielded from the generator."
                    )

                cur_len = f["images"].shape[0]
                new_len = cur_len + len(images_batch)
                f["images"].resize((new_len, *f["images"].shape[1:]))
                f["images"][cur_len:new_len] = np.array(images_batch, dtype="float32")

                f["class_label"].resize((new_len,))
                f["class_label"][cur_len:new_len] = np.array(class_batch, dtype="int64")

                f["confounder_label"].resize((new_len,))
                f["confounder_label"][cur_len:new_len] = np.array(
                    confounder_batch, dtype="int64"
                )

                f["wr_masks"].resize((new_len, *f["wr_masks"].shape[1:]))
                f["wr_masks"][cur_len:new_len] = np.array(
                    wr_masks_batch, dtype="float32"
                ).reshape(-1, 1, *np.array(wr_masks_batch[0]).shape)

                f["rr_masks"].resize((new_len, *f["rr_masks"].shape[1:]))
                f["rr_masks"][cur_len:new_len] = np.array(
                    rr_masks_batch, dtype="float32"
                ).reshape(-1, 1, *np.array(rr_masks_batch[0]).shape)

                f["gen_masks"].resize((new_len, *f["gen_masks"].shape[1:]))
                f["gen_masks"][cur_len:new_len] = np.array(
                    gen_masks_batch, dtype="float32"
                ).reshape(-1, 1, *np.array(gen_masks_batch[0]).shape)
            else:
                if not datasets_initialized:
                    self.logger.warning(
                        f"No data was yielded from the generator for split: {split_key}. Create an empty dataset."
                    )
                    # Create empty datasets if no data was found
                    f.create_dataset(
                        "images",
                        shape=(0, 3, 224, 224),
                        dtype="float32",
                    )
                    f.create_dataset(
                        "class_label",
                        shape=(0,),
                        dtype="int64",
                    )
                    f.create_dataset(
                        "confounder_label",
                        shape=(0,),
                        dtype="int64",
                    )
                    f.create_dataset(
                        "wr_masks",
                        shape=(0, 1, 224, 224),
                        dtype="float32",
                    )
                    f.create_dataset(
                        "rr_masks",
                        shape=(0, 1, 224, 224),
                        dtype="float32",
                    )
                    f.create_dataset(
                        "gen_masks",
                        shape=(0, 1, 224, 224),
                        dtype="float32",
                    )

        self.logger.info(
            f"Created/Updated h5 file for '{split_key}' split for identifier '{identifier}'."
        )

    def load_dataloaders_and_loss_weights(
        self,
        identifier,
        augmented_images_dir=None,
        augmented_gt_label_path=None,
        generated_masks_dir=None,
        batch_size_processing_img=1000,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
    ):
        """Loads the data from h5, builds the datasets, and returns the dataloaders and loss weights for the training and test sets.

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
        batch_size : int
            Batch size for the dataloaders.
        num_workers : int
            Number of workers for the dataloaders.
        pin_memory : bool
            If to pin memory for the dataloaders.
        """

        def calculate_loss_weights(counter, total_samples):
            if total_samples == 0:
                return torch.tensor([0.0, 0.0]).float()
            class0 = counter.get(0, 0)  # Get count of class 0, default to 0 if missing
            # class1 = counter.get(1, 0)  # Get count of class 1, default to 0 if missing
            class0_ratio = class0 / total_samples
            class1_ratio = 1 - class0_ratio
            # Avoid division by zero
            if class0_ratio == 0:
                return torch.tensor([1.0, 0.0]).float()
            if class1_ratio == 0:
                return torch.tensor([0.0, 1.0]).float()
            class0_weight = 1 / class0_ratio
            class1_weight = 1 / class1_ratio
            weights = np.asarray([class0_weight, class1_weight])
            weights /= weights.sum()
            return torch.tensor(weights).float()

        dataloaders = {}
        loss_weights = {}
        datasets = {}

        # For each split, create a lazy-loading dataset.
        for split_key in ["train", "val", "test"]:
            # Create HDF5 files if they don't exist.
            h5_identifier_dir = os.path.join(self.h5_dir, identifier)
            h5_file = os.path.join(h5_identifier_dir, f"{split_key}_data.h5")
            if not os.path.exists(h5_file):
                self.create_h5_file(
                    identifier,
                    split_key,
                    augmented_images_dir,
                    augmented_gt_label_path,
                    generated_masks_dir,
                    batch_size_processing_img,
                )
            else:
                self.logger.info(f"Using existing HDF5 file for {split_key} set.")

            self.logger.info(f"Building dataset for {split_key} set...")

            dataset = H5Dataset(h5_file)
            datasets[split_key] = dataset

            dataloaders[split_key] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_key == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            # For loss weights, load only the class labels (should be much smaller in size).
            with h5py.File(h5_file, "r") as f:
                class_labels = f["class_label"][:]
            counter = Counter(class_labels)
            loss_weights[split_key] = calculate_loss_weights(counter, len(class_labels))

            self.logger.info(f"Number of samples in {split_key} set: {len(dataset)}")
            self.logger.info(f"{split_key.upper()} class distribution: {counter}")
            self.logger.info(
                f"{split_key.upper()} loss weights: {loss_weights[split_key]}\n"
            )

        # Build the worst-group test set using indices computed from the test HDF5 file.
        test_h5_file = os.path.join(h5_identifier_dir, "test_data.h5")
        with h5py.File(test_h5_file, "r") as f:
            test_class_labels = f["class_label"][:]
            test_confounder = f["confounder_label"][:]

        worst_group_indices = np.where(
            (test_class_labels == self.worst_group_class_label) & (test_confounder == 0)
        )[0].tolist()

        if len(worst_group_indices) > 0:
            worst_group_set = torch.utils.data.Subset(
                datasets["test"], worst_group_indices
            )
            dataloaders["worst_group_test"] = DataLoader(
                worst_group_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                prefetch_factor=3,
            )
            self.logger.info(
                f"Number of samples in worst-group test set: {len(worst_group_indices)}\n"
            )
        else:
            dataloaders["worst_group_test"] = None
            self.logger.warning("Worst-group test set is empty.")

        return dataloaders, loss_weights


# Custom Dataset that reads from an HDF5 file lazily.
class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.data = None  # This will be lazily initialized per worker
        # Get dataset length by opening the file briefly.
        with h5py.File(h5_path, "r") as f:
            self.length = f["images"].shape[0]

    def _init_file(self):
        # In multi-worker setups, each worker should open its own file handle.
        if self.data is None:
            self.data = h5py.File(self.h5_path, "r")

    def __getitem__(self, index):
        self._init_file()

        # Retrieve the sample from the HDF5 file.
        image = self.data["images"][index]
        class_label = self.data["class_label"][index]
        confounder_label = self.data["confounder_label"][index]
        wr_mask = self.data["wr_masks"][index]
        rr_mask = self.data["rr_masks"][index]
        gen_mask = self.data["gen_masks"][index]

        # Convert arrays to PyTorch tensors.
        image = torch.tensor(image, dtype=torch.float32)
        class_label = torch.tensor(class_label, dtype=torch.long)
        confounder_label = torch.tensor(confounder_label, dtype=torch.long)
        wr_mask = torch.tensor(wr_mask, dtype=torch.float32)
        rr_mask = torch.tensor(rr_mask, dtype=torch.float32)
        gen_mask = torch.tensor(gen_mask, dtype=torch.float32)

        return image, class_label, confounder_label, wr_mask, rr_mask, gen_mask

    def __len__(self):
        return self.length

    def __del__(self):
        if self.data is not None:
            self.data.close()
