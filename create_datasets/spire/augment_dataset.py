import numpy as np
import yaml
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import argparse
import utils
import csv
import random

def augment_dataset(dataset_name, augmented_dir, no_class_change):
    with open(f"data/datasets_info.yaml", "r") as file:
        dataset_yaml = yaml.safe_load(file)
    # set raw and gt paths
    raw_dir = dataset_yaml[dataset_name]["raw_dir"]
    wr_masks_dir = dataset_yaml[dataset_name]["wr_masks_dir"]
    rr_masks_dir = dataset_yaml[dataset_name]["rr_masks_dir"]
    # get gt metadata
    gt_label_path = dataset_yaml[dataset_name]["gt_label_path"]
    metadata = pd.read_csv(gt_label_path)

    groups = utils.get_groups(metadata) # already only contains training examples
    if no_class_change:
        # only modify spurious objects (but do that always)
        sampling_probs = {'B2M': 1, 'B2S': 0, 'M2N': 0, 'S2N': 1, 'N2M': 0, 'N2S': 1, 'M2B': 1, 'S2B': 0}
    else:
        sampling_probs = utils.compute_augmentations(groups)

    augmented_csv_info = []

    # iterate over all images from the respective categories
    for file_name in groups['both']:
        # check whether any augmentation is going to be applied:
        rm_conf = np.random.rand() <= sampling_probs['B2M']
        rm_class = np.random.rand() <= sampling_probs['B2S']
        if not rm_conf and not rm_class:
            continue

        # load image and masks:
        raw_img_path = os.path.join(raw_dir, file_name)
        wr_mask_path = os.path.join(wr_masks_dir, file_name)
        rr_mask_path = os.path.join(rr_masks_dir, file_name)

        image = Image.open(raw_img_path).convert("RGB")
        image = np.array(image.resize((299, 299), resample=Image.LANCZOS))
        if os.path.exists(wr_mask_path):
            if not dataset_name == "ISIC19":
                wr_mask = Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
                wr_mask = np.array(wr_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                wr_mask = np.array(Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            wr_mask = None
        if os.path.exists(rr_mask_path):
            if not dataset_name == "ISIC19":
                rr_mask = Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
                rr_mask = np.array(rr_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                rr_mask = np.array(Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            rr_mask = None

        aug_imgs = []
        aug_class_labels = []
        aug_confounder_labels = []
        # create one augmented version with rr and wr removed if it exists
        if rm_conf:
            if wr_mask is None:
                print(f"Warning: No wr mask found for {file_name} (group: both)")
            else:
                aug_imgs.append(utils.remove_object(image, wr_mask))
                aug_class_labels.append(1)
                aug_confounder_labels.append(0)
        if rm_class:
            if rr_mask is None:
                print(f"Warning: No rr mask found for {file_name} (group: both)")
            else:
                aug_imgs.append(utils.remove_object(image, rr_mask))
                aug_class_labels.append(0)
                aug_confounder_labels.append(1)

        # save the new images
        for i, aug_img in enumerate(aug_imgs):
            new_img_path = os.path.join(augmented_dir, f"{file_name[:-4]}_aug_{i}.jpg")
            aug_img.save(new_img_path)
            augmented_csv_info.append({'filename': f'{file_name[:-4]}_aug_{i}.jpg', 'class_label': aug_class_labels[i], 'confounder_label': aug_confounder_labels[i], 'split': 0})

    for file_name in groups['just_class']:
        # check whether any augmentation is going to be applied:
        rm_class = np.random.rand() <= sampling_probs['M2N']
        add_conf = np.random.rand() <= sampling_probs['M2B']
        if not add_conf and not rm_class:
            continue

        # load image and masks:
        raw_img_path = os.path.join(raw_dir, file_name)
        wr_mask_path = os.path.join(wr_masks_dir, file_name)
        rr_mask_path = os.path.join(rr_masks_dir, file_name)

        image = Image.open(raw_img_path).convert("RGB")
        image = np.array(image.resize((299, 299), resample=Image.LANCZOS))

        if os.path.exists(wr_mask_path):
            if not dataset_name == "ISIC19":
                wr_mask = Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
                wr_mask = np.array(wr_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                wr_mask = np.array(Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            wr_mask = None
        if os.path.exists(rr_mask_path):
            if not dataset_name == "ISIC19":
                rr_mask = Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
                rr_mask = np.array(rr_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                rr_mask = np.array(Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            rr_mask = None

        aug_imgs = []
        aug_class_labels = []
        aug_confounder_labels = []
        # create one augmented version with rr and wr removed if it exists
        if add_conf:
            # select a random image wich contains a confounder
            reference_image_name = random.choice(groups['confounded'])
            # load reference image and mask
            ref_image = Image.open(os.path.join(raw_dir, reference_image_name)).convert("RGB")
            ref_image = np.array(ref_image.resize((299, 299), resample=Image.LANCZOS))
            if not dataset_name == "ISIC19":
                ref_mask = Image.open(os.path.join(wr_masks_dir, reference_image_name)).convert("L").point(lambda x: 0 if x < 128 else 255)
                ref_mask = np.array(ref_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                ref_mask = np.array(Image.open(os.path.join(wr_masks_dir, reference_image_name)).convert("L").point(lambda x: 0 if x < 128 else 255)) 
            aug_imgs.append(utils.add_object(image, ref_mask, ref_image))
            aug_class_labels.append(1)
            aug_confounder_labels.append(1)
        if rm_class:
            if rr_mask is None:
                print(f"Warning: No wr mask found for {file_name} (group: just class)")
            else:
                aug_imgs.append(utils.remove_object(image, rr_mask))
                aug_class_labels.append(0)
                aug_confounder_labels.append(0)

        # save the new images
        for i, aug_img in enumerate(aug_imgs):
            new_img_path = os.path.join(augmented_dir, f"{file_name[:-4]}_aug_{i}.jpg")
            aug_img.save(new_img_path)
            augmented_csv_info.append({'filename': f'{file_name[:-4]}_aug_{i}.jpg', 'class_label': aug_class_labels[i], 'confounder_label': aug_confounder_labels[i], 'split': 0})

    for file_name in groups['just_confounded']:
        # check whether any augmentation is going to be applied:
        rm_conf = np.random.rand() <= sampling_probs['S2N']
        add_class = np.random.rand() <= sampling_probs['S2B']
        if not rm_conf and not add_class:
            continue

        # load image and masks:
        raw_img_path = os.path.join(raw_dir, file_name)
        wr_mask_path = os.path.join(wr_masks_dir, file_name)
        rr_mask_path = os.path.join(rr_masks_dir, file_name)

        image = Image.open(raw_img_path).convert("RGB")
        image = np.array(image.resize((299, 299), resample=Image.LANCZOS))
        if os.path.exists(wr_mask_path):
            if not dataset_name == "ISIC19":
                wr_mask = Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
                wr_mask = np.array(wr_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                wr_mask = np.array(Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            wr_mask = None
        if os.path.exists(rr_mask_path):
            if not dataset_name == "ISIC19":
                rr_mask = Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
                rr_mask = np.array(rr_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                rr_mask = np.array(Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            rr_mask = None

        aug_imgs = []
        aug_class_labels = []
        aug_confounder_labels = []
        # create one augmented version with rr and wr removed if it exists
        if rm_conf:
            if wr_mask is None:
                print(f"Warning: No wr mask found for {file_name} (group: just confounded)")
            else:
                aug_imgs.append(utils.remove_object(image, wr_mask))
                aug_class_labels.append(0)
                aug_confounder_labels.append(0)
        if add_class:
            # select a random image wich contains the class
            reference_image_name = random.choice(groups['class'], 1)
            # load reference image and mask     
            ref_image = Image.open(os.path.join(raw_dir, reference_image_name)).convert("RGB")
            ref_image = np.array(ref_image.resize((299, 299), resample=Image.LANCZOS))
            if not dataset_name == "ISIC 19":
                ref_mask = Image.open(os.path.join(rr_masks_dir, reference_image_name)).convert("L").point(lambda x: 0 if x < 128 else 255)
                ref_mask = np.array(ref_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                ref_mask = np.array(Image.open(os.path.join(rr_masks_dir, reference_image_name)).convert("L").point(lambda x: 0 if x < 128 else 255)) 
            aug_imgs.append(utils.add_object(image, ref_mask, ref_image))
            aug_class_labels.append(1)
            aug_confounder_labels.append(1)

        # save the new images
        for i, aug_img in enumerate(aug_imgs):
            new_img_path = os.path.join(augmented_dir, f"{file_name[:-4]}_aug_{i}.jpg")
            aug_img.save(new_img_path)
            augmented_csv_info.append({'filename': f'{file_name[:-4]}_aug_{i}.jpg', 'class_label': aug_class_labels[i], 'confounder_label': aug_confounder_labels[i], 'split': 0})

    for file_name in groups['neither']:
        # check whether any augmentation is going to be applied:
        add_conf = np.random.rand() <= sampling_probs['N2S']
        add_class = np.random.rand() <= sampling_probs['N2M']
        if not add_conf and not add_class:
            continue

        # load image and masks:
        raw_img_path = os.path.join(raw_dir, file_name)
        wr_mask_path = os.path.join(wr_masks_dir, file_name)
        rr_mask_path = os.path.join(rr_masks_dir, file_name)

        image = Image.open(raw_img_path).convert("RGB")
        image = np.array(image.resize((299, 299), resample=Image.LANCZOS))
        if os.path.exists(wr_mask_path):
            if not dataset_name == "ISIC19":
                wr_mask = Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
                wr_mask = np.array(wr_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                wr_mask = np.array(Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            wr_mask = None
        if os.path.exists(rr_mask_path):
            if not dataset_name == "ISIC19":
                rr_mask = Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255)
                rr_mask = np.array(rr_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                rr_mask = np.array(Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            rr_mask = None

        aug_imgs = []
        aug_class_labels = []
        aug_confounder_labels = []
        # create one augmented version with rr and wr removed if it exists
        if add_conf:
            # select a random image wich contains a confounder
            reference_image_name = random.choice(groups['confounded'])
            # load reference image and mask
            ref_image = Image.open(os.path.join(raw_dir, reference_image_name)).convert("RGB")
            ref_image = np.array(ref_image.resize((299, 299), resample=Image.LANCZOS))
            if not dataset_name == "ISIC19":
                ref_mask = Image.open(os.path.join(wr_masks_dir, reference_image_name)).convert("L").point(lambda x: 0 if x < 128 else 255)
                ref_mask = np.array(ref_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                ref_mask = np.array(Image.open(os.path.join(wr_masks_dir, reference_image_name)).convert("L").point(lambda x: 0 if x < 128 else 255)) 
            aug_imgs.append(utils.add_object(image, ref_mask, ref_image))
            aug_class_labels.append(0)
            aug_confounder_labels.append(1)
        if add_class:
            # select a random image wich contains a confounder
            reference_image_name = random.choice(groups['class'])
            # load reference image and mask
            ref_image = Image.open(os.path.join(raw_dir, reference_image_name)).convert("RGB")
            ref_image = np.array(ref_image.resize((299, 299), resample=Image.LANCZOS))
            if not dataset_name == "ISIC19":
                ref_mask = Image.open(os.path.join(rr_masks_dir, reference_image_name)).convert("L").point(lambda x: 0 if x < 128 else 255)
                ref_mask = np.array(ref_mask.resize((299, 299), resample=Image.LANCZOS))
            else:
                ref_mask = np.array(Image.open(os.path.join(rr_masks_dir, reference_image_name)).convert("L").point(lambda x: 0 if x < 128 else 255)) 
            aug_imgs.append(utils.add_object(image, ref_mask, ref_image))
            aug_class_labels.append(1)
            aug_confounder_labels.append(0)

        # save the new images
        for i, aug_img in enumerate(aug_imgs):
            new_img_path = os.path.join(augmented_dir, f"{file_name[:-4]}_aug_{i}.jpg")
            aug_img.save(new_img_path)
            augmented_csv_info.append({'filename': f'{file_name[:-4]}_aug_{i}.jpg', 'class_label': aug_class_labels[i], 'confounder_label': aug_confounder_labels[i], 'split': 0})
    
    with open(os.path.join(augmented_dir, "gt_labels.csv"), mode="w", newline="") as file:
        fieldnames = ["filename", "class_label", "confounder_label", "split"]  # Column headers
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()  # Write column names
        writer.writerows(augmented_csv_info)  # Write data



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create an augmented version of the dataset for Spire")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset config", default="coco")
    parser.add_argument("--augmented_dir", type=str, help="Path to the augmented directory", default="data/coco/augmented/spire")
    parser.add_argument("-no_class_change", action="store_true", help="Whether the class attribute can be modified by Spire")

    args = parser.parse_args()

    augment_dataset(args.dataset_name, args.augmented_dir, args.no_class_change)