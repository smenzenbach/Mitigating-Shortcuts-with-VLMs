import numpy as np
import yaml
import pandas as pd
import tqdm
import os
from PIL import Image
import argparse
import utils
import csv
import random


def augment_dataset(dataset_name, augmented_dir, no_class_change):
    with open(f"/data/datasets.yaml", "r") as file:
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
        # only add modify spurious objects (but always)
        sampling_probs = {'B2M': 1, 'B2S': 0, 'M2N': 0, 'S2N': 1, 'N2M': 0, 'N2S': 1, 'M2B': 1, 'S2B': 0}
    else:
        sampling_probs = utils.get_sampling_probabilities(groups)

    augmented_csv_info = []

    # iterate over all images from the respective categories
    for file_name in groups['both']:
        # check whether any augmentation is going to be applied:
        if np.random.rand() <= sampling_probs['B2M']:
            # remove a confounder
            rm_conf = True
        if np.random.rand() <= sampling_probs['B2S']:
            # remove a class label
            rm_class = True
        if not rm_conf and not rm_class:
            continue

        # load image and masks:
        raw_img_path = os.path.join(raw_dir, file_name)
        wr_mask_path = os.path.join(wr_masks_dir, file_name)
        rr_mask_path = os.path.join(rr_masks_dir, file_name)

        image = np.array(Image.open(raw_img_path).convert("RGB"))
        if os.path.exists(wr_mask_path):
            wr_mask = np.array(Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            wr_mask = None
        if os.path.exists(rr_mask_path):
            rr_mask = np.array(Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            rr_mask = None

        aug_imgs = []
        aug_class_labels = []
        aug_confounder_labels = []
        # create one augmented version with rr and wr removed if it exists
        if rm_conf == 1:
            aug_imgs.append(utils.remove_object(image, wr_mask))
            aug_class_labels.append(1)
            aug_confounder_labels.append(0)
        if rm_class == 1:
            aug_imgs.append(utils.remove_object(image, rr_mask))
            aug_class_labels.append(0)
            aug_confounder_labels.append(1)

        # save the new images
        for i, aug_img in enumerate(aug_imgs):
            new_img_path = os.path.join(augmented_dir, f"{file_name[:-4]}_aug_{i}.jpg")
            Image.fromarray(aug_img).save(new_img_path)
            augmented_csv_info.append({'filename': f'{file_name[:-4]}_aug_{i}.jpg', 'class_label': aug_class_labels[i], 'confounder_label': aug_confounder_labels[i], 'split': 0})

    for file_name in groups['just_class']:
        # check whether any augmentation is going to be applied:
        if np.random.rand() <= sampling_probs['M2N']:
            # remove the class
            rm_class = True
        if np.random.rand() <= sampling_probs['M2B']:
            # add a confounder
            add_conf = True
        if not add_conf and not rm_class:
            continue

        # load image and masks:
        raw_img_path = os.path.join(raw_dir, file_name)
        wr_mask_path = os.path.join(wr_masks_dir, file_name)
        rr_mask_path = os.path.join(rr_masks_dir, file_name)

        image = np.array(Image.open(raw_img_path).convert("RGB"))
        if os.path.exists(wr_mask_path):
            wr_mask = np.array(Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            wr_mask = None
        if os.path.exists(rr_mask_path):
            rr_mask = np.array(Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            rr_mask = None

        aug_imgs = []
        aug_class_labels = []
        aug_confounder_labels = []
        # create one augmented version with rr and wr removed if it exists
        if add_conf == 1:
            # select a random image wich contains a confounder
            reference_image_name = random.choice(groups['confounded'])
            # load reference image and mask
            ref_image = np.array(Image.open(os.path.join(raw_dir, reference_image_name)))
            ref_mask = np.array(Image.open(os.path.join(wr_masks_dir, reference_image_name))) 
            aug_imgs.append(utils.add_object(image, ref_mask, ref_image))
            aug_class_labels.append(1)
            aug_confounder_labels.append(1)
        if rm_class == 1:
            aug_imgs.append(utils.remove_object(image, rr_mask))
            aug_class_labels.append(0)
            aug_confounder_labels.append(0)

        # save the new images
        for i, aug_img in enumerate(aug_imgs):
            new_img_path = os.path.join(augmented_dir, f"{file_name[:-4]}_aug_{i}.jpg")
            Image.fromarray(aug_img).save(new_img_path)
            augmented_csv_info.append({'filename': f'{file_name[:-4]}_aug_{i}.jpg', 'class_label': aug_class_labels[i], 'confounder_label': aug_confounder_labels[i], 'split': 0})


    for file_name in groups['just_confounded']:
        # check whether any augmentation is going to be applied:
        if np.random.rand() <= sampling_probs['S2N']:
            # remove the confounder
            rm_conf = True
        if np.random.rand() <= sampling_probs['S2B']:
            # add the class
            add_class = True
        if not rm_conf and not add_class:
            continue

        # load image and masks:
        raw_img_path = os.path.join(raw_dir, file_name)
        wr_mask_path = os.path.join(wr_masks_dir, file_name)
        rr_mask_path = os.path.join(rr_masks_dir, file_name)

        image = np.array(Image.open(raw_img_path).convert("RGB"))
        if os.path.exists(wr_mask_path):
            wr_mask = np.array(Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            wr_mask = None
        if os.path.exists(rr_mask_path):
            rr_mask = np.array(Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            rr_mask = None

        aug_imgs = []
        aug_class_labels = []
        aug_confounder_labels = []
        # create one augmented version with rr and wr removed if it exists
        if rm_conf == 1:
            aug_imgs.append(utils.remove_object(image, wr_mask))
            aug_class_labels.append(0)
            aug_confounder_labels.append(0)
        if add_class == 1:
            # select a random image wich contains the class
            reference_image_name = random.choice(groups['class'], 1)
            # load reference image and mask
            ref_image = np.array(Image.open(os.path.join(raw_dir, reference_image_name)))
            ref_mask = np.array(Image.open(os.path.join(rr_masks_dir, reference_image_name))) 
            aug_imgs.append(utils.add_object(image, ref_mask, ref_image))
            aug_class_labels.append(1)
            aug_confounder_labels.append(1)

        # save the new images
        for i, aug_img in enumerate(aug_imgs):
            new_img_path = os.path.join(augmented_dir, f"{file_name[:-4]}_aug_{i}.jpg")
            Image.fromarray(aug_img).save(new_img_path)
            augmented_csv_info.append({'filename': f'{file_name[:-4]}_aug_{i}.jpg', 'class_label': aug_class_labels[i], 'confounder_label': aug_confounder_labels[i], 'split': 0})

    for file_name in groups['neither']:
        # check whether any augmentation is going to be applied:
        if np.random.rand() <= sampling_probs['N2S']:
            # remove the confounder
            add_conf = True
        if np.random.rand() <= sampling_probs['N2M']:
            # add the class
            add_class = True
        if not add_conf and not add_class:
            continue

        # load image and masks:
        raw_img_path = os.path.join(raw_dir, file_name)
        wr_mask_path = os.path.join(wr_masks_dir, file_name)
        rr_mask_path = os.path.join(rr_masks_dir, file_name)

        image = np.array(Image.open(raw_img_path).convert("RGB"))
        if os.path.exists(wr_mask_path):
            wr_mask = np.array(Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            wr_mask = None
        if os.path.exists(rr_mask_path):
            rr_mask = np.array(Image.open(rr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            rr_mask = None

        aug_imgs = []
        aug_class_labels = []
        aug_confounder_labels = []
        # create one augmented version with rr and wr removed if it exists
        if add_conf == 1:
            # select a random image wich contains a confounder
            reference_image_name = random.choice(groups['confounded'])
            # load reference image and mask
            ref_image = np.array(Image.open(os.path.join(raw_dir, reference_image_name)))
            ref_mask = np.array(Image.open(os.path.join(wr_masks_dir, reference_image_name))) 
            aug_imgs.append(utils.add_object(image, ref_mask, ref_image))
            aug_class_labels.append(0)
            aug_confounder_labels.append(1)
        if add_class == 1:
            # select a random image wich contains a confounder
            reference_image_name = random.choice(groups['class'])
            # load reference image and mask
            ref_image = np.array(Image.open(os.path.join(raw_dir, reference_image_name)))
            ref_mask = np.array(Image.open(os.path.join(rr_masks_dir, reference_image_name))) 
            aug_imgs.append(utils.add_object(image, ref_mask, ref_image))
            aug_class_labels.append(1)
            aug_confounder_labels.append(0)

        # save the new images
        for i, aug_img in enumerate(aug_imgs):
            new_img_path = os.path.join(augmented_dir, f"{file_name[:-4]}_aug_{i}.jpg")
            Image.fromarray(aug_img).save(new_img_path)
            augmented_csv_info.append({'filename': f'{file_name[:-4]}_aug_{i}.jpg', 'class_label': aug_class_labels[i], 'confounder_label': aug_confounder_labels[i], 'split': 0})
    
    with open(os.path.join(augmented_dir, "gt_labels.csv"), mode="w", newline="") as file:
        fieldnames = ["filename", "class_label", "confounder_label", "split"]  # Column headers
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()  # Write column names
        writer.writerows(augmented_csv_info)  # Write data



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create an augmented version of the dataset for Spire")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset config")
    parser.add_argument("augmented_dir", type=str, help="Path to the augmented directory")
    parser.add_argument("--no_class_change", action="store_true", help="Whether the class attribute can be modified by Spire")

    args = parser.parse_args()

    augment_dataset(args.dataset_name, args.augmented_dir, args.no_class_change)