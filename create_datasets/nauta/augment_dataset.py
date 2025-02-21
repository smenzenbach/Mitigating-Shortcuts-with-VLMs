import numpy as np
import yaml
import pandas as pd
import tqdm
import os
from PIL import Image
import argparse
import torch

from model import InpaintingModel_GMCNN


def augment_dataset(dataset_name, augmented_dir, model_path):
    with open(f"/data/datasets.yaml", "r") as file:
        dataset_yaml = yaml.safe_load(file)
    # set raw and gt paths
    raw_dir = dataset_yaml[dataset_name]["raw_dir"]
    wr_masks_dir = dataset_yaml[dataset_name]["wr_masks_dir"]
    # get gt metadata
    gt_label_path = dataset_yaml[dataset_name]["gt_label_path"]
    metadata = pd.read_csv(gt_label_path)

    model = torch.load(model_path)
    model.eval()

    # load and modify dataset
    for i, row in tqdm(metadata.iterrows()):
        # check whether the example contains a shortcut and is in the training set
        if confounder_label == 0:
            continue
        if row["split"] != 0:
            continue

        # get paths
        raw_img_path = os.path.join(raw_dir, row["filename"])
        wr_mask_path = os.path.join(wr_masks_dir, row["filename"])

        # load images and masks
        image = np.array(Image.open(raw_img_path).convert("RGB"))
        if os.path.exists(wr_mask_path):
            wr_mask = np.array(Image.open(wr_mask_path).convert("L").point(lambda x: 0 if x < 128 else 255))
        else:
            continue  # no inpainting if there is no wr mask
        confounder_label = int(row["confounder_label"])

        # generate inpainted image
        if confounder_label == 1:
            new_img = model(image, wr_mask)
        
        # save the new img (under the same image as the original image)
        new_img_path = os.path.join(augmented_dir, row['filename'])
        Image.fromarray(new_img).save(new_img_path)

        # TODO check whether the inpainting output is correct and in the correct format


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create inpainted images according to the method of Nauta et al. for a dataset")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset config")
    parser.add_argument("augmented_dir", type=str, help="Path to the augmented directory")
    parser.add_argument("model_dir", type=str, help="Path to the augmented directory")

    args = parser.parse_args()

    augment_dataset(args.dataset_name, args.augmented_dir, args.model_dir)