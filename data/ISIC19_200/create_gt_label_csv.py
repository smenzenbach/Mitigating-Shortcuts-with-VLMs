import csv
import os
import pandas as pd

from utils import create_file_names_list
import numpy as np

# create csv file
# repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_directory = os.path.dirname(os.path.realpath(__file__))
csv_path = f"{current_directory}/gt_label.csv"

# get confounded images through gt segmentations (wr_masks)
confounded_images = create_file_names_list(f"{current_directory}/wr_masks")
images = create_file_names_list(f"{current_directory}/raw")

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["filename", "class_label", "confounder_label", "split"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # get class label through meta csv
    df = pd.read_csv("/workspace/datasets/ISIC19/meta.csv")
    # get indices for the split in train, val and test
    total_entries = len(images)
    indices = np.arange(total_entries)
    np.random.shuffle(indices)

    train_split = int(0.8 * total_entries)
    val_split = int(0.9 * total_entries)  # 10% validation, 10% test

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    for index, image_name in enumerate(images):
        image_id = image_name.replace(".jpg", "")
        row = df[df["isic_id"] == image_id].iloc[0]
        filename = row["isic_id"] + ".jpg"
        class_label = int(row["benign_malignant"] == "malignant")
        confounder_label = int(filename in confounded_images)

        if index in train_indices:
            split = 0
        elif index in val_indices:
            split = 1
        else:
            split = 2

        writer.writerow(
            {
                "filename": filename,
                "class_label": class_label,
                "confounder_label": confounder_label,
                "split": split,
            }
        )
