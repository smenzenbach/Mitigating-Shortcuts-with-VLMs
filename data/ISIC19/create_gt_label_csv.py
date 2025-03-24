import csv
import os
import pandas as pd

from utils import create_file_names_list
import numpy as np

# create csv file
current_directory = os.path.dirname(os.path.realpath(__file__))
csv_path = f"{current_directory}/gt_label.csv"

# get confounded images through gt segmentations
confounded_images = create_file_names_list("/workspace/datasets/ISIC19/segmentation")

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["filename", "class_label", "confounder_label", "split"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # get class label through meta csv
    df = pd.read_csv("/workspace/datasets/ISIC19/meta.csv")
    # get indices for the split in train, val and test
    total_entries = len(df)
    indices = np.arange(total_entries)
    np.random.shuffle(indices)

    train_split = int(0.8 * total_entries)
    val_split = int(0.9 * total_entries)  # 10% validation, 10% test

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    for index, row in df.iterrows():
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
