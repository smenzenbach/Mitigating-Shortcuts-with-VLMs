# copy 100 malignant and 50 further benign images to /workspace/home/Mitigating-Shortcuts-with-VLMs/data/ISIC19_200
import os
import shutil
import pandas as pd

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
source_dir = "/workspace/datasets/ISIC19/raw"
target_dir = f"{repo_dir}/data/ISIC19_200/wr_masks"

malignant = 100
benign = 50

meta_csv = "/workspace/datasets/ISIC19/meta.csv"

# read meta csv
meta = pd.read_csv(meta_csv)

# get 100 malignant images
malignant_images = meta[meta["benign_malignant"] == "malignant"]["isic_id"].tolist()[
    :malignant
]

# get 50 benign images
benign_images = meta[meta["benign_malignant"] == "benign"]["isic_id"].tolist()[:benign]

# copy images
for image in malignant_images + benign_images:
    shutil.copy(f"{source_dir}/{image}.jpg", f"{target_dir}/{image}.jpg")
