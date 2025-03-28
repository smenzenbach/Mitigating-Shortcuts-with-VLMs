from pycocotools.coco import COCO
import os
import csv

from utils import sort_csv_by_column


# create csv file
current_directory = os.path.dirname(os.path.realpath(__file__))
csv_path = f"{current_directory}/gt_label.csv"

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["filename", "class_label", "confounder_label", "split"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # coco dir
    coco_dir = "datasets/coco"

    for split_key in ["train", "val"]:  # no annotations for test: val = test
        dataType = split_key + "2017"
        annFile = f"{coco_dir}/annotations/instances_{dataType}.json"

        # initialize COCO api for instance annotations
        coco = COCO(annFile)

        # get all img_ids
        imgIds = coco.getImgIds()

        # get all images containing given categories
        catId_spurious = coco.getCatIds(catNms=["sheep"])
        catId_main = coco.getCatIds(catNms=["dog"])
        imgIds_spurious = coco.getImgIds(catIds=catId_spurious)
        imgIds_main = coco.getImgIds(catIds=catId_main)



        # find intersection of images with person and racket and complements
        imgIds_spurious = set(imgIds_spurious)
        imgIds_main = set(imgIds_main)
        imgIds_both = imgIds_spurious.intersection(imgIds_main)
        imgIds_only_spurious = imgIds_spurious - imgIds_both
        imgIds_only_main = imgIds_main - imgIds_both

        print(f"Split: {split_key}")
        print(f"Images: {len(imgIds)}")
        print(f"Images with spurious: {len(imgIds_spurious)}")
        print(f"Images with main: {len(imgIds_main)}")
        print(f"Images with both: {len(imgIds_both)}")
        print(f"Images with only spurious: {len(imgIds_only_spurious)}")
        print(f"Images with only main: {len(imgIds_only_main)}")

        # write to csv
        for imgId in imgIds:
            filename = str(imgId).zfill(12) + ".jpg"
            class_label = (
                1 if imgId in imgIds_main else 0
            )  # racket = 1, no racket = 0
            confounder_label = (
                1 if imgId in imgIds_spurious else 0
            )  # person = 1, no person = 0
            split = 0 if split_key == "train" else 2  # no validation set, so val = test
            # move all images with only main to test
            # if class_label == 1 and confounder_label == 0:
            #     split = 2


            writer.writerow(
                {
                    "filename": filename,
                    "class_label": class_label,
                    "confounder_label": confounder_label,
                    "split": split,
                }
            )

# sort csv by filename
sort_csv_by_column(csv_path, 0)


