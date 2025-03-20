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
    coco_dir = "/workspace/datasets/coco"

    for split_key in ["train", "val"]:  # no annotations for test: val = test
        dataType = split_key + "2017"
        annFile = f"{coco_dir}/annotations/instances_{dataType}.json"

        # initialize COCO api for instance annotations
        coco = COCO(annFile)

        # get all img_ids
        imgIds = coco.getImgIds()

        # get all images containing given categories
        catId_person = coco.getCatIds(catNms=["person"])
        catId_racket = coco.getCatIds(catNms=["tennis racket"])
        imgIds_person = coco.getImgIds(catIds=catId_person)
        imgIds_racket = coco.getImgIds(catIds=catId_racket)

        # find intersection of images with person and racket and complements
        imgIds_person = set(imgIds_person)
        imgIds_racket = set(imgIds_racket)
        imgIds_person_racket = imgIds_person.intersection(imgIds_racket)
        imgIds_only_person = imgIds_person - imgIds_person_racket
        imgIds_only_racket = imgIds_racket - imgIds_person_racket

        print(f"Split: {split_key}")
        print(f"Images: {len(imgIds)}")
        print(f"Images with person: {len(imgIds_person)}")
        print(f"Images with racket: {len(imgIds_racket)}")
        print(f"Images with person and racket: {len(imgIds_person_racket)}")
        print(f"Images with only person: {len(imgIds_only_person)}")
        print(f"Images with only racket: {len(imgIds_only_racket)}")

        # write to csv
        for imgId in imgIds:
            filename = str(imgId).zfill(12) + ".jpg"
            class_label = (
                1 if imgId in imgIds_racket else 0
            )  # racket = 1, no racket = 0
            confounder_label = (
                1 if imgId in imgIds_person else 0
            )  # person = 1, no person = 0
            split = 0 if split_key == "train" else 2  # no validation set, so val = test

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
