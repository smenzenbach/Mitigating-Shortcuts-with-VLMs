import csv
import os
import pandas as pd

# read this csv with header "image_id,5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young"
# and create a new csv file with the header "filename,class_label,confounder_label" that has as label has as class label (blond hair=1, dark hair=0) and as confounded label the label of the confounder (male= 0, female=1)

# create csv file
current_directory = os.path.dirname(os.path.realpath(__file__))
csv_path = f"{current_directory}/gt_label.csv"
with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["filename", "class_label", "confounder_label", "split"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Read the CSV files
    attr_df = pd.read_csv(f"{current_directory}/celebA_v1.0/list_attr_celeba.csv")
    split_df = pd.read_csv(f"{current_directory}/celebA_v1.0/list_eval_partition.csv")

    # Merge the dataframes on image_id
    merged_df = pd.merge(attr_df, split_df, on="image_id")

    # Iterate through the merged dataframe
    for _, row in merged_df.iterrows():
        filename = row["image_id"]
        # blond hair = 1, not blond hair = 0
        class_label = 1 if row["Blond_Hair"] == 1 else 0
        # female = 1, male = 0
        confounder_label = 0 if row["Male"] == 1 else 1
        split = row["partition"]
        writer.writerow(
            {
                "filename": filename,
                "class_label": class_label,
                "confounder_label": confounder_label,
                "split": split,
            }
        )
