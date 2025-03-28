import os
import sys
import yaml
import toml
import csv
import argparse
import pandas as pd
from huggingface_hub import login


from pipeline.pipe_confounder_classification.confounder_classification_sgl import (
    build_pred_csv_sgl_with_backend,
)

from utils import create_file_names_list_from_csv, get_images_from_list, MyLogger

login("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")

parser = argparse.ArgumentParser(description="Perform confounder classification")

parser.add_argument(
    "--config", required=True, type=str, help="Path to the configuration file"
)

parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="0",
    help="Device to use for classification. Default: 0",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # get dir of repo
    repo_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Load the configuration file
    config = toml.load(args.config)

    # get id that is between the first "_" and "." in the config file name
    id_tag = args.config.split("/")[-1].split("_", 1)[1].split(".")[0]

    # set up logger
    logger_dir = f"{repo_dir}/pipeline/logs"
    os.makedirs(logger_dir, exist_ok=True)
    logger = MyLogger(f"{logger_dir}/{id_tag}.log").logger

    # check if confounder classification should be skipped
    if "confounder_classification" not in config:
        logger.info(
            "No confounder classification specified in config file. This step will be skipped. Exiting."
        )
        sys.exit(0)

    # get dataset specific data from datasets_info.yaml
    dataset_name = config["dataset"]["name"]
    with open(f"{repo_dir}/data/datasets_info.yaml", "r") as datasets_info_yaml:
        datasets_info = yaml.safe_load(datasets_info_yaml)
        dataset_info = datasets_info[dataset_name]

    # get gt label path
    gt_label_path = dataset_info["gt_label_path"]

    # set target path for confounder classification csv
    target_csv_path = f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{id_tag}/confounder_pred.csv"

    # check if existing confounder classification should be used
    if "use_existing" in config["confounder_classification"]:
        ex_conf_class_id = config["confounder_classification"]["use_existing"]
        logger.info(
            f"Using existing confounder classification with id {ex_conf_class_id}"
        )
        if ex_conf_class_id == "gt":
            # create csv from gt csv
            logger.info("Creating csv from gt csv...")
            with open(target_csv_path, "w", newline="") as csvfile:
                fieldnames = ["filename", "label", "outputs"]
                writer = csv.DictWriter(target_csv_path, fieldnames=fieldnames)
                writer.writeheader()

                # read gt csv
                gt_label_df = pd.read_csv(gt_label_path)

                # iterate through gt csv and write to new csv
                for _, row in gt_label_df.iterrows():
                    writer.writerow(
                        {
                            "filename": row["filename"],
                            "label": row["confounder_label"],
                            "outputs": "gt label",
                        }
                    )
            logger.info(f"CSV created at {target_csv_path}")
        else:
            # create symlink to existing confounder classification csv
            source_csv_path = f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{ex_conf_class_id}/confounder_pred.csv"
            if not os.path.exists(source_csv_path):
                logger.error(
                    f"Existing confounder classification with id {ex_conf_class_id} not found. Exiting."
                )
                sys.exit(1)

            os.symlink(source_csv_path, target_csv_path)
            logger.info(f"Symlink created from {source_csv_path} to {target_csv_path}.")
        logger.info("Exiting confounder classification.")
        sys.exit(0)

    # if not using existing confounder classification, perform confounder classification
    logger.info("Performing confounder classification...")

    ##### get image paths - start
    # get image names that should be classified (images in train set)
    to_be_classified = create_file_names_list_from_csv(
        gt_label_path, filter_func=lambda row: row[3] == "0"
    )
    # get already processed images (if any) and remove them from to_be_classified
    if os.path.exists(target_csv_path):
        already_processed = create_file_names_list_from_csv(target_csv_path)
        missing_files = list(set(to_be_classified) - set(already_processed))
        logger.info(
            f"Already classified {len(already_processed)} images. {len(missing_files)}/{len(to_be_classified)} images remaining."
        )
        to_be_classified = missing_files

    # get image paths
    image_paths = get_images_from_list(
        to_be_classified,
        dataset_info["raw_dir"],
        sub_dir=True,
        allowed_file_types=["jpg", "jpeg", "png"],
    )
    ##### get image paths - end

    # make sure output directory exists
    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)

    # get model/method for confounder classification
    model_name = config["confounder_classification"]["model"]
    logger.info(f"Using model {model_name} for confounder classification.")
    if model_name == "sgl":
        model_path = config["confounder_classification"][model_name]["model_path"]
        tokenizer_path = config["confounder_classification"][model_name].get(
            "tokenizer_path", None
        )
        chat_template = config["confounder_classification"][model_name].get(
            "chat_template", None
        )
        # Build the prediction CSV using sgl
        build_pred_csv_sgl_with_backend(
            target_csv_path,
            image_paths,
            config["confounder_classification"][model_name]["user_input_list"],
            config["confounder_classification"]["output_for_true"],
            batch_size=config["confounder_classification"]["batch_size"],
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            chat_template=chat_template,
            device=args.device,
            resize_size=config["confounder_classification"].get("resize_size", None),
            logger=logger,
        )
    elif model_name == "gpt4":
        # Build the prediction CSV using gpt4
        logger.info(
            f"Model {model_name} not implemented. Need to be transformed to batch processing. Exiting."
        )
    else:
        logger.error(f"Unknown model: {model_name}. Exiting.")
        sys.exit(1)
