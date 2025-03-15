import os
import sys
from tqdm import tqdm
import yaml
import toml
import argparse


from pipeline.pipe_segmentation.segmentation_groundingRMBG import (
    segment_instance_groundingRMBG,
    setup_processor_model_pipeline,
)

from utils import (
    create_file_names_list_from_csv,
    create_file_names_list,
    get_images_from_list,
    file_name_from_file_path,
    MyLogger,
)

parser = argparse.ArgumentParser(description="Perform segmentation")

parser.add_argument(
    "--config",
    required=True,
    type=str,
    help="Path to the configuration file",
)

parser.add_argument(
    "-d",
    "--device",
    default=0,
    type=int,
    help="GPU device number",
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

    # segmentation cannot be skipped
    if "segmentation" not in config:
        logger.error(
            "No segmentation specified in config file. This step is mandatory. Exiting."
        )
        sys.exit(1)

    # get dataset specific data from datasets_info.yaml
    dataset_name = config["dataset"]["name"]
    with open(f"{repo_dir}/data/datasets_info.yaml", "r") as datasets_info_yaml:
        datasets_info = yaml.safe_load(datasets_info_yaml)
        dataset_info = datasets_info[dataset_name]

    # get gt label path
    gt_label_path = dataset_info["gt_label_path"]

    # set target path for segmentation
    target_seg_path = (
        f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{id_tag}/seg"
    )

    # check if existing segmentation should be used
    if "use_existing" in config["segmentation"]:
        ex_seg_id = config["segmentation"]["use_existing"]
        logger.info(f"Using existing segmentation with id {ex_seg_id}")
        # create symlink to existing segmentation
        if ex_seg_id == "gt":
            source_seg_path = dataset_info["gt_seg_path"]
        else:
            source_seg_path = (
                f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{ex_seg_id}/seg"
            )
        if not os.path.exists(source_seg_path):
            logger.error(
                f"Existing segmentation not found at {source_seg_path}. Exiting."
            )
            sys.exit(1)

        os.symlink(source_seg_path, target_seg_path)
        logger.info(f"Created symlink from {source_seg_path} to {target_seg_path}.")
        logger.info("Exiting segmentation.")
        sys.exit(0)

    # if not using existing segmentation, perform segmentation
    logger.info("Performing segmentation...")

    ##### get image paths - start
    # check if module confounder classification is off
    # if module is off, segment all images
    if "confounder_classification" not in config:
        logger.info(
            "No confounder classification specified in config file. All images will be segmented."
        )
        # TODO: add possibility to give filter function like in inpainting
        to_be_segmented = create_file_names_list_from_csv(
            gt_label_path, filter_func=lambda row: row[3] == "0"
        )  # images in train split
    else:
        pred_label_csv = f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{id_tag}/confounder_pred.csv"
        to_be_segmented = create_file_names_list_from_csv(
            pred_label_csv, filter_func=lambda row: row[1] == "1"
        )  # images with predicted confounder label 1

    # get already segmented images (if any) and remove them from to_be_segmented
    if os.path.exists(target_seg_path):
        already_seg_images = create_file_names_list(target_seg_path)
        missing_files = list(set(to_be_segmented) - set(already_seg_images))
        logger.info(
            f"Already segmented {len(already_seg_images)} images. {len(missing_files)}/{len(to_be_segmented)} images remaining."
        )
        to_be_segmented = missing_files

    # get image paths
    image_paths = get_images_from_list(
        to_be_segmented,
        dataset_info["raw_dir"],
        sub_dir=True,
        allowed_file_types=["jpg", "jpeg", "png"],
    )
    ##### get image paths - end

    # make sure output directory exists
    os.makedirs(target_seg_path, exist_ok=True)

    # get model/method for segmentation
    model_name = config["segmentation"]["model"]
    logger.info(f"Using model {model_name} for segmentation.")

    # get custom vocabulary for detection
    custom_vocabulary = config["segmentation"].get("custom_vocabulary", None)

    # get threshold for bounding box if specified
    box_lower_threshold = config["segmentation"].get("box_lower_threshold", None)
    box_upper_threshold = config["segmentation"].get("box_upper_threshold", None)

    if model_name == "groundingRMBG":
        processor, grounding_model, segmentation_pipe = setup_processor_model_pipeline(
            args.device
        )
    else:
        logger.error(f"Unknown segmentation model: {model_name}. Exiting.")
        sys.exit(1)

    # run segmentation
    for img_path in tqdm(image_paths, desc="Segmenting images", total=len(image_paths)):
        # TODO: using batches?
        # set path were the segmented image should be saved
        save_path = f"{target_seg_path}/{file_name_from_file_path(img_path)}"
        segment_instance_groundingRMBG(
            img_path,
            custom_vocabulary,
            grounding_model,
            processor,
            segmentation_pipe,
            device=args.device,
            save_path=save_path,
            box_lower_threshold=box_lower_threshold,
            box_upper_threshold=box_upper_threshold,
        )
