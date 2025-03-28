import os, sys
from tqdm import tqdm
import yaml
import toml
import json
import argparse

from utils import (
    create_file_names_list,
    get_images_from_list,
    filter_list_by_csv,
    chunks,
    MyLogger,
)
from pipeline.pipe_inpainting.inpainting_lama_sdxl import inpaint_batch_lama_sdxl
from pipeline.pipe_inpainting.inpainting_lama import inpaint_batch_lama
from pipeline.pipe_inpainting.inpainting_sdxl import (
    inpaint_batch_sdxl,
    setup_diffusion_model,
)

parser = argparse.ArgumentParser(description="Perform inpainting")

parser.add_argument(
    "--config", required=True, type=str, help="Path to the configuration file"
)

parser.add_argument(
    "-d",
    "--device",
    default=2,
    type=int,
    help="Device to run the inpainting on",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # set the device
    # extract number from device string
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # Set the environment variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # get dir of repo
    repo_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Load the configuration file
    config = toml.load(args.config)

    # get id that is between "_" and "." in the config file name
    id_tag = args.config.split("/")[-1].split("_", 1)[1].split(".")[0]

    # set up logger
    logger_dir = f"{repo_dir}/pipeline/logs"
    os.makedirs(logger_dir, exist_ok=True)
    logger = MyLogger(f"{logger_dir}/{id_tag}.log").logger

    # check if inpainting should be skipped
    if "inpainting" not in config:
        logger.info(
            "No inpainting specified in config file. This step will be skipped. Exiting."
        )
        sys.exit(0)

    # get dataset specific data from datasets_info.yaml
    dataset_name = config["dataset"]["name"]
    with open(f"{repo_dir}/data/datasets_info.yaml", "r") as datasets_info_yaml:
        datasets_info = yaml.safe_load(datasets_info_yaml)
        dataset_info = datasets_info[dataset_name]

    # get gt label path
    gt_label_path = dataset_info["gt_label_path"]

    # set target path for inpainting
    target_inpainting_path = (
        f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{id_tag}/inpainting"
    )
    target_intermediate_inpainting_path = f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{id_tag}/inpainting_intermediate"
    os.makedirs(target_inpainting_path, exist_ok=True)
    os.makedirs(target_intermediate_inpainting_path, exist_ok=True)

    # perform inpainting
    logger.info("Performing inpainting...")

    # get segmentation masks path
    seg_path = f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{id_tag}/seg"
    # check if seg_path exists
    if not os.path.exists(seg_path):
        logger.error(
            f"Segmentation path {seg_path} does not exist. Perform segmentation first. Exiting."
        )
        sys.exit(1)

    # set up models
    if "sdxl" in config["inpainting"]["model"]:
        # setup sdxl
        sdxl_pipe, sdxl_generator = setup_diffusion_model(args.device)

    # get inpainting parameters
    dilation_iterations = config["inpainting"].get("dilation_iterations", 5)
    dilation_ratio = config["inpainting"].get("dilation_ratio", 60)
    batch_size = config["inpainting"].get("batch_size", 50)

    # get prompt
    prompts = json.loads(config["inpainting"]["prompts"])
    for prompt_id, key in enumerate(prompts):
        # get prompt and prompt specific parameters
        prompt = prompts[key]["prompt"]
        print(f"Prompt: {prompt}")
        if "filter_func" in prompts[key]:
            filter_function = eval(
                prompts[key]["filter_func"]
            )  # Be cautious with eval!
        else:
            filter_function = None
        if "num_imgs" in prompts[key]:
            num_imgs = prompts[key]["num_imgs"]
        else:
            num_imgs = None

        # get images and masks for the prompt
        file_names = create_file_names_list(seg_path, ignore_copy=True)
        if filter_function is None:
            if num_imgs is None:
                print("Getting all (missing) images and masks in seg file...")
            else:
                print(f"Getting {num_imgs} images and masks from seg file...")
                file_names = file_names[:num_imgs]
        else:  # filter images and masks
            if num_imgs is None:
                print(f"Filtering images and masks (for prompt {key})...")
            else:
                print(
                    f"Filtering images and masks (get {num_imgs} for prompt {key})..."
                )
            print(f"Filter function: {prompts[key]['filter_func']}")

            file_names = filter_list_by_csv(
                file_names,
                gt_label_path,
                filter_func=filter_function,
                ignore_copy=True,
                num_imgs=num_imgs,
            )

        # Sort out the images that are already inpainted and only inpaint the missing ones
        print(f"Check if images are already inpainted and process missing ones...")

        # check if all files are already segmented
        already_inpainted_images = create_file_names_list(target_inpainting_path)
        to_be_inpainted_list = file_names
        missing_files = list(set(to_be_inpainted_list) - set(already_inpainted_images))
        if len(missing_files) == 0:
            print(f"All images of prompt {key} are already inpainted.")
            continue
        else:
            print(
                f"Found {len(missing_files)}/{len(to_be_inpainted_list)} images to process for prompt {key}."
            )
            # only process the missing files
            file_names = missing_files

        images_paths = get_images_from_list(
            file_names, dataset_info["raw_dir"], sub_dir=True
        )
        segmentation_masks = get_images_from_list(
            file_names, seg_path, ignore_copy=True
        )

        # TODO: add this in data_utils functions
        # if args.start is not None:
        #    print(f"Starting from image index {args.start}")
        #    mask_names = [
        #       name
        #        for name in mask_names
        #        if extract_index(name) >= args.start and extract_index(name) < args.end
        #    ]

        for batch_index, (mask_path_list, image_path_list) in enumerate(
            tqdm(
                zip(
                    chunks(segmentation_masks, batch_size),
                    chunks(images_paths, batch_size),
                ),
                total=len(images_paths) // batch_size,
            )
        ):
            if "model" in config["inpainting"]:
                model_name = config["inpainting"]["model"]
            else:
                model_name = None  # default to lama and sdxl
            if model_name == "lama":
                # remove output_dir
                inpaint_batch_lama(
                    mask_path_list,
                    image_path_list,
                    iterations=dilation_iterations,
                    ratio=dilation_ratio,
                    save_dir=target_inpainting_path,
                )
            elif model_name == "sdxl":
                inpaint_batch_sdxl(
                    mask_path_list,
                    image_path_list,
                    prompt,
                    sdxl_pipe,
                    sdxl_generator,
                    iterations=dilation_iterations,
                    ratio=dilation_ratio,
                    save_dir=target_inpainting_path,
                )
            else:
                inpaint_batch_lama_sdxl(
                    mask_path_list,
                    image_path_list,
                    prompt,
                    sdxl_pipe,
                    sdxl_generator,
                    iterations=dilation_iterations,
                    ratio=dilation_ratio,
                    save_dir_intermediate=target_intermediate_inpainting_path,
                    save_dir_final=target_inpainting_path,
                )
            print(
                f"Finished batch {batch_index+1} of {len(images_paths) // batch_size} from prompt {key}: {prompt_id+1} of {len(prompts)}"
            )

    # tmux new -s celebA_inpainting
    # tmux a -t celebA_inpainting
    # python run_inpainting.py --config /workspace/home01/pipeline_configs/CelebA/config_lama_sdxl.toml

    # python3 run_inpainting.py --config /storage-01/ml-stud-share/ml-stud21/pipeline_configs/CelebA/config_lama_sdxl.toml
