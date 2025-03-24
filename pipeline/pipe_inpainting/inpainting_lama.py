import os, sys
import numpy as np
from tqdm import tqdm
import tempfile
import cv2
import subprocess


# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils import file_name_from_file_path, inpainting_preprocessing


def run_lama(imgs_path, masks_path, output_dir):
    init_model = "lama"  # @param ['lama', 'sd1.5', 'paint_by_example', 'ldm']
    device = "cuda"  # @param ['cuda', 'cpu', 'mps']

    # Construct the command as a list of arguments
    command = [
        "iopaint",
        "run",
        f"--model={init_model}",
        f"--device={device}",
        f"--image={imgs_path}",
        f"--mask={masks_path}",
        f"--output={output_dir}",
    ]

    # Run the command
    subprocess.run(command)


def inpaint_batch_lama(
    mask_path_list,
    image_path_list,
    kernel_size=5,
    iterations=4,
    dilation_size=None,
    save_dir=None,
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define the path for the temporary mask file
        masks_temp_path = os.path.join(tmpdirname, f"masks")
        imgs_temp_path = os.path.join(tmpdirname, f"images")

        # Create the directories
        os.makedirs(masks_temp_path, exist_ok=True)
        os.makedirs(imgs_temp_path, exist_ok=True)

        # Save the dilated mask and images to the temporary file
        for mask_path, image_path in tqdm(zip(mask_path_list, image_path_list)):
            file_name = file_name_from_file_path(mask_path)

            # check if already inpainted
            inpainting_file = os.path.join(save_dir, file_name.replace(".jpg", ".png"))
            img_temp_path = os.path.join(imgs_temp_path, file_name)
            mask_temp_path = os.path.join(masks_temp_path, file_name)
            if inpainting_file is not None and os.path.exists(inpainting_file):
                print(f"Already inpainted {inpainting_file}")
                continue

            # get image and dilated mask
            img, dilated_mask = inpainting_preprocessing(
                image_path, mask_path, dilation_size, kernel_size, iterations
            )

            cv2.imwrite(img_temp_path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_temp_path, np.array(dilated_mask))

        run_lama(
            img_temp_path,
            mask_temp_path,
            save_dir,
        )
