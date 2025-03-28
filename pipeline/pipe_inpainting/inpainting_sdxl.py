from diffusers import AutoPipelineForInpainting
from sympy import im
import torch
import os, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
import tempfile
import cv2
import subprocess
import toml


# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils import file_name_from_file_path, inpainting_preprocessing


def setup_diffusion_model(device):

    device_str = f"cuda:{device}"
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipeline = pipeline.to(device_str)
    pipeline.enable_attention_slicing()  # Recommended if your computer has < 64 GB of RAM

    generator = torch.Generator(device_str).manual_seed(0)
    return pipeline, generator


def run_sdxl(image, mask, prompt, pipeline, generator):
    image_resized = image.resize((1024, 1024))
    mask_resized = mask.resize((1024, 1024))
    # Run the diffuser
    result_sdxl = pipeline(
        prompt=prompt,
        image=image_resized,
        mask_image=mask_resized,
        guidance_scale=30.0,  # 8.0,
        num_inference_steps=25,  # steps between 15 and 30 work well for us (25)
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
        # padding_mask_crop=32,
    ).images[0]
    return result_sdxl.resize(image.size)


def inpaint_instance_sdxl(
    mask_path,
    image_path,
    prompt,
    pipeline,
    generator,
    iterations=4,
    ratio=60,
    save_dir=None,
):
    # Save the dilated mask and images to the temporary file
    file_name = file_name_from_file_path(mask_path)

    # check if already inpainted
    if save_dir is not None and os.path.exists(save_dir):
        print(f"Already inpainted {save_dir}")
        return

    # get image and dilated mask
    img, dilated_mask = inpainting_preprocessing(
        image_path, mask_path, iterations, ratio
    )

    # run inpainting using sdxl
    result_sdxl = run_sdxl(
        img,
        dilated_mask,
        prompt,
        pipeline,
        generator,
    )

    # save or return final results
    if save_dir is not None:
        save_path_final = os.path.join(save_dir, file_name)
        result_sdxl.save(save_path_final)
        print(f"\033[31m Saved {save_path_final} \033[0m")
    else:
        return result_sdxl


def inpaint_batch_sdxl(
    mask_path_list,
    image_path_list,
    prompt,
    pipeline,
    generator,
    iterations=4,
    ratio=60,
    save_dir=None,
):
    # Save the dilated mask and images to the temporary file
    for mask_path, image_path in tqdm(zip(mask_path_list, image_path_list)):
        file_name = file_name_from_file_path(mask_path)

        # check if already inpainted
        final_inpainting_file = os.path.join(save_dir, file_name)
        if final_inpainting_file is not None and os.path.exists(final_inpainting_file):
            print(f"Already inpainted {final_inpainting_file}")
            continue

        # get image and dilated mask
        img, dilated_mask = inpainting_preprocessing(
            image_path, mask_path, iterations, ratio
        )

        # run inpainting using sdxl
        result_sdxl = run_sdxl(
            img,
            dilated_mask,
            prompt,
            pipeline,
            generator,
        )

        # save inpainted image
        result_sdxl.save(final_inpainting_file)
