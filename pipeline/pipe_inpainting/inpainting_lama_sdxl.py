import os, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import tempfile
import cv2


# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils import file_name_from_file_path, inpainting_preprocessing, extract_index
from inpainting_sdxl import run_sdxl
from inpainting_lama import run_lama


def inpaint_instance_lama_sdxl(
    mask_path,
    image_path,
    prompt,
    pipeline,
    generator,
    iterations=4,
    ratio=60,
    save_dir_intermediate=None,
    save_dir_final=None,
):
    image_fname = file_name_from_file_path(image_path)
    mask_fname = file_name_from_file_path(mask_path)
    assert extract_index(image_fname) == extract_index(
        mask_fname
    ), f"Image and mask file names do not match\n{image_fname}\n{mask_fname}"

    # check if already inpainted
    if save_dir_final is not None and os.path.exists(save_dir_final):
        print(f"Already inpainted {save_dir_final}")
        return

    # get image and dilated mask
    img, dilated_mask = inpainting_preprocessing(
        image_path, mask_path, iterations, ratio
    )

    with tempfile.TemporaryDirectory() as tmpdirname:

        # Define the path for the temporary mask file to save result for second iteration
        mask_temp_path = os.path.join(tmpdirname, "mask.jpg")
        img_temp_path = os.path.join(tmpdirname, "image.jpg")
        if save_dir_intermediate:
            output_dir = save_dir_intermediate
        else:
            output_dir = os.path.join(tmpdirname, "output")

        # Save the dilated mask to the temporary file
        cv2.imwrite(mask_temp_path, np.array(dilated_mask))
        img_array = np.array(img)
        cv2.imwrite(img_temp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        # run inpainting using lama
        run_lama(img_temp_path, mask_temp_path, output_dir)

        # path of lama inpainted image
        result_lama_path = os.path.join(output_dir, "image.png")

        # print files in output_dir

        #!iopaint run --model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" --device={device} --image={img_url} --mask={mask_path} --output={output_dir_confounder2}

        # run inpainting using sdxl
        result_lama = Image.open(result_lama_path)

        result_sdxl = run_sdxl(
            result_lama,
            dilated_mask,
            prompt,
            pipeline,
            generator,
        )

        # save or return final results
        if save_dir_final is not None:
            save_path_final = os.path.join(save_dir_final, image_fname)
            result_sdxl.save(save_path_final)
            print(f"\033[31m Saved {save_path_final} \033[0m")
        else:
            return result_lama, result_sdxl


def inpaint_batch_lama_sdxl(
    mask_path_list,
    image_path_list,
    prompt,
    pipeline,
    generator,
    iterations=4,
    ratio=60,
    save_dir_intermediate=None,
    save_dir_final=None,
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define the path for the temporary mask file
        masks_temp_path = os.path.join(tmpdirname, f"masks")
        imgs_temp_path = os.path.join(tmpdirname, f"images")

        # Create the directories
        os.makedirs(masks_temp_path, exist_ok=True)
        os.makedirs(imgs_temp_path, exist_ok=True)

        # Save the dilated mask and images to the temporary file
        for mask_path, image_path in tqdm(
            zip(mask_path_list, image_path_list)
        ):  # error mask and image not fitting
            mask_fname = file_name_from_file_path(mask_path)
            image_fname = file_name_from_file_path(image_path)
            assert extract_index(image_fname) == extract_index(
                mask_fname
            ), f"Image and mask file names do not match\n{image_fname}\n{mask_fname}"

            mask_temp_path = os.path.join(masks_temp_path, mask_fname)
            img_temp_path = os.path.join(imgs_temp_path, image_fname)

            # check if already inpainted
            final_inpainting_file = os.path.join(save_dir_final, image_fname)
            if final_inpainting_file is not None and os.path.exists(
                final_inpainting_file
            ):
                print(f"Already inpainted {final_inpainting_file}")
                continue

            # get image and dilated mask
            img, dilated_mask = inpainting_preprocessing(
                image_path, mask_path, iterations, ratio
            )

            cv2.imwrite(img_temp_path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_temp_path, np.array(dilated_mask))

        if save_dir_intermediate:
            output_dir = save_dir_intermediate
        else:
            output_dir = os.path.join(tmpdirname, "output")

        # run inpainting using lama
        run_lama(imgs_temp_path, masks_temp_path, output_dir)

        for mask_path, image_path in tqdm(zip(mask_path_list, image_path_list)):
            mask_fname = file_name_from_file_path(mask_path)
            image_fname = file_name_from_file_path(image_path)

            # check if already inpainted (necesarry to check again)
            final_inpainting_file = os.path.join(save_dir_final, image_fname)
            if final_inpainting_file is not None and os.path.exists(
                final_inpainting_file
            ):
                print(f"Already inpainted {final_inpainting_file}")
                continue

            # paths for second iteration (path of lama inpainted image, dilated mask and original image)
            inpainted_path = os.path.join(
                output_dir, image_fname.replace(".jpg", ".png")
            )
            mask_path = os.path.join(masks_temp_path, mask_fname)
            img_path = os.path.join(imgs_temp_path, image_fname)

            #!iopaint run --model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" --device={device} --image={img_url} --mask={mask_path} --output={output_dir_confounder2}

            # Run inpainting using sdxl
            result_lama = Image.open(inpainted_path)
            dilated_mask = Image.open(mask_path).convert("L")

            result_sdxl = run_sdxl(
                result_lama,
                dilated_mask,
                prompt,
                pipeline,
                generator,
            )

            # resize sdxl result to original image size
            result_sdxl = result_sdxl.resize(Image.open(img_path).size)

            # save or return final results
            if save_dir_final is not None:
                save_path_final = os.path.join(save_dir_final, image_fname)
                result_sdxl.save(save_path_final)
                print(f"\033[31m Saved {save_path_final} \033[0m")
            else:
                return result_lama, result_sdxl
