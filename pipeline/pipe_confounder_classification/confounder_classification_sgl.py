import os
import sglang as sgl
from sglang.lang.chat_template import get_chat_template
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io


from utils import (
    sort_csv_by_column,
    create_label_from_output,
    file_name_from_file_path,
    chunks,
)


@sgl.function
def _image_multi_turn(s, image, user_input_list):
    """

    Returns the generated answers to the user input questions.
    The user input questions are stored in a list. The number of items in the list determine the number QA turns.

    Parameters
    ----------
    image : str, bytes, or PIL.Image
        The image to be processed. The image can be a file path, bytes, or a PIL.Image object (not tested).
    user_input_list : list
        A list of user input questions.

    Returns
    -------
    list
        A list of generated answers.
    """
    s += sgl.user(sgl.image(image))

    for i, user_input in enumerate(user_input_list):
        s += sgl.user(user_input)
        s += sgl.assistant(sgl.gen(f"output_{i+1}", max_tokens=256, temperature=0))


def run_instance_sgl(img_path, user_input_list, output_for_true):

    # Generate answers for the user input questions
    state = _image_multi_turn.run(img_path, user_input_list)
    outputs = [state[f"output_{i+1}"] for i in range(len(user_input_list))]

    # Find predicted label
    label = create_label_from_output(outputs[-1], output_for_true)

    return outputs, label


def build_pred_csv_sgl_batch(
    csv_path,
    image_paths,
    user_input_list,
    output_for_true,
    batch_size=50,
    resize_size=None,  # Add resize dimensions as a parameter
    logger=None,
):
    """
    Generate a csv file with the image file names and their corresponding labels (1 for confounder and 0 for no confounder).

    User inputs are provided in a list. The number of items in the list determine the number of QA turns.

    ### Using sgl:
    Ensure that the sgl server is running before calling this function. Set the runtime endpoint to the sgl server before calling this function.

    Parameters
    ----------
    csv_path : str
        The path to save the csv file.
    image_paths: list of str
        A list of paths to the images.
    user_input_list : list
        A list of user input questions given to the sgl function.
    output_for_true : list
        A list of outputs that correspond to a label of 1 (confounder is present).
    batch_size : int, optional
        The number of images to process in one batch, by default 50.
    resize_size : tuple, optional
        Target size for image resizing (width, height), by default None.

    Returns
    -------
    None
    """

    logged_resize = False

    for batch_index, paths_batch in enumerate(
        tqdm(
            chunks(image_paths, batch_size),
            total=(len(image_paths) + batch_size - 1) // batch_size,
        )
    ):
        # Check if the CSV file already exists
        if os.path.exists(csv_path):
            existing_data = pd.read_csv(csv_path)
        else:
            existing_data = pd.DataFrame(columns=["filename", "label", "outputs"])
        print(
            f"Processing batch {batch_index + 1}/{(len(image_paths) + batch_size - 1) // batch_size}"
        )
        print(f"Generate input for model")
        inp = []
        for image_path in paths_batch:
            image_file_name = file_name_from_file_path(image_path)

            if resize_size is None:
                inp.append({"image": image_path, "user_input_list": user_input_list})

            else:
                if not logged_resize:
                    if logger is not None:
                        logger.info(f"Resizing images to {resize_size}")
                    else:
                        print(f"Resizing images to {resize_size} on longer side")
                    logged_resize = True
                # Open image, resize it in memory and convert to bytes
                with Image.open(image_path) as img:
                    img.thumbnail((resize_size, resize_size), Image.LANCZOS)
                    img_resized = img
                    img_byte_arr = io.BytesIO()
                    img_resized.save(img_byte_arr, format="JPEG")
                    img_byte_arr = img_byte_arr.getvalue()  # Image as bytes

                inp.append({"image": img_byte_arr, "user_input_list": user_input_list})

        # Generate answers for the user input questions
        states = _image_multi_turn.run_batch(inp)

        new_data = []

        for idx, image_path in enumerate(tqdm(paths_batch, total=len(paths_batch))):
            image_file_name = file_name_from_file_path(image_path)
            outputs = [
                states[idx][f"output_{i+1}"] for i in range(len(user_input_list))
            ]

            # Find predicted label
            label = create_label_from_output(outputs[-1], output_for_true)

            new_data.append([image_file_name, label, outputs])

        # Save intermediate results
        new_data_df = pd.DataFrame(new_data, columns=["filename", "label", "outputs"])

        combined_data_df = pd.concat([existing_data, new_data_df]).drop_duplicates(
            subset=["filename"]
        )

        # Sort combined data by image_file_name
        combined_data_df.sort_values(by="filename", inplace=True)
        combined_data_df.to_csv(csv_path, index=False)
        print(
            f"Saved batch {batch_index + 1}/{(len(image_paths) + batch_size - 1) // batch_size} to {csv_path}"
        )

    # sort the final csv by filename
    # sort_csv_by_column(csv_path, "filename")

    print(f"All data saved to {csv_path}")


def build_pred_csv_sgl_with_backend(
    csv_path,
    image_paths,
    user_input_list,
    output_for_true,
    batch_size=50,
    model_path="liuhaotian/llava-v1.6-vicuna-7b",
    tokenizer_path=None,  # "llava-hf/llava-1.5-7b-hf",
    chat_template=None,  # "vicuna_v1.1",
    device="0",
    resize_size=None,
    logger=None,
):
    """
    Generate a csv file with the image file names and their corresponding labels (1 for confounder and 0 for no confounder).

    User inputs are provided in a list. The number of items in the list determine the number of QA turns.

    Parameters
    ----------
    csv_path : str
        The path to save the csv file.
    image_paths: list of str
        A list of paths to the images.
    user_input_list : list
        A list of user input questions given to the sgl function.
    output_for_true : list
        A list of outputs that correspond to a label of 1 (confounder is present).
    batch_size : int, optional
        The number of images to process in one batch, by default 50.
    model_path : str, optional
        The path to the model, by default "liuhaotian/llava-v1.6-vicuna-7b".
    tokenizer_path : str, optional
        The path to the tokenizer, by default "llava-hf/llava-1.5-7b-hf".
    chat_template : str, optional
        The chat template to use, by default "vicuna_v1.1".
    """

    # Set the device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    mp.set_start_method("spawn", force=True)
    if tokenizer_path is None:
        tokenizer_path = model_path
    runtime = sgl.Runtime(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        trust_remote_code=True,
        mem_fraction_static=0.9,
    )
    if chat_template is not None:
        runtime.endpoint.chat_template = get_chat_template(chat_template)
    sgl.set_default_backend(runtime)

    build_pred_csv_sgl_batch(
        csv_path,
        image_paths,
        user_input_list,
        output_for_true,
        batch_size,
        resize_size=resize_size,
        logger=logger,
    )

    runtime.shutdown()
