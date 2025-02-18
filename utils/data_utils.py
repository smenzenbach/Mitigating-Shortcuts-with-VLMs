import re
import os
from PIL import Image
import warnings
import csv
import h5py
from regex import F


def file_name_from_file_path(filepath):
    """Extracts the file name from a given filepath"""
    # Split the path into components and return the last component
    return os.path.basename(filepath)


def extract_index(filename):
    """
    Extract the index from a filename in the format "text_index.jpg" and "index.jpg". Leading zeros are removed.

    Parameters
    ----------
    filename : str
        The filename to extract the index from.

    Returns
    -------
    int
        The extracted index as an integer.

    Raises
    ------
    Warning
        If the index is not found in the filename. In this case, the index 0 is returned.

    """
    # Check if the filename contains an underscore
    if "_" in filename:
        # remove the text part
        index = filename.split("_")[1]
        # remove the file extension
        index = index.split(".")[0]
    else:
        # remove the file extension
        index = filename.split(".")[0]
    # remove the leading zeros
    index = index.lstrip("0")
    if index == "":
        index = "0"
        warnings.warn(f"Index not found in filename {filename}. Using index 0.")
    index = int(index)
    return index


def create_file_names_list(
    dir_path, sub_dir=False, ignore_copy=False, allowed_file_types=None, num_imgs=None
):
    file_list = []
    counter = 0
    for root, _, files in os.walk(dir_path):
        for file_name in files:
            if ignore_copy and "copy" in file_name.lower():
                continue
            if (
                allowed_file_types is not None
                and file_name.split(".")[-1].lower() not in allowed_file_types
            ):
                continue
            file_list.append(file_name)
            counter += 1
            if num_imgs is not None and counter >= num_imgs:
                break
        if not sub_dir or (num_imgs is not None and counter >= num_imgs):
            break
    if num_imgs is not None and len(file_list) < num_imgs:
        print(
            f"Only {len(file_list)} images found in the directory, less than the requested {num_imgs}."
        )
    if num_imgs is not None:
        assert (
            len(file_list) == num_imgs
        ), f"Expected {num_imgs} images, got {len(file_list)}."
    return sorted(file_list)


def create_file_names_list_from_csv(
    csv_file_path, header=True, filter_func=None, ignore_copy=False, num_imgs=None
):
    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        if header:
            try:
                next(csv_reader)
            except StopIteration:
                print(f"Empty CSV file: {csv_file_path}")
                return []
        file_list = []
        counter = 0
        for row in csv_reader:
            if num_imgs is not None and counter >= num_imgs:
                print(f"{num_imgs} images added to list.")
                break
            if filter_func is None:
                if ignore_copy and "copy" in row[0].lower():
                    continue
                file_list.append(row[0])
                counter += 1
            else:
                if filter_func(row) and (
                    not ignore_copy or "copy" not in row[0].lower()
                ):
                    file_list.append(row[0])
                    counter += 1
        if num_imgs is not None and len(file_list) < num_imgs:
            print(
                f"Only {len(file_list)} images found in the CSV file, less than the requested {num_imgs}."
            )
    if num_imgs is not None:
        assert (
            len(file_list) == num_imgs
        ), f"Expected {num_imgs} images, got {len(file_list)}."
    return file_list


def filter_list_by_csv(
    file_list, csv_file_path, filter_func, header=True, ignore_copy=False, num_imgs=None
):
    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        if header:
            try:
                next(csv_reader)
            except StopIteration:
                print(f"Empty CSV file: {csv_file_path}")
                return []
        filtered_file_list = []
        counter = 0
        for row in csv_reader:
            if num_imgs is not None and counter >= num_imgs:
                print(f"{num_imgs} images added to list.")
                break
            if (
                row[0] in file_list
                and filter_func(row)
                and (not ignore_copy or "copy" not in row[0].lower())
            ):
                filtered_file_list.append(row[0])
                counter += 1
        if num_imgs is not None and len(file_list) < num_imgs:
            print(
                f"Only {len(file_list)} images found in the CSV file, less than the requested {num_imgs}."
            )
    if num_imgs is not None:
        assert (
            len(filtered_file_list) == num_imgs
        ), f"Expected {num_imgs} images, got {len(filtered_file_list)}."
    return filtered_file_list


def get_images_from_list(
    file_name_list,
    dir_path,
    sub_dir=False,
    as_dict=False,
    ignore_copy=True,
    allowed_file_types=None,
    num_imgs=None,
):
    """Returns per default a list of file paths of the files in the given list. If as_dict is True, returns a list of dictionaries with the keys 'file_path', 'file_name' and 'image'."""

    def process_image(image_name, image_path):
        try:
            if not as_dict:
                return image_path
            else:
                image = Image.open(image_path)
                img_dict = {
                    "file_path": image_path,
                    "file_name": image_name,
                    "image": image,
                }
                return img_dict
        except Exception as e:
            print(f"Error loading {image_name}: {e}")

    images = []
    counter = 0
    # Traverse the folder and subfolders (if enabled)
    for root, _, files in os.walk(dir_path):
        if file_name_list is None:
            files = sorted(files)
            for file_name in files:
                if ignore_copy and "copy" in file_name.lower():
                    continue
                if allowed_file_types is not None:
                    if file_name.split(".")[-1].lower() not in allowed_file_types:
                        continue
                file_path = os.path.join(root, file_name)
                images.append(process_image(file_name, file_path))
                counter += 1
                if num_imgs is not None and counter >= num_imgs:
                    break
        else:
            file_name_list = sorted(file_name_list)
            for file_name in file_name_list:
                if file_name in files:
                    if ignore_copy and "copy" in file_name.lower():
                        continue
                    if allowed_file_types is not None:
                        if file_name.split(".")[-1].lower() not in allowed_file_types:
                            continue
                    file_path = os.path.join(root, file_name)
                    images.append(process_image(file_name, file_path))
                    counter += 1
                    if num_imgs is not None and counter >= num_imgs:
                        break

        if not sub_dir or (num_imgs is not None and counter >= num_imgs):
            # Break after processing the top-level folder if not searching subfolders
            break

    # sort images according to filename
    if as_dict:
        images = sorted(images, key=lambda x: x["file_name"])
    else:
        # Function to extract the numeric part from the filename
        def extract_number(path):
            return int(re.search(r"(\d+)\.jpg$", path).group(1))

        # Sorting based on the extracted number
        images = sorted(images, key=extract_number)

    if num_imgs is not None and len(images) < num_imgs:
        print(
            f"Only {len(images)} images found in the list, less than the requested {num_imgs}."
        )
    if num_imgs is not None:
        assert (
            len(images) == num_imgs
        ), f"Expected {num_imgs} images, got {len(images)}."
    return images


def get_images_from_directory(
    dir_path,
    sub_dir=False,
    as_dict=False,
    ignore_copy=True,
    allowed_file_types=None,
    num_imgs=None,
):
    return get_images_from_list(
        None,
        dir_path,
        sub_dir,
        as_dict,
        ignore_copy,
        allowed_file_types=allowed_file_types,
        num_imgs=num_imgs,
    )


def get_images_from_csv(
    csv_file_path,
    dir_path,
    sub_dir=False,
    as_dict=False,
    ignore_copy=True,
    header=True,
    filter_func=None,
    num_imgs=None,
):
    image_names = create_file_names_list_from_csv(
        csv_file_path, header, filter_func, ignore_copy, num_imgs
    )
    if as_dict:
        print(f"{len(image_names)} image dicts added to list from {csv_file_path}.")
    else:
        print(f"{len(image_names)} image paths added to list from {csv_file_path}.")
    return get_images_from_list(image_names, dir_path, sub_dir, as_dict, ignore_copy)


def get_n_confounded_m_non_confounded_images(
    dir_path,
    gt_label_csv_path,
    n,
    m,
    sub_dir=False,
    as_dict=False,
    ignore_copy=True,
    label_column=1,
    allowed_file_types=None,
):
    # Load the ground truth labels from the CSV file
    with open(gt_label_csv_path, "r") as file:
        reader = csv.reader(file)
        # Skip the header row
        next(reader)
        gt_labels = {row[0]: row[label_column] for row in reader}

    # Get the list of image names from csv file (in case images are in subdirectories)
    image_names = list(gt_labels.keys())

    # Get the images from the directory
    images = get_images_from_list(
        image_names, dir_path, sub_dir, as_dict, ignore_copy, allowed_file_types
    )

    # Separate the images into confounded and non-confounded groups
    confounded_images = []
    non_confounded_images = []
    n_confounded = 0
    n_non_confounded = 0
    for image in images:
        if not as_dict:
            image_name = file_name_from_file_path(image)
        else:
            image_name = image["file_name"]
        if gt_labels[image_name] == "1" and n_confounded < n:
            confounded_images.append(image)
            n_confounded += 1
        elif gt_labels[image_name] == "0" and n_non_confounded < m:
            non_confounded_images.append(image)
            n_non_confounded += 1
        if n_confounded > n and n_non_confounded > m:
            break

    assert len(confounded_images) == n
    assert len(non_confounded_images) == m
    return confounded_images, non_confounded_images


def sort_csv_by_column(csv_path, column_index, header=True):
    """
    Sort a CSV file by the values in a specific column. The sorted CSV file will overwrite the original file.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file to be sorted.
    column_index : int
        The index of the column to sort by.
    header : bool, optional
        Whether the CSV file contains a header row with column names, by default True
    """

    # Read the CSV file into a list of rows
    with open(csv_path, "r") as file:
        rows = list(csv.reader(file))

    # Sort the rows based on the values in the specified column
    if header:
        sorted_rows = sorted(rows[1:], key=lambda row: row[column_index])
        # Add the header row back to the sorted list if it was present
        sorted_rows.insert(0, rows[0])
    else:
        sorted_rows = sorted(rows, key=lambda row: row[column_index])

    # Write the sorted rows back to the original CSV file
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(sorted_rows)


def merge_csv_files(output_filename, input_filenames, sort=True, delete_merged=True):
    with open(output_filename, "w", newline="") as out_file:
        writer = csv.writer(out_file)
        header_written = False
        for filename in input_filenames:
            with open(filename, "r") as in_file:
                reader = csv.reader(in_file)
                header = next(reader)
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)
            if delete_merged:
                os.remove(filename)
                print(f"Temporary file {filename} cleaned up.")
        if sort:
            sort_csv_by_column(output_filename, 0, header=True)


def read_h5(h5_file_path):
    with h5py.File(h5_file_path, "r") as f:
        return {key: f[key][()] for key in f.keys()}


def get_config_id_from_path(config_path):
    return config_path.split("/")[-1].split("_", 1)[1].split(".")[0]
