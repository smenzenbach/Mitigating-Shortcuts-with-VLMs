import os, sys
import csv
import yaml
import warnings
import argparse
import toml

from dataclasses import dataclass

# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import get_images_from_directory, create_file_names_list_from_csv


def build_gt_csv_from_seg_gt(csv_path, images_dir_path, seg_dir_path):
    """
    Build a csv file with ground truth values if a confounder is in the image. 1 if confounder is present, 0 otherwise.
    It is assumed that the confounder is present in the images in the seg_dir_path.

    Parameters
    ----------
    csv_path : str
        The path to save the csv file.
    images_dir_path : str
        The path to the directory of the images of the dataset.
    seg_dir_path : str
        The path to the directory where the segmented images are stored.

    Returns
    -------
    None
    """

    images = get_images_from_directory(images_dir_path, without_img=True, sub_dir=True)
    seg_images = get_images_from_directory(seg_dir_path, without_img=True, sub_dir=True)

    seg_images = [img["file_name"] for img in seg_images]

    with open(csv_path, mode="w") as csv_file:
        writer = csv.writer(csv_file)

        # Write header
        writer.writerow(["image_file_name", "label"])

        for img_dict in images:
            image_file_name = img_dict["file_name"]
            label = 1 if image_file_name in seg_images else 0
            writer.writerow([image_file_name, label])


# dataclass
@dataclass
class EvaluationMetrics:
    accuracy: float
    recall: float
    precision: float
    f1: float
    false_positive: int = 0
    false_negative: int = 0
    true_positive: int = 0
    true_negative: int = 0
    num_imgs: int = 0


def evaluation_metrics(gt_label, predicted_label):
    """
    Compare the ground truth values with the predictions and return the evaluation metrics: accuracy, recall, precision, and f1.

    Parameters
    ----------
    gt_label : list
        A list of ground truth labels.
    predicted_label : list
        A list of predicted labels.

    Returns
    -------
    tuple
        A tuple of the evaluation metrics: accuracy, recall, precision, and f1.
    """

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for gt, pred in zip(gt_label, predicted_label):
        if gt == 1 and pred == 1:
            true_positive += 1
        elif gt == 0 and pred == 1:
            false_positive += 1
        elif gt == 0 and pred == 0:
            true_negative += 1
        elif gt == 1 and pred == 0:
            false_negative += 1

    accuracy = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative
    )
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1 = 2 * (precision * recall) / (precision + recall)

    num_imgs = len(gt_label)

    return EvaluationMetrics(
        accuracy,
        recall,
        precision,
        f1,
        false_positive,
        false_negative,
        true_positive,
        true_negative,
        num_imgs,
    )


def evaluation_metrics_csv(gt_csv, predictions_csv, coco=False):
    """
    Compare the ground truth values with the predictions and return the evaluation metrics: accuracy, recall, precision, and f1.
    It is assumed that the csv files have a header.


    You can use the output like this:
    ```
    metrics = compare_gt_with_predictions("gt.csv", "predictions.csv")
    accuracy = metrics.accuracy
    ```

    Parameters
    ----------
    gt_csv : str
        The path to the ground truth csv file.
    predictions_csv : str
        The path to the predictions csv file.

    Returns
    -------
    EvaluationMetrics
        A dataclass with the evaluation metrics: accuracy, recall, precision, and f1.
        - accuracy: Measures the proportion of true results among the total number of cases examined.
        - recall: Measures the proportion of actual positives that are correctly identified.
        - precision: Measures the proportion of positive identifications that were actually correct.
        - f1: The harmonic mean of precision and recall.
        - false_positive: The number of false positives.
        - false_negative: The number of false negatives.
        - true_positive: The number of true positives.
        - true_negative: The number of true negatives.
        - num_imgs: The number of images in the predictions csv file.

    Warnings
    --------
    If there are images in the ground truth csv that are not in the predictions csv, a warning will be raised.
    """

    with open(gt_csv, mode="r") as gt_file, open(
        predictions_csv, mode="r"
    ) as pred_file:
        gt_reader = csv.reader(gt_file)
        # filter gt reader (only train split)
        gt_reader = filter(lambda row: row[3] == "0", gt_reader)
        pred_reader = csv.reader(pred_file)

        # Skip header
        next(gt_reader)
        next(pred_reader)

        if coco:
            gt_dict = {
                row[0]: int((row[1] == "1") & (row[2] == "1")) for row in gt_reader
            }
        else:
            gt_dict = {
                row[0]: int(row[2]) for row in gt_reader
            }  # confounder label in column 2
        pred_dict = {row[0]: int(row[1]) for row in pred_reader}

        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0

        for img_name, pred_label in pred_dict.items():
            gt_label = gt_dict.get(img_name, None)
            if gt_label is not None and pred_label == 1:
                if gt_label == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            elif gt_label is not None and pred_label == 0:
                if gt_label == 0:
                    true_negative += 1
                else:
                    false_negative += 1

        accuracy = (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative
        )
        recall = true_positive / (true_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        f1 = 2 * (precision * recall) / (precision + recall)

        num_imgs = len(pred_dict)

        # check if there are any images in the ground truth csv that are not in the predictions csv
        if len(gt_dict) > len(pred_dict):
            missing_imgs = set(gt_dict.keys()) - set(pred_dict.keys())
            warnings.warn(
                f"The following {len(missing_imgs)} images are missing in the predictions csv:\n{sorted(missing_imgs)}"
            )

        return EvaluationMetrics(
            accuracy,
            recall,
            precision,
            f1,
            false_positive,
            false_negative,
            true_positive,
            true_negative,
            num_imgs,
        )


def create_evaluation_yaml(yaml_path, metrics, num_error=None):
    """
    Create a yaml file with the evaluation metrics.

    Parameters
    ----------
    yaml_path : str
        The path to save the yaml file.
    metrics : EvaluationMetrics
        A dataclass with the evaluation metrics.
    num_error : int, optional
        The number of errors in the predictions csv file. Default is None.

    Returns
    -------
    None
    """

    data = {
        "accuracy": metrics.accuracy,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "f1": metrics.f1,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "true_positive": metrics.true_positive,
        "true_negative": metrics.true_negative,
        "num_imgs": metrics.num_imgs,
    }

    if num_error is not None:
        data["num_error"] = num_error

    with open(yaml_path, "w") as yaml_file:
        yaml.dump(data, yaml_file)


def count_errors(csv_file):
    """
    Count the number of errors in the csv file.

    Parameters
    ----------
    csv_file : str
        The path to the csv file.

    Returns
    -------
    int
        The number of errors in the csv file.
    """

    with open(csv_file, mode="r") as csv_file:
        reader = csv.reader(csv_file)

        # Skip header
        next(reader)

        error_count = 0
        for row in reader:
            if "error" in row[2].lower():
                error_count += 1

        return error_count


parser = argparse.ArgumentParser(description="Evaluate confounder classification")

parser.add_argument(
    "--config", required=True, type=str, help="Path to the configuration file"
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

    # get dataset specific data from datasets_info.yaml
    dataset_name = config["dataset"]["name"]
    with open(f"{repo_dir}/data/datasets_info.yaml", "r") as datasets_info_yaml:
        datasets_info = yaml.safe_load(datasets_info_yaml)
        dataset_info = datasets_info[dataset_name]

    # get gt label path
    gt_label_path = dataset_info["gt_label_path"]

    # get predicted label path
    pred_label_path = f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{id_tag}/confounder_pred.csv"

    # evaluate the confounder classification
    if config["dataset"]["name"] == "coco":
        metrics = evaluation_metrics_csv(gt_label_path, pred_label_path, coco=True)
    else:
        metrics = evaluation_metrics_csv(gt_label_path, pred_label_path)

    print(metrics)

    # create yaml file with evaluation metrics
    evaluation_yaml_path = f"{repo_dir}/pipeline/pipeline_outputs/{dataset_name}/{id_tag}/confounder_evaluation.yaml"
    create_evaluation_yaml(evaluation_yaml_path, metrics)
