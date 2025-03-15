from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torchvision


def nms_grounding(
    detection_results: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Perform non-maximum suppression on the detection results.

    detection_results is a list of dict with the following structure:
    [
        {
            'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'),
            'labels': ['car', 'tire', 'tire', 'tire'],
            'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                            [1392.4701,  554.4064, 1628.6133,  777.5872],
                            [ 436.1182,  621.8940,  676.5255,  851.6897],
                            [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
        }
    ]
    """
    # Check if detection_result is empty
    if len(detection_results) == 0:
        return detection_results

    processed_results = []
    for detection_result in detection_results:
        boxes = detection_result["boxes"]
        scores = detection_result["scores"]

        # Perform non-maximum suppression
        # Boxes are expected to be in (x1, y1, x2, y2) format
        nms_indices = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)

        # Filter out the detections that were not selected by non-maximum suppression
        detection_result["boxes"] = boxes[nms_indices]
        detection_result["scores"] = scores[nms_indices]
        detection_result["labels"] = [
            detection_result["labels"][i] for i in nms_indices
        ]

        # add to processed reulsts list
        processed_results.append(detection_result)
    return processed_results


def remove_small_box_grounding(
    detection_results: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    area_threshold: float = 0.1,
):
    """
    Remove detections that fill less than x percent of the image. The percentage is given by the area_threshold.

    detection_result is a list of dict with the following structure:
    [
        {
            'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'),
            'labels': ['car', 'tire', 'tire', 'tire'],
            'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                            [1392.4701,  554.4064, 1628.6133,  777.5872],
                            [ 436.1182,  621.8940,  676.5255,  851.6897],
                            [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
        }
    ]
    """
    return remove_big_box_grounding(
        detection_results, image_shape, area_threshold, remove_small_box=True
    )


def remove_big_box_grounding(
    detection_results: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    area_threshold: float = 0.9,
    remove_small_box: bool = False,
):
    """
    Remove detections that fill more than x percent of the image. The percentage is given by the area_threshold.

    detection_result is a list of dict with the following structure:
    [
        {
            'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'),
            'labels': ['car', 'tire', 'tire', 'tire'],
            'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                            [1392.4701,  554.4064, 1628.6133,  777.5872],
                            [ 436.1182,  621.8940,  676.5255,  851.6897],
                            [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
        }
    ]
    """
    # Check if detection_result is empty
    if len(detection_results) == 0:
        return detection_results

    processed_results = []

    for detection_result in detection_results:
        boxes = detection_result["boxes"].cpu().numpy()

        # Check if there's only one bounding box
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)

        # Filter out the detections that fill the entire image
        filtered_indices = []
        for idx, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            bb_area = width * height
            img_area = image_shape[0] * image_shape[1]
            if remove_small_box:
                if bb_area / img_area > area_threshold:
                    filtered_indices.append(idx)
            else:
                if bb_area / img_area < area_threshold:
                    filtered_indices.append(idx)

        if len(filtered_indices) == 0:
            return [detection_result]

        # Filter out the detections that were not selected
        detection_result["boxes"] = detection_result["boxes"][filtered_indices]
        detection_result["scores"] = detection_result["scores"][filtered_indices]
        detection_result["labels"] = [
            detection_result["labels"][i] for i in filtered_indices
        ]

        # add to processed reulsts list
        processed_results.append(detection_result)
    return processed_results
