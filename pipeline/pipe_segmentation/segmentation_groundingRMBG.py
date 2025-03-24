import torch
from PIL import Image
from torchvision.ops import box_convert

import os, sys

path = "/workspace/home/GroundingDINO"
if path not in sys.path:
    sys.path.append(path)


# Add the project root directory to sys.path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import pipeline

# from groundingdino.util.inference import load_model, load_image, predict, annotate

from utils import nms_grounding, remove_big_box_grounding, remove_small_box_grounding


def setup_processor_model_pipeline(device=0):
    # build grounding dino from huggingface
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        f"cuda:{device}"
    )
    # processor = None
    # grounding_model = load_model(
    #     "/workspace/home/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    #     "/workspace/home/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    #     device=f"cuda:{device}",
    # )

    pipe = pipeline(
        "image-segmentation",
        model="briaai/RMBG-1.4",
        trust_remote_code=True,
        device=f"cuda:{device}",
    )

    return processor, grounding_model, pipe


def segment_instance_groundingRMBG(
    img_path,
    custom_vocab,
    detection_model,
    processor,
    segmentation_pipe,
    device=0,
    box_lower_threshold=None,
    box_upper_threshold=None,
    save_path=None,
):
    image = Image.open(img_path)
    text = ""
    for vocab in custom_vocab:
        text += f"{vocab.lower()} . "

    inputs = processor(image, text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detection_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    # image_source, img = load_image(img_path)
    # boxes, logits, phrases = predict(
    #     model=detection_model,
    #     image=img,
    #     caption=text,
    #     box_threshold=0.35,
    #     text_threshold=0.25,
    # )

    # annotated_frame = annotate(image_source, boxes, logits, phrases)

    # Image.fromarray(annotated_frame).show()

    # # adapt boxes so they are they are not percentage values but pixel values
    # h, w, _ = image_source.shape
    # resized_boxes = boxes * torch.Tensor([w, h, w, h])
    # xyxy = box_convert(boxes=resized_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # results = [
    #     {
    #         "scores": logits,
    #         "labels": phrases,
    #         "boxes": xyxy,
    #     }
    # ]

    """
    Results is a list of dict with the following structure:
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
    Boxes are in (x1, y1, x2, y2) format.
    """

    # Post-process the results
    results = nms_grounding(results)
    if box_upper_threshold is not None:
        results = remove_big_box_grounding(results, image.size, box_upper_threshold)
    if box_lower_threshold is not None:
        results = remove_small_box_grounding(results, image.size, box_lower_threshold)
    image.show()

    # Create a blank mask for the original image
    original_mask = Image.new("L", image.size, 0)

    bounding_box_on_image = image.convert("RGBA")

    for box in results[0]["boxes"].cpu().numpy():
        # pre process the image for segmentation
        # crop the image to the bounding box of the object
        cropped_image = image.crop(box)
        cropped_image.show()

        # run the segmentation model
        pil_mask = segmentation_pipe(cropped_image, return_mask=True)

        # make pil_mask a binary mask
        pil_mask = pil_mask.convert("L").point(lambda x: 255 if x > 128 else 0)

        # Resize the segmentation mask to the size of the bounding box
        x1, y1, x2, y2 = map(int, box)
        resized_mask = pil_mask.resize((x2 - x1, y2 - y1))

        # Create a mask of the same size as the original image, filled with 0 (background)
        temp_mask = Image.new("L", image.size, 0)

        # Paste the resized mask into the correct position on the temp_mask
        temp_mask.paste(resized_mask, (x1, y1))

        # Combine the temp_mask with the original_mask using the maximum value (logical OR)
        original_mask = Image.composite(temp_mask, original_mask, temp_mask)

        # Draw the bounding box on the image
        bounding_box_on_image = Image.alpha_composite(
            bounding_box_on_image,
            Image.new("RGBA", bounding_box_on_image.size, (255, 255, 255, 0)),
        )

    assert (
        original_mask.size[0] == image.size[0]
        and original_mask.size[1] == image.size[1]
    )

    # Save the mask or display it
    if save_path:
        original_mask.save(save_path)
    else:
        return original_mask, bounding_box_on_image


def segment_batch_groundingRMBG(
    img_paths,
    custom_vocab,
    detection_model,
    processor,
    segmentation_pipe,
    device=0,
    box_lower_threshold=None,
    box_upper_threshold=None,
):
    """
    WIP
    """
    # create list of images
    images = []
    image_sizes = []
    for img_path in img_paths:
        image = Image.open(img_path)
        images.append(image)
        image_sizes.append(image.size[::-1])

    text = ""
    for vocab in custom_vocab:
        text += f"{vocab.lower()} . "

    inputs = processor(images, text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = detection_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=image_sizes,
    )

    """
    Results is a list of dict with the following structure:
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
    Boxes are in (x1, y1, x2, y2) format.
    """

    # Post-process the results
    results = nms_grounding(results)
    if box_upper_threshold is not None:
        results = remove_big_box_grounding(results, image.size, box_upper_threshold)
    if box_lower_threshold is not None:
        results = remove_small_box_grounding(results, image.size, box_lower_threshold)

    # Create a blank mask for the original image
    original_mask = Image.new("L", image.size, 0)

    bounding_box_on_image = image.convert("RGBA")

    for box in results[0]["boxes"].cpu().numpy():
        # pre process the image for segmentation
        # crop the image to the bounding box of the object
        cropped_image = image.crop(box)
        cropped_image.show()

        # run the segmentation model
        pil_mask = segmentation_pipe(cropped_image, return_mask=True)

        # make pil_mask a binary mask
        pil_mask = pil_mask.convert("L").point(lambda x: 255 if x > 128 else 0)

        # Resize the segmentation mask to the size of the bounding box
        x1, y1, x2, y2 = map(int, box)
        resized_mask = pil_mask.resize((x2 - x1, y2 - y1))

        # Create a mask of the same size as the original image, filled with 0 (background)
        temp_mask = Image.new("L", image.size, 0)

        # Paste the resized mask into the correct position on the temp_mask
        temp_mask.paste(resized_mask, (x1, y1))

        # Combine the temp_mask with the original_mask using the maximum value (logical OR)
        original_mask = Image.composite(temp_mask, original_mask, temp_mask)

        # Draw the bounding box on the image
        bounding_box_on_image = Image.alpha_composite(
            bounding_box_on_image,
            Image.new("RGBA", bounding_box_on_image.size, (255, 255, 255, 0)),
        )

    assert (
        original_mask.size[0] == image.size[0]
        and original_mask.size[1] == image.size[1]
    )

    # save masks
