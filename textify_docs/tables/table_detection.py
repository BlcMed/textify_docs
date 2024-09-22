"""
This module provides functionality for detecting and cropping tables from images.
It utilizes a pre-trained table detection model to identify table regions and
returns cropped images of detected tables.
"""

from ..config import DETECTION_CLASS_THRESHOLDS, PADDING, TABLE_DETECTION_MODEL_NAME
from ..models.preprocessing import detection_transform
from ..models.table_transformer import infer_objects, load_table_model


def detect_tables(image):
    """
    Detects tables in the given image.

    Args:
        image: The input image to detect tables from.

    Returns:
        tables: A list of cropped table images detected in the input image.
    """
    detection_model = load_table_model(TABLE_DETECTION_MODEL_NAME)
    objects = infer_objects(image, detection_model, detection_transform)
    tables = objects_to_crops(
        image, objects, class_thresholds=DETECTION_CLASS_THRESHOLDS, padding=PADDING
    )
    return tables


def objects_to_crops(image, objects, class_thresholds, padding):
    """
    Crop regions from an image based on detected objects and their bounding boxes,
    applying class-specific thresholds and optional padding.

    Args:
        img (PIL.Image.Image): The original image from which to crop regions.
        objects (list): A list of dictionaries,
            each dictionary represents a detected object with keys:
            - 'label': The class label of the object.
            - 'score': The confidence score of the detection.
            - 'bbox': The bounding box coordinates (x_min, y_min, x_max, y_max) for the object.
        class_thresholds (dict): A dictionary mapping class labels to a confidence score threshold.
            Objects with scores below the threshold will be ignored.
        padding (int): The number of pixels to pad around each bounding box when cropping.

    Returns:
        list: A list of cropped image regions (PIL.Image.Image) corresponding to objects
            that meet the class-specific score thresholds.
    """
    crops = []
    for obj in objects:
        if obj["score"] < class_thresholds[obj["label"]]:
            continue
        bbox = obj["bbox"]
        padded_bbox = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ]
        cropped_img = image.crop(padded_bbox)
        if obj["label"] == "table rotated":
            cropped_img = cropped_img.rotate(270, expand=True)
        crops.append({"image": cropped_img, "bbox": padded_bbox})
    return crops
