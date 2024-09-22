"""
This module contains the basic utilities needed to work with the TATR models.
"""

import torch
from transformers import TableTransformerForObjectDetection

from ..config import MODELS_PATH


def load_table_model(model_name, cache_dir=MODELS_PATH):
    """
    Loads a pre-trained table detection model.

    Args:
        model_name (str): The name or path of the pre-trained model to load.
        cache_dir (str, optional): Directory to cache the model. Defaults to MODELS_PATH.

    Returns:
        TableTransformerForObjectDetection: The loaded model configured for object detection.
    """
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TableTransformerForObjectDetection.from_pretrained(
        model_name, cache_dir=cache_dir
    ).to(device)
    model.eval()
    # Update id2label dictionary
    id2label = model.config.id2label.copy()
    id2label[len(id2label)] = "no object"
    model.config.id2label = id2label
    return model


def infer_objects(image, model, transform):
    """
    Infers objects in an image using the specified model.

    Args:
        image (PIL.Image.Image): The input image from which to detect objects.
        model (TableTransformerForObjectDetection): The model used for object detection.
        transform (callable): A transformation function to preprocess the image.
    """
    size = image.size
    pixel_values = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(pixel_values)
    id2label = model.config.id2label
    objects = _parse_model_outputs_to_objects(outputs, size, id2label)
    return objects


def _parse_model_outputs_to_objects(outputs, img_size, id2label):
    """
    Convert model outputs into a list of detected objects with their labels, scores, and bounding boxes.
    It is used for both table detection and table structure recognition models.

    Args:
        outputs (dict): A dictionary containing the model outputs,
                        Expected keys are 'logits' and 'pred_boxes'.
        img_size (tuple): A tuple (img_w, img_h) representing the width and height of the image.
        id2label (dict): A dictionary mapping class indices to human-readable labels.

    Returns:
        list: A list of dictionaries, each representing a detected object. Each dictionary contains:
            - 'label': The class label of the detected object.
            - 'score': The confidence score of the detection.
            - 'bbox': The bounding box coordinates (x_min, y_min, x_max, y_max) for the object in the image.
    """
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in _rescale_bboxes(pred_bboxes, img_size)]
    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )
    return objects


def _rescale_bboxes(out_bbox, size):
    """
    Rescale bounding boxes to match the dimensions of the image.

    Args:
        out_bbox (Tensor): Tensor of shape (N, 4), where N is the number of bounding boxes.
                           The tensor contains bounding boxes in (center_x, center_y, width, height) format.
        size (tuple): A tuple (img_w, img_h) representing the width and height of the image.

    Returns:
        Tensor: Tensor of shape (N, 4) with rescaled bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    img_w, img_h = size
    b = _box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def _box_cxcywh_to_xyxy(x):
    """
    Convert bounding box format from (center_x, center_y, width, height) (cxcywh)
    to (x_min, y_min, x_max, y_max) (xyxy).

    Args:
        x (Tensor): Tensor of shape (N, 4), where N is the number of bounding boxes.
                    The tensor contains (center_x, center_y, width, height) for each bounding box.

    Returns:
        Tensor: Tensor of shape (N, 4) with bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
