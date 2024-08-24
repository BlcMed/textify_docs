from torchvision import transforms
import torch
from ..utils import *
from ..config import (DETECTION_CLASS_THRESHOLDS, PADDING)


detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_tables(image, detection_model=detection_model):
    size = image.size
    tables=[]
    pixel_values = detection_transform(image)
    pixel_values = detection_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = detection_model(pixel_values)
    
    objects = outputs_to_objects(outputs, size, detection_id2label)
    
    tables = objects_to_crops(image, objects, DETECTION_CLASS_THRESHOLDS, padding=PADDING)
    return tables


def objects_to_crops(image, objects, class_thresholds=DETECTION_CLASS_THRESHOLDS, padding=PADDING):
    """
    Crop regions from an image based on detected objects and their bounding boxes, applying class-specific thresholds and optional padding.
    
    Args:
        img (PIL.Image.Image): The original image from which to crop regions.
        objects (list): A list of dictionaries, where each dictionary represents a detected object with keys:
            - 'label': The class label of the object.
            - 'score': The confidence score of the detection.
            - 'bbox': The bounding box coordinates (x_min, y_min, x_max, y_max) for the object.
        class_thresholds (dict): A dictionary mapping class labels to a confidence score threshold. Objects with scores below the threshold will be ignored.
        padding (int, optional): The number of pixels to add as padding around each bounding box when cropping. Default is 10.
    
    Returns:
        list: A list of cropped image regions (PIL.Image.Image) corresponding to objects that meet the class-specific score thresholds.
    """

    crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue
        bbox = obj['bbox']
        padded_bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
        cropped_img = image.crop(padded_bbox)
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
        crops.append({'image': cropped_img, 'bbox': padded_bbox})
    return crops
