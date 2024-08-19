import sys
sys.path.insert(0,"textify_docs/")

from torchvision import transforms
import torch
from utils import *

# Constants
DETECTION_CLASS_THRESHOLDS = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}
PADDING = 10 

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
    tables = [table.convert("RGB") for table in tables]
    return tables


def objects_to_crops(img, objects, class_thresholds, padding=10):
    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue
        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
        cropped_img = img.crop(bbox)
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
        table_crops.append(cropped_img)
    return table_crops
