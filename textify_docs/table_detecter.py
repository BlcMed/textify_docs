from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
from torchvision import transforms
import torch
from utils import *

#load model
#DETECTION_MODEL_PATH = "/models/pubtables1m_detection_detr_r18.pth"
#model = TableTransformerForObjectDetection.from_pretrained(DETECTION_MODEL_PATH)
#detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
#id2label = detection_model.config.id2label
#id2label[len(detection_model.config.id2label)] = "no object"

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
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #model.to(device)
    pixel_values = detection_transform(image)
    pixel_values = detection_transform(image).unsqueeze(0)
    #pixel_values = pixel_values.to(device)
    with torch.no_grad():
        outputs = detection_model(pixel_values)
    
    objects = outputs_to_objects(outputs, size, detection_id2label)
    
    tables = objects_to_crops(image, objects, DETECTION_CLASS_THRESHOLDS, padding=PADDING)
    #tables = [table['image'].convert("RGB") for table in tables]
    tables = [table.convert("RGB") for table in tables]
    return tables


def objects_to_crops(img, objects, class_thresholds, padding=10):
    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue
        #cropped_table = {}
        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
        cropped_img = img.crop(bbox)
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
        #cropped_table['image'] = cropped_img
        #table_crops.append(cropped_table)
        table_crops.append(cropped_img)
    return table_crops
