from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
import torch

#load model
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# Constants
DETECTION_CLASS_THRESHOLDS = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}
PADDING = 10 

def detect_tables(image):
    size = image.size
    tables=[]
    feature_extractor = DetrFeatureExtractor()
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    objects = outputs_to_objects(outputs, size, id2label)
    
    tables = objects_to_crops(image, objects, DETECTION_CLASS_THRESHOLDS, PADDING=PADDING)
    tables = [table['image'].convert("RGB") for table in tables]
    return tables


def objects_to_crops(img, objects, class_thresholds, padding=10):
    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue
        cropped_table = {}
        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
        cropped_img = img.crop(bbox)
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
        cropped_table['image'] = cropped_img
        table_crops.append(cropped_table)
    return table_crops

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]
    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score), 'bbox': [float(elem) for elem in bbox]})
    return objects

    
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
#