from transformers import TableTransformerForObjectDetection
from ..config import MODELS_PATH


def load_detection_model(cache_dir=MODELS_PATH):
    detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection",cache_dir=cache_dir)
    detection_id2label = detection_model.config.id2label
    detection_id2label[len(detection_model.config.id2label)] = "no object"
    return detection_model, detection_id2label
