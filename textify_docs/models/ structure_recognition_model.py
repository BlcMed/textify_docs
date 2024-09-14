from transformers import TableTransformerForObjectDetection
from ..config import MODELS_PATH


def load_structure_recognition_model(cache_dir=MODELS_PATH):
    structure_recognition_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all", cache_dir=cache_dir)
    structure_id2label = structure_recognition_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"
    return structure_recognition_model, structure_id2label