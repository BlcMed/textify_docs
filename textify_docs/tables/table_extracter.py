"""
This module provides functionality for extracting tabular data from images.
It utilizes optical character recognition (OCR) to extract text from detected
tables within an image and returns both the textual content and their bounding
boxes.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image

from ..config import TESSERACT_CONFIG_CELL
from .table_detection import detect_tables
from .table_structure_recognition import recognize_table


def extract_tables_from_image(image, language):
    """
    Extract tables from an image and return their textual content
    along with thier bounding boxes.

    Args:
        image (PIL.Image.Image): The input image from which
            tables are to be extracted.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - 'table_text': textual content of the table extracted using OCR.
            - 'bbox': bounding box coordinates (x_min, y_min, x_max, y_max)
                of the table in the image.
    """
    tables_dicts = extract_tables_from_image_as_dict(image, language)
    tables = []
    for table_dict in tables_dicts:
        table_text = _flatten_dict_to_text(table_dict["table_dict"])
        tables.append({"table_text": table_text, "bbox": table_dict["bbox"]})
    return tables


def extract_tables_from_image_as_dict(image, language):
    """
    Extract tables from an image and return their textual content as dictionary
    along with their bounding boxes.

    Args:
        image (PIL.Image.Image): The input image from which
            tables are to be extracted.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - 'table_dict': The tabular data extracted from the table using OCR
            - 'bbox': The bounding box coordinates
                (x_min, y_min, x_max, y_max) of the table in the image.
    """
    image = image.convert("RGB")
    tables_crops = detect_tables(image)
    tables = []
    print("-" * 20)
    print(f"{len(tables_crops)} tables detected in this image:")
    for table_crop in tables_crops:
        table_image = table_crop["image"]
        cells_coordinates = recognize_table(table_image=table_image)
        table_data = _apply_ocr_to_cells(
            cells_coordinates=cells_coordinates,
            table_image=table_image,
            language=language,
        )  # dict
        tables.append({"table_dict": table_data, "bbox": table_crop["bbox"]})
        print(
            f"with {len(cells_coordinates)} row and "
            f'{len(cells_coordinates[0]["cells"])} columns.'
        )
    return tables


def _flatten_dict_to_text(data_dict):
    """
    Flattens a dictionary representing tabular data into
    a readable text format.

    Args:
        data_dict (dict): Dictionary where
            keys are row numbers
            values are lists of row data.

    Returns:
        str: Flattened text representation of the table.
    """
    text_representation = ""
    for _, row_data in data_dict.items():
        row_str = " ; ".join(row_data) + ".\n"
        text_representation += row_str + " "
    return text_representation.strip()


def _apply_ocr_to_cells(
    cells_coordinates, table_image, language, config=TESSERACT_CONFIG_CELL
):
    """
    Returns:
        data (dict): Dictionary where
            keys are row numbers
            values are lists of row data.
    """
    data = {}
    max_num_columns = 0

    for idx, row in enumerate(cells_coordinates):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(table_image.crop(cell["cell"]))
            # Convert cell image to grayscale
            gray_image = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
            text = pytesseract.image_to_string(gray_image, lang=language, config=config)
            text = text.replace("|", "")
            text = text.strip()
            row_text.append(text)
        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)
        data[idx] = row_text
    # Pad rows which don't have max_num_columns elements
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + [""] * (max_num_columns - len(row_data))
        data[row] = row_data
    return data


if __name__ == "__main__":
    image_test = Image.open("./data/png.png")
    tables_test = extract_tables_from_image(image=image_test, language="fra")
