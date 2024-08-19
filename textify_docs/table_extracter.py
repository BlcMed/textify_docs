import sys
sys.path.insert(0,"textify_docs/")

import pytesseract
import numpy as np
import cv2
from PIL import Image

from table_detecter import detect_tables
from table_structure_recognizer import recognize_table

def extract_from_image(image):
    image = image.convert("RGB")
    tables_images  = detect_tables(image)
    print('-'*40)
    print(f'we got {len(tables_images)} table images')
    tables=[]
    for table_image in tables_images:
        cells_coordinates = recognize_table(table_image=table_image)
        table = _apply_ocr_to_cells(cells_coordinates=cells_coordinates,table_image=table_image)
        print('-'*40)
        print(f'we got {len(cells_coordinates)} rows')
        print(f'we got {len(cells_coordinates[0]["cells"])} columns')
        table_text = _flatten_dict_to_text(table)
        print(table_text)
        tables.append(table_text)
    return tables

def _flatten_dict_to_text(data_dict):
    """
    Flattens a dictionary representing tabular data into a readable text format.

    Args:
        data_dict (dict): Dictionary where keys are row numbers and values are lists of row data.

    Returns:
        str: Flattened text representation of the table.
    """
    text_representation = ""
    for row_num, row_data in data_dict.items():
        row_str = ", ".join(row_data) + ".\n"
        text_representation += row_str + " "
    
    return text_representation.strip()



def _apply_ocr_to_cells(cells_coordinates, table_image, progress_callback=None):
    """
    Returns:
        data (dict): Dictionary where keys are row numbers and values are lists of row data.
    """
    data = dict()
    max_num_columns = 0
    total_rows = len(cells_coordinates)
    
    for idx, row in enumerate(cells_coordinates):
        if progress_callback:
            progress_callback(idx, total_rows)
        
        row_text = []
        for cell in row["cells"]:
            # Crop cell out of image
            cell_image = np.array(table_image.crop(cell["cell"]))
            # Convert cell image to grayscale
            gray_image = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
            # Perform OCR using pytesseract
            text = pytesseract.image_to_string(gray_image)
            # Append OCR result to row text
            row_text.append(text.strip())

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
    image = Image.open("./documents/png.png")
    tables = extract_from_image(image=image)