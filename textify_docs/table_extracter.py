from table_detecter import detect_tables
from table_structure_recognizer import recognize_table
import pytesseract
import numpy as np
import cv2
from PIL import Image

def extract_from_image(image):
    tables_images  = detect_tables(image)
    #tables_images = [table['image'] for table in tables]
    tables=[]
    for table_image in tables_images:
        cells_coordinates = recognize_table(table_image=table_image)
        table = _extract_table_data_from_cells(cells_coordinates)
        tables.append(table)

    return tables

def _extract_table_data_from_cells(cell_coordinates, table_image, progress_callback=None):
    data = dict()
    max_num_columns = 0
    total_rows = len(cell_coordinates)
    
    for idx, row in enumerate(cell_coordinates):
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
    image = Image.open("../documents/png.png")
    extract_from_image(image=image)