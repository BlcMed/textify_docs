from ..models.table_transformer import load_table_model, infer_objects
from ..config import TABLE_STRUCTURE_RECOGNITION_MODEL_NAME
from ..models.preprocessing import structure_recognition_transform


structure_recognition_model = load_table_model(TABLE_STRUCTURE_RECOGNITION_MODEL_NAME)

def recognize_table(table_image):
    cells = infer_objects(table_image, structure_recognition_model, structure_recognition_transform)
    cell_coordinates = get_cell_coordinates_by_row(cells)
    return cell_coordinates


def get_cell_coordinates_by_row(cells):
    # Extract rows and columns
    rows = [entry for entry in cells if entry['label'] == 'table row']
    columns = [entry for entry in cells if entry['label'] == 'table column']
    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])
    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox
    # Generate cell coordinates and count cells in each row
    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})
        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])
        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})
    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])
    return cell_coordinates
