"""
This module provides functionality for recognizing the structure of tables in images.
It utilizes a pre-trained table structure recognition model to identify rows and columns
within a table, returning their bounding box coordinates.
"""

from ..config import TABLE_STRUCTURE_RECOGNITION_MODEL_NAME
from ..models.preprocessing import structure_recognition_transform
from ..models.table_transformer import infer_objects, load_table_model

structure_recognition_model = load_table_model(TABLE_STRUCTURE_RECOGNITION_MODEL_NAME)


def recognize_table(table_image):
    """
    Processes a table image and returns the coordinates of the cells organized by rows.

    Args:
        table_image (PIL.Image.Image): The input image containing the table to be recognized.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - 'row': Bounding box coordinates of the row.
            - 'cells': A list of dictionaries for each cell, with bounding box coordinates of the cell
              and its corresponding column.
            - 'cell_count': The number of cells in the row.
    """
    cells = infer_objects(
        table_image, structure_recognition_model, structure_recognition_transform
    )
    cell_coordinates = get_cell_coordinates_by_row(cells)
    return cell_coordinates


def get_cell_coordinates_by_row(cells):
    """
    Extracts and organizes cell coordinates from detected rows and columns.

    Args:
        cells (list): A list of dictionaries, where each dictionary represents a detected
            object with keys:
            - 'label': The class label of the object ('table row' or 'table column').
            - 'bbox': Bounding box coordinates (x_min, y_min, x_max, y_max) for the object.

    Returns:
        list: A structured list of dictionaries, where each dictionary contains:
            - 'row': Bounding box coordinates of the row.
            - 'cells': A list of dictionaries for each cell, with bounding box coordinates
                of the cell and its corresponding column.
            - 'cell_count': The number of cells in the row.
    """
    # Extract rows and columns
    rows = [entry for entry in cells if entry["label"] == "table row"]
    columns = [entry for entry in cells if entry["label"] == "table column"]
    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x["bbox"][1])
    columns.sort(key=lambda x: x["bbox"][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [
            column["bbox"][0],
            row["bbox"][1],
            column["bbox"][2],
            row["bbox"][3],
        ]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({"column": column["bbox"], "cell": cell_bbox})
        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x["column"][0])
        # Append row information to cell_coordinates
        cell_coordinates.append(
            {"row": row["bbox"], "cells": row_cells, "cell_count": len(row_cells)}
        )
    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x["row"][1])
    return cell_coordinates
