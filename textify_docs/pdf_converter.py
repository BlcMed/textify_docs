"""
This module provides the PDfConverter class, which converts PDF
documents to plain text by extracting text from each page as an image.
"""

from pdf2image import convert_from_path

from .base import BaseConverter


class PDFConverter(BaseConverter):
    """
    Responsible for converting PDF documents to plain text by extracting
    text from each page as an image.

    Args:
        image_converter (ImageConverter): An instance of an image converter
        used to extract text from images of PDF pages.
    """

    def __init__(self, image_converter):
        """
        Initialize the PDFConverter with a specified image converter.
        """
        self.image_converter = image_converter

    def convert_to_text(self, file_path):
        """
        Convert the specified PDF document to plain text.
        """
        try:
            images = convert_from_path(file_path)
            full_text = []

            for _, img in enumerate(images):
                # Preprocess and extract text from each page image
                text = self.image_converter.extract_text_from_image(img)
                full_text.append(text)

            return "\n".join(full_text)

        except FileNotFoundError:
            print(f"The file was not found: {file_path}")
            return None
        except ValueError as e:
            print(f"Value error encountered: {e}")
            return None
