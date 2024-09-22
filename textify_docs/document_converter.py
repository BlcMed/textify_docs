"""
This module provides the DocumentConverter class, which facilitates the conversion
of various document formats into plain text. Supported formats include DOCX, PDF,
various image formats (PNG, JPG, etc.), and Excel files (XLS, XLSX).
"""

import os

from .base import BaseConverter
from .docx_converter import DocxConverter
from .image_converter import ImageConverter
from .pdf_converter import PDFConverter
from .xlsx_converter import XlsxConverter


class DocumentConverter(BaseConverter):
    """
    A class for converting documents to plain text using specific converters for each supported format.
    """

    def __init__(self):
        """
        Initialize the DocumentConverter with instances of all supported converters.
        """
        self.image_converter = ImageConverter()
        self.pdf_converter = PDFConverter(self.image_converter)
        self.docx_converter = DocxConverter()
        self.excel_converter = XlsxConverter()
        self.converters = {
            ".docx": self.docx_converter,
            ".pdf": self.pdf_converter,
            ".png": self.image_converter,
            ".jpg": self.image_converter,
            ".jpeg": self.image_converter,
            ".bmp": self.image_converter,
            ".tiff": self.image_converter,
            ".xls": self.excel_converter,
            ".xlsx": self.excel_converter,
        }

    def convert_to_text(self, file_path):
        """
        Convert the specified document to plain text using the appropriate converter.

        :param file_path: Path to the document file.
        :return: A string containing the plain text extracted from the document.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        try:
            return self.converters[file_extension](file_path)
        except KeyError as exc:
            raise ValueError(f"Unsupported file type: {file_extension}.") from exc


if __name__ == "__main__":
    document_converter = DocumentConverter()

    DOCUMENT_PATH = "./data/docx.docx"
    text = document_converter.convert_to_text(DOCUMENT_PATH)
    with open("./data/docx_text.txt", "w", encoding="utf-8") as file:
        file.writelines(text)

    XLSX_PATH = "./data/xlsx.xlsx"
    text = document_converter.convert_to_text(XLSX_PATH)
    print(text)
    with open("./data/xlsx_text.txt", "w", encoding="utf-8") as file:
        file.writelines(text)
