import os
from .docx_converter import DocxConverter
from .pdf_converter import PDFConverter
from .image_converter import ImageConverter
from .xlsx_converter import XlsxConverter

class DocumentConverter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.converter = self._select_converter(file_path)

    def _select_converter(self, file_path):
        """
        Select the appropriate converter based on file extension.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in ['.docx']:
            return DocxConverter(file_path)
        elif file_extension in ['.pdf']:
            return PDFConverter(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return ImageConverter(file_path)
        elif file_extension in ['.xlsx']:
            return XlsxConverter(file_path)
        else:
            raise ValueError("Unsupported file type")

    def convert_to_text(self):
        """
        Convert any kond of document to plain text using the selected converter.
        """
        return self.converter.convert_to_text()
