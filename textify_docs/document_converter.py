import os
from .docx_converter import DocxConverter
from .pdf_converter import PDFConverter
from .image_converter import ImageConverter
from .xlsx_converter import XlsxConverter

class DocumentConverter:
    def __init__(self):
        self.converter = None

    def _select_converter(self, file_path):
        """
        Select the appropriate converter based on file extension.
        
        :param file_path: Path to the document file.
        :return: An instance of the appropriate converter.
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

    def convert_to_text(self, file_path):
        """
        Convert the specified document to plain text using the appropriate converter.
        
        :param file_path: Path to the document file.
        :return: A string containing the plain text extracted from the document.
        """
        self.converter = self._select_converter(file_path)
        return self.converter.convert_to_text()
