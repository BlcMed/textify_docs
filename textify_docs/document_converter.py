import os
from .docx_converter import DocxConverter
from .pdf_converter import PDFConverter
from .image_converter import ImageConverter
from .xlsx_converter import XlsxConverter
from .base import BaseConverter

class DocumentConverter(BaseConverter):

    def __init__(self):
        """
        Initialize the DocumentConverter with instances of all supported converters.
        """
        self.image_converter = ImageConverter()
        self.pdf_converter = PDFConverter(self.image_converter)
        self.docx_converter = DocxConverter()
        self.excel_converter = XlsxConverter()

    def convert_to_text(self, file_path):
        """
        Convert the specified document to plain text using the appropriate converter.

        :param file_path: Path to the document file.
        :return: A string containing the plain text extracted from the document.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.docx':
            return self.docx_converter.convert_to_text(file_path)
        elif file_extension == '.pdf':
            return self.pdf_converter.convert_to_text(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return self.image_converter.convert_to_text(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            return self.excel_converter.convert_to_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")


if __name__ == "__main__":
    document_converter = DocumentConverter()

    path="./data/docx.docx"
    text = document_converter.convert_to_text(path)
    with open("./data/docx_text.txt", 'w') as file:
        file.writelines(text) 

    path="./data/xlsx.xlsx"
    text = document_converter.convert_to_text(path)
    print(text)
    with open("./data/xlsx_text.txt", 'w') as file:
        file.writelines(text)