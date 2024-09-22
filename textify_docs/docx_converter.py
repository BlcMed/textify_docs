from docx import Document

from .base import BaseConverter


class DocxConverter(BaseConverter):
    """
    Converts a DOCX file to plain text.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The plain text content of the DOCX file.
    """

    def convert_to_text(self, file_path):
        try:
            document = Document(file_path)
            text = []
            for paragraph in document.paragraphs:
                text.append(paragraph.text)
            text = "\n".join(text)
            text = "\n".join(line for line in text.splitlines() if line.strip())
            return text

        except FileNotFoundError as e:
            raise Exception(f"An error occurred while converting the DOCX file: {e}")
