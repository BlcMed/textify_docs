from docx import Document
from .base import BaseConverter

class DocxConverter(BaseConverter):

    def convert_to_text(self, file_path):
        try:
            document = Document(file_path)
            text = []
            
            for paragraph in document.paragraphs:
                text.append(paragraph.text)
            
            text = "\n".join(text)
            text = '\n'.join(line for line in text.splitlines() if line.strip())
            return text
        
        except Exception as e:
            print(f"An error occurred while converting the DOCX file: {e}")
            return None
