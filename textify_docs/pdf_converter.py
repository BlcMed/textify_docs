from pdf2image import convert_from_path
from .base import BaseConverter

class PDFConverter(BaseConverter):

    def __init__(self, image_converter):
        self.image_converter = image_converter

    def convert_to_text(self, file_path):
        try:
            images = convert_from_path(file_path)
            full_text = []
            
            for i, img in enumerate(images):
                # Preprocess and extract text from each page image
                text = self.image_converter.extract_text_from_image(img)
                full_text.append(text)
            
            return "\n".join(full_text)
        
        except Exception as e:
            print(f"An error occurred while converting the PDF file: {e}")
            return None
