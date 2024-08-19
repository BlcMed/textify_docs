from pdf2image import convert_from_path
from .base import BaseConverter
from .image_converter import ImageConverter

class PDFConverter(BaseConverter):

    def __init__(self, file_path):
        super().__init__(file_path)
        self.image_converter = ImageConverter(file_path)

    def convert_to_text(self):
        try:
            images = convert_from_path(self.file_path)
            full_text = []
            
            for i, img in enumerate(images):
                # Preprocess and extract text from each page image
                text = self.image_converter.convert_to_text_from_image(img)
                full_text.append(text)
            
            return "\n".join(full_text)
        
        except Exception as e:
            print(f"An error occurred while converting the PDF file: {e}")
            return None
