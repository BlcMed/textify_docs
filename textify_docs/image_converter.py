import cv2 as cv
from PIL import Image
import pytesseract
import numpy as np
from .base import BaseConverter
from .tables.table_extracter import extract_tables_from_image
from .config import SEPARATOR


class ImageConverter(BaseConverter):

    def convert_to_text(self, file_path):
        """
        Convert the image file to plain text using OCR by processing tables and gaps in between them chronologically.
        
        Returns:
            str: A string containing the combined plain text from tables and gaps between them, ordered as they appear in the image.
                If an error occurs during processing, returns None.
        """
        try:
            with Image.open(file_path) as img:
                full_text = self.extract_text_from_image(img=img)
                return full_text
        
        except Exception as e:
            # Handle errors and return None
            print(f"An error occurred: {e}")
            return None


    def extract_text_from_image(self, img):
        """
        Convert the given image (in PIL format) to plain text using OCR.
        
        :param img: The preprocessed image in PIL format.
        :return: A string containing the plain text extracted from the image.
        """
        try:

            img = self.preprocess_image(img)
            img_width, img_height = img.size
            tables_crops = extract_tables_from_image(img)
            table_bboxes = sorted([table["bbox"] for table in tables_crops], key=lambda b: b[1])  # Sort by ymin
            full_text = []
            
            # Handle tables and gaps between them that contain simple plain textual data
            previous_ymax = 0
            for bbox, table_crop in zip(table_bboxes, tables_crops):
                ymin, ymax = bbox[1], bbox[3]
                # Extract text from the gap above the current table (doesn't contain tabular data theoritically)
                if ymin > previous_ymax:
                    gap_bbox = (0, previous_ymax, img_width, ymin)
                    img_crop = img.crop(gap_bbox)
                    #gap_text = self.extract_plain_text_from_image(img_crop)
                    gap_text = pytesseract.image_to_string(img_crop)
                    full_text.append(gap_text)
                # Add the text from the current table
                full_text.append(table_crop["table_text"])
                # Update previous ymax to the current table's ymax
                previous_ymax = ymax
            
            # Handle the gap after the last table
            if previous_ymax < img_height:
                gap_bbox = (0, previous_ymax, img_width, img_height)
                img_crop = img.crop(gap_bbox)
                #gap_text = self.extract_plain_text_from_image(img_crop)
                gap_text = pytesseract.image_to_string(img_crop)
                full_text.append(gap_text)

            #full_text = "\n".join(full_text)
            full_text = SEPARATOR.join(full_text)
            # Remove unnecessary line breaks
            full_text = '\n'.join(line for line in full_text.splitlines() if line.strip())
            return full_text
        
        except Exception as e:
            print(f"An error occurred while converting the image to text: {e}")
            return None

    def extract_plain_text_from_image(self, img):
        text = pytesseract.image_to_string(img)
        return text
        

    def preprocess_image(self, img):
        """
        Preprocess the image using OpenCV functions.

        :param img: The image in PIL format.
        :return: The preprocessed image in PIL format.
        """
        # Convert PIL Image to a NumPy array
        img_np = np.array(img)
        img_np = self._preprocess_image(img_np)
        # Convert the NumPy array back to a PIL Image
        preprocessed_img = Image.fromarray(img_np)
        return preprocessed_img

    def _preprocess_image(self, img, grey=1, threshold=180, adapt=0, blur=0, thresh=0, sharp=0, edge_cascade=0, edge1=50, edge2=200):
        """
        Apply preprocessing steps to the image using OpenCV functions.

        :param img: The image in NumPy array format.
        :param grey: Flag to apply grayscale conversion.
        :param threshold: Threshold value for binary thresholding.
        :param adapt: Flag to apply adaptive thresholding.
        :param blur: Flag to apply median blur.
        :param thresh: Flag to apply binary thresholding.
        :param sharp: Flag to apply sharpening.
        :param edge_cascade: Flag to apply edge detection.
        :param edge1: First threshold for edge detection.
        :param edge2: Second threshold for edge detection.
        :return: The preprocessed image in NumPy array format.
        """
        newImg = img
        if grey:
            newImg = self._grey(newImg)
        if edge_cascade:
            newImg = self._edge_cascade(newImg, edge1, edge2)
        if blur:
            newImg = self._blur(newImg)
        if thresh:
            newImg = self._threshold(newImg, threshold)
        if adapt:
            newImg = self._adaptive_threshold(newImg)
        if sharp:
            newImg = self._sharpen(newImg)
        return newImg

    def _grey(self, img):
        return cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)

    def _edge_cascade(self, img, t1=50, t2=200):
        return cv.Canny(img, t1, t2)

    def _blur(self, img):
        return cv.medianBlur(img, 3)

    def _threshold(self, img, threshold=180):
        _, newImg = cv.threshold(img, threshold, 250, cv.THRESH_BINARY)
        return newImg

    def _adaptive_threshold(self, img):
        return cv.adaptiveThreshold(img, 200, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)

    def _sharpen(self, img, kernel_sharp=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])):
        return cv.filter2D(img, -1, kernel_sharp)

 
if __name__ == "__main__":
    image_converter = ImageConverter("./data/png.png")
    text = image_converter.convert_to_text()
    print(text)
    with open("./data/text result.txt", 'w') as file:
        file.writelines(text) 