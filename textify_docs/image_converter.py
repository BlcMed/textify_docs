from .base import BaseConverter
import cv2 as cv
from PIL import Image
import pytesseract
import numpy as np

class ImageConverter(BaseConverter):

    def convert_to_text(self):
        """
        Convert the image file to plain text using OCR after preprocessing.

        :return: A string containing the plain text extracted from the image.
        """
        try:
            # Open the image file
            with Image.open(self.file_path) as img:
                text = self.convert_to_text_from_image(img)
            return text
        
        except Exception as e:
            print(f"An error occurred while converting the image file: {e}")
            return None

    def convert_to_text_from_image(self, img):
        """
        Convert the given image (in PIL format) to plain text using OCR.
        
        :param img: The preprocessed image in PIL format.
        :return: A string containing the plain text extracted from the image.
        """
        try:
            preprocessed_img = self.preprocess_image(img)
            # OCR with tesseract
            text = pytesseract.image_to_string(preprocessed_img)
            return text
        
        except Exception as e:
            print(f"An error occurred while converting the image to text: {e}")
            return None

    def preprocess_image(self, img):
        """
        Preprocess the image using OpenCV functions.

        :param img: The image in PIL format.
        :return: The preprocessed image in PIL format.
        """
        # Convert PIL Image to a NumPy array
        img_np = np.array(img)

        # Apply preprocessing steps
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