import cv2
from PIL import Image
import pytesseract
import numpy as np
from .base import BaseConverter
from .tables.table_extracter import extract_tables_from_image
from .config import LANGUAGE, SEPARATOR, TESSERACT_CONFIG_PLAIN_TEXT, GREY, THRESHOLD,ADAPT, BLUR, THRESH, SHARP, EDGE_CASCADE,EDGE1,EDGE2, KERNEL_SHARP, MAXVALUE, BLOCKSIZE, C, BLUR_KERNEL_SIZE


class ImageConverter(BaseConverter):

    def convert_to_text(self, file_path):
        """
        Convert the image file to plain text using OCR by processing tables and gaps in between them chronologically.
        
        :param img: The preprocessed image in PIL format.

        :return:
            str: A string containing the combined plain text from tables and gaps between them, ordered as they appear in the image.
                If an error occurs during processing, returns None.
        """
        try:
            with Image.open(file_path) as img:
                img = self.preprocess_image(img)
                img_width, img_height = img.size
                tables_crops = extract_tables_from_image(img, language=LANGUAGE)
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
                        gap_text = pytesseract.image_to_string(img_crop,lang=LANGUAGE, config=TESSERACT_CONFIG_PLAIN_TEXT)
                        full_text.append(gap_text)
                    full_text.append(table_crop["table_text"])
                    # Update previous ymax to the current table's ymax
                    previous_ymax = ymax
                
                # Handle the gap after the last table
                if previous_ymax < img_height:
                    gap_bbox = (0, previous_ymax, img_width, img_height)
                    img_crop = img.crop(gap_bbox)
                    gap_text = pytesseract.image_to_string(img_crop,lang=LANGUAGE, config=TESSERACT_CONFIG_PLAIN_TEXT)
                    full_text.append(gap_text)

                #full_text = "\n".join(full_text)
                full_text = SEPARATOR.join(full_text)
                # Remove unnecessary line breaks
                full_text = '\n'.join(line for line in full_text.splitlines() if line.strip())
                #full_text = self.extract_text_from_image(img=img)
                return full_text
        except Exception as e:
            print(f"An error occurred while converting the image to text: {e}")
            return None

    def preprocess_image(self, img, grey=GREY, adapt=ADAPT, blur=BLUR, thresh=THRESH, sharp=SHARP, edge_cascade=EDGE_CASCADE, threshold= THRESHOLD, edge1=EDGE1, edge2=EDGE2,blur_kernel_size=BLUR_KERNEL_SIZE,kernel_sharp=KERNEL_SHARP, maxValue=MAXVALUE, blockSize=BLOCKSIZE, C=C):
        """
        Preprocess the image (in PIL fomrat) using Opencv2.functions.

        :param img: The image in PIL format.
        :param grey: Flag to apply grayscale conversion.
        :param adapt: Flag to apply adaptive thresholding.
        :param blur: Flag to apply median blur.
        :param thresh: Flag to apply binary thresholding.
        :param sharp: Flag to apply sharpening.
        :param edge_cascade: Flag to apply edge detection.
        :param threshold: Threshold value for binary thresholding.
        :param edge1: First threshold for edge detection.
        :param edge2: Second threshold for edge detection.

        :return: The preprocessed image in PIL format.
        """
        # Convert PIL Image to a NumPy array
        img_np = np.array(img)

        newImg = img_np
        if grey:
            newImg = cv2.cvtColor(newImg, code=cv2.COLOR_BGR2GRAY)
            # Adaptive threshhold algorithm requires the image to be grayscale before using adapt to 
            if adapt:
                newImg = cv2.adaptiveThreshold(newImg, maxValue=maxValue, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=blockSize, C=C)
        if edge_cascade:
            newImg = cv2.Canny(newImg, edge1, edge2)
        if blur:
            newImg = cv2.medianBlur(newImg, ksize=blur_kernel_size)
        if thresh:
            _, newImg = cv2.threshold(newImg, threshold, 255, cv2.THRESH_BINARY)
        if sharp:
            np_kernel_sharp=np.array(kernel_sharp)
            newImg = cv2.filter2D(newImg, -1, np_kernel_sharp)
        # Convert the NumPy array back to a PIL Image
        preprocessed_img = Image.fromarray(newImg)
        return preprocessed_img

 
if __name__ == "__main__":
    path="./data/png.png"
    image_converter = ImageConverter()
    text = image_converter.convert_to_text(path)
    print(text)
    with open("./data/text result.txt", 'w') as file:
        file.writelines(text) 