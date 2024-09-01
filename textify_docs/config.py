SEPARATOR = "\n" + "-" * 20 +"\n"

DETECTION_CLASS_THRESHOLDS = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}
PADDING = 10 

SUPPORTED_FILE_FORMATS=['docx', '.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.xls', '.xlsx']

# PREPROCESSING IMAGE CONSTANTS
#grey: Flag to apply grayscale conversion.
#threshold: Threshold value for binary thresholding.
#adapt: Flag to apply adaptive thresholding.
#blur: Flag to apply median blur.
#thresh: Flag to apply binary thresholding.
#sharp: Flag to apply sharpening.
#edge_cascade: Flag to apply edge detection.
#edge1: First threshold for edge detection.
#edge2: Second threshold for edge detection.
GREY=1
THRESHOLD=180
ADAPT=0
BLUR=0
THRESH=0
SHARP=0
EDGE_CASCADE=0
EDGE1=50
EDGE2=200

MAX_SIZE= 800

# TESSERACT CONFIG
TESSERACT_CONFIG_PLAIN_TEXT="--psm 3 --oem 3"
TESSERACT_CONFIG_CELL="--psm 6 --oem 3"