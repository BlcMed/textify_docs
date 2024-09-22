SUPPORTED_FILE_FORMATS = [
    "docx",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".xls",
    ".xlsx",
]


# TESSERACT CONFIG ##
LANGUAGE = "fra"
TESSERACT_CONFIG_PLAIN_TEXT = "--psm 3 --oem 3"
TESSERACT_CONFIG_CELL = "--psm 6 --oem 3"


# TABLE EXTRACTION CONSTANTS ##
# The threshold used to determine if a detected object is a table or not.
DETECTION_CLASS_THRESHOLDS = {"table": 0.5, "table rotated": 0.5, "no object": 10}
# the padding to add to each cell bounding box in a table.
PADDING = 10


# The separator used to separate the different parts of the image (e.g., text, tables, etc.).
SEPARATOR = "\n" + "-" * 20 + "\n"


# PATHS ##
# Path to the directory where the models are stored.
MODELS_PATH = "cached_models/"
TABLE_DETECTION_MODEL_NAME = "microsoft/table-transformer-detection"
TABLE_STRUCTURE_RECOGNITION_MODEL_NAME = (
    "microsoft/table-structure-recognition-v1.1-all"
)

# PREPROCESSING IMAGE CONSTANTS ##

# DEFAULT FLAGS
# grey: Flag to apply grayscale conversion.
GREY = 1
# adapt: Flag to apply adaptive thresholding.
ADAPT = 0
# blur: Flag to apply median blur.
BLUR = 0
# thresh: Flag to apply binary thresholding.
THRESH = 0
# sharp: Flag to apply sharpening.
SHARP = 0

# DEFAULT CONSTANTS
KERNEL_SHARP = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
# edge_cascade: Flag to apply edge detection.
EDGE_CASCADE = 0
# threshold: Threshold value for binary thresholding.
THRESHOLD = 180
# edge1: First threshold for edge detection.
EDGE1 = 50
# edge2: Second threshold for edge detection.
EDGE2 = 200
# blur
BLUR_KERNEL_SIZE = 3
# adaptive threshold
MAXVALUE = 200
BLOCKSIZE = 11
C = 3

MAX_SIZE = 800
