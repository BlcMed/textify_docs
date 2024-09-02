import pytest
from textify_docs.image_converter import ImageConverter
from PIL import Image
import numpy as np

@pytest.fixture
def image_converter():
    """Fixture to provide a fresh ImageConverter instance for each test."""
    return ImageConverter()

@pytest.fixture
def test_image():
    """Fixture to load a test image."""
    # Create a simple test image (e.g., 100x100 pixels with a solid color or pattern)
    image = Image.new('RGB', (100, 100), color='white')
    return image

def test_image_conversion(image_converter):
    """Test image conversion to text."""
    image_path = "./data/test/test_image_with_tables.png"
    text = image_converter.convert_to_text(image_path)
    assert isinstance(text, str), "The output should be a string"
    assert len(text) > 0, "The output text should not be empty"

def test_preprocess_image_grey(image_converter, test_image):
    """Test grayscale preprocessing."""
    grey_image = image_converter.preprocess_image(test_image, grey=True, blur=False, thresh=False, adapt=False, sharp=False, edge_cascade=False)
    assert grey_image.mode == 'L', "The image should be in grayscale mode"

def test_preprocess_image_blur(image_converter, test_image):
    """Test blurring preprocessing."""
    blurred_image = image_converter.preprocess_image(test_image, grey=False, blur=True, thresh=False, adapt=False, sharp=False, edge_cascade=False)
    blurred_image_np = np.array(blurred_image)
    assert blurred_image_np.shape == (100, 100, 3), "The image should remain the same size after blurring"

def test_preprocess_image_threshold(image_converter, test_image):
    """Test binary threshold preprocessing."""
    threshold_image = image_converter.preprocess_image(test_image, grey=True, blur=False, thresh=True, adapt=False, sharp=False, edge_cascade=False)
    threshold_image_np = np.array(threshold_image)
    unique_values = np.unique(threshold_image_np)
    assert len(unique_values) <= 2, "Thresholded image should have at most two unique values (cv.THRESH_BINARY type)"

def test_preprocess_image_adaptive_threshold(image_converter, test_image):
    """Test adaptive threshold preprocessing."""
    adapt_image = image_converter.preprocess_image(test_image, grey=True, blur=False, thresh=False, adapt=True, sharp=False, edge_cascade=False)
    adapt_image_np = np.array(adapt_image)
    unique_values = np.unique(adapt_image_np)
    assert len(unique_values) <= 2, "Adaptive thresholding should  at most two unique values (binary)"

def test_preprocess_image_sharp(image_converter, test_image):
    """Test sharpening preprocessing."""
    sharp_image = image_converter.preprocess_image(test_image, grey=False, blur=False, thresh=False, adapt=False, sharp=True, edge_cascade=False)
    sharp_image_np = np.array(sharp_image)
    assert sharp_image_np.shape == (100, 100, 3), "The image should remain the same size after sharpening"

def test_preprocess_image_edge_detection(image_converter, test_image):
    """Test edge detection preprocessing."""
    edge_image = image_converter.preprocess_image(test_image, grey=True, blur=False, thresh=False, adapt=False, sharp=False, edge_cascade=True)
    edge_image_np = np.array(edge_image)
    assert edge_image_np.ndim == 2, "Edge detection should return a single channel image (grayscale)"
