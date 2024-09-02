import pytest
from textify_docs.image_converter import ImageConverter

@pytest.fixture
def image_converter():
    """Fixture to provide a fresh ImageConverter instance for each test."""
    return ImageConverter()

def test_image_conversion(image_converter):
    """Test image conversion to text."""
    image_path = "./data/test/test_image_with_tables.png"
    text = image_converter.convert_to_text(image_path)
    assert isinstance(text, str), "The output should be a string"
    assert len(text) > 0, "The output text should not be empty"
