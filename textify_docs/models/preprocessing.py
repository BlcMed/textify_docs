from torchvision import transforms

from ..config import MAX_SIZE


class _MaxResize:
    """Resize an image to a maximum size while maintaining aspect ratio."""

    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )
        return resized_image


# For the moment we apply the same transformation to both detection and structure recognition
detection_transform = transforms.Compose(
    [
        _MaxResize(max_size=MAX_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

structure_recognition_transform = transforms.Compose(
    [
        _MaxResize(max_size=MAX_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
