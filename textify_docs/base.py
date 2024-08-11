from abc import ABC, abstractmethod

class BaseConverter(ABC):
    """
    Abstract base class for document converters.
    """

    def __init__(self, file_path):
        """
        Initialize the converter with the path to the document file.

        :param file_path: Path to the document file to be converted.
        """
        self.file_path = file_path

    @abstractmethod
    def convert_to_text(self):
        """
        Abstract method to convert the document to plain text.
        Must be implemented by all subclasses.

        :return: A string containing the plain text extracted from the document.
        """
        pass
