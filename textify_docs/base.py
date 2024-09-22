"""
This module defines the abstract base class for document converters.
Subclasses must implement the method to convert various document formats into plain text.
"""

from abc import ABC, abstractmethod


class BaseConverter(ABC):
    """
    Abstract base class for document converters.
    """

    @abstractmethod
    def convert_to_text(self, file_path):
        """
        Abstract method to convert the document to plain text.
        Must be implemented by all subclasses.

        :param file_path: Path to the document file to be converted.

        :return: A string containing the plain text extracted from the document.
        """
