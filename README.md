# Document Converter

**Document Converter** is a Python package designed to convert various document types (such as PDF, DOCX, and images) into plain text. This package supports preprocessing of images to enhance OCR (Optical Character Recognition) results and can handle multiple file types without needing to specify the document type manually.

## Features

- Convert PDFs, DOCX, and images to plain text.
- Image preprocessing using OpenCV functions like grayscale conversion, blurring, thresholding, and edge detection.
- Automatic selection of the appropriate converter based on the document type.
- Easy to use with a single method to convert documents.

## Installation

You can install the package via pip:

```bash
pip install document-converter
