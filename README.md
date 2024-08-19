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
```

you also need to install some system dependencies:

### On Linux

```bash
sudo apt-get install -y poppler-utils
sudo apt-get install -y tesseract-ocr
```

### On Windows

1. **Poppler**: 
   - Download the latest Poppler binaries from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/).
   - Extract the contents and add the `bin` folder to your system's PATH environment variable.

2. **Tesseract**:
   - Download the latest Tesseract installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
   - Run the installer and follow the prompts.
   - Make sure to add Tesseract to your system's PATH during installation.

### On macOS

```bash
brew install poppler
brew install tesseract
```
