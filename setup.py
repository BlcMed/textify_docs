from setuptools import setup, find_packages

# Function to read the requirements from the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="textify_docs",
    version="1.0.1", 
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        "dev": [
            "pytest",
        ]
    },
    description="A package to convert various document types to plain text.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Mohamed Boulaich",
    author_email="boulaich.mohamed970@gmail.com",
    url="https://github.com/BlcMed",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    python_requires='>=3.6',
)
