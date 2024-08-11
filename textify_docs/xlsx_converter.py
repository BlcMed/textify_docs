from .base import BaseConverter
import pandas as pd

class XlsxConverter(BaseConverter):
    def convert_to_text(self):
        try:
            """
            Extract text from an XLSX file.
            """
            df = pd.read_excel(self.file_path)
            return df.to_string()
        except Exception as e:
            print(f"An error occurred while converting the excel file: {e}")
            return None
