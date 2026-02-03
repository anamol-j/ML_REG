import pandas as pd

class DataIngestion:
    def __init__(self,file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)
        return df