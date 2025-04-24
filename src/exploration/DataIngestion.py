from abc import ABC, abstractmethod
import pandas as pd

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass


class CSVDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)


class ExcelDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        return pd.read_excel(file_path)


class JSONDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        return pd.read_json(file_path)


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        if file_extension == ".csv":
            return CSVDataIngestor()
        elif file_extension == ".json":
            return JSONDataIngestor()
        elif file_extension in [".xls", ".xlsx"]:
            return ExcelDataIngestor()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    