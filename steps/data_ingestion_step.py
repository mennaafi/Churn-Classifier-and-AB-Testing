from zenml import step
import pandas as pd
from src.exploration.DataIngestion import DataIngestorFactory
import os

@step
def data_ingestion_step(data_path: str) -> pd.DataFrame:

    _, file_extension = os.path.splitext(data_path)
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(data_path)
    return df
