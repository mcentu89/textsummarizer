from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.data_ingestion import DataIngestion
from src.textSummarizer.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        logger.info("Iniciando la etapa de ingesta de datos para el entrenamiento.")
        data_ingestion_config = ConfigurationManager().get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)

        data_ingestion.download_file(url=data_ingestion_config.source_URL_train, file_name="train.csv")
        data_ingestion.download_file(url=data_ingestion_config.source_URL_test, file_name="test.csv")
        data_ingestion.download_file(url=data_ingestion_config.source_URL_val, file_name="val.csv")
        logger.info("Etapa de ingesta de datos para el entrenamiento completada exitosamente.")




