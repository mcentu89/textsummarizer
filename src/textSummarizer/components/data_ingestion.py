# Importamos las librerias para ingestar los datos y descargar los datos.
import os
import requests
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self, url: str, file_name: str) -> None:
        file_path = os.path.join(self.config.root_dir, file_name)
        logger.info(f"Descargando el archivo desde la URL: {url} a la ruta: {file_path}")
        response = requests.get(url)
        response.raise_for_status()  # Verificar si la solicitud fue exitosa

        with open(file_path, "wb") as file:
            file.write(response.content)
        logger.info(f"Archivo descargado exitosamente en: {file_path}")
