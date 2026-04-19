from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.constants import *
from src.textSummarizer.entity import DataIngestionConfig, DataTransformationConfig

# Creamos el configuration mangarer, que se encargará de cargar el archivo de configuración y devolver la configuración de cada componente.
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Creamos el directorio de Artifacts, que es donde se guardarán todos los artefactos del proyecto.
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Obtenemos la configuración de ingesta de datos del archivo de config.yaml con las rutas definidas
        config = self.config.data_ingestion

        # Creamos el directorio de ingesta de datos (A la función siempre le pasamos una lista)
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(root_dir=config.root_dir,
                                                    source_URL_train=config.source_URL_train,
                                                    source_URL_test=config.source_URL_test,
                                                    source_URL_val=config.source_URL_val)
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        data_transformation_config = DataTransformationConfig(
            root_dir = Path(config.root_dir),
            data_path = Path(config.data_path),
            tokenizer_name = config.tokenizer_name)
        
        return data_transformation_config