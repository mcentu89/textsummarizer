from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.constants import *
from src.textSummarizer.entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig

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
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])
    
        model_trainer_config = ModelTrainerConfig(
            root_dir = Path(config.root_dir),
            data_path = Path(config.data_path),
            model_ckpt = config.model_ckpt,
            num_train_epochs = params.num_train_epochs,
            warmup_steps = params.warmup_steps,
            per_device_train_batch_size = params.per_device_train_batch_size,
            per_device_eval_batch_size = params.per_device_eval_batch_size,
            weight_decay = params.weight_decay,
            logging_steps = params.logging_steps,
            evaluation_strategy = params.evaluation_strategy,
            eval_steps = params.eval_steps,
            save_steps = params.save_steps,
            gradient_accumulation_steps = params.gradient_accumulation_steps
        )
        
        return model_trainer_config
    
    def get_evaluation_config(self) -> ModelEvaluationConfig:
        evaluation_config = self.config.model_evaluation

        create_directories([evaluation_config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(evaluation_config.root_dir),
            data_path=Path(evaluation_config.data_path),
            model_path=Path(evaluation_config.model_path),
            tokenizer_path=Path(evaluation_config.tokenizer_path),
            metric_file_name=Path(evaluation_config.metric_file_name)
        )

        return model_evaluation_config
    
    
