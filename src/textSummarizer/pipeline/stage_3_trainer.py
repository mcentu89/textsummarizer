from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_trainer import ModelTrainer
from src.textSummarizer.logging import logger

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        logger.info("Iniciando la etapa de entrenamiento del modelo.")
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
        logger.info("Etapa de entrenamiento del modelo completada exitosamente.")