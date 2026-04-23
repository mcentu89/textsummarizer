from src.textSummarizer.components.model_evaluation import ModelEvaluation
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.logging import logger


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        logger.info("Iniciando la etapa de evaluación del modelo.")
        config = ConfigurationManager()
        model_evaluation_config = config.get_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()
        logger.info("Etapa de evaluación del modelo completada exitosamente.")
