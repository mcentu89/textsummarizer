from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_3_trainer import ModelTrainerPipeline
from src.textSummarizer.pipeline.stage_4_model_evaluation import ModelEvaluationPipeline
from src.textSummarizer.logging import logger




def main():
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<")
        data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_training_pipeline.initiate_data_ingestion()
        logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in stage {STAGE_NAME}: {e}")
        raise e
    
    STAGE_NAME_2 = "Data Transformation Stage"
    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME_2} started <<<<<<<")
        data_transformation_training_pipeline = DataTransformationTrainingPipeline()
        data_transformation_training_pipeline.initiate_data_transformation()
        logger.info(f">>>>>>> Stage {STAGE_NAME_2} completed <<<<<<<")

    except Exception as e:
        logger.exception(f"An error occurred in stage {STAGE_NAME_2}: {e}")
        raise e
    
    STAGE_NAME_3 = "Model Training Stage"
    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME_3} started <<<<<<<")
        model_trainer_pipeline = ModelTrainerPipeline()
        model_trainer_pipeline.initiate_model_trainer()
        logger.info(f">>>>>>> Stage {STAGE_NAME_3} completed <<<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in stage {STAGE_NAME_3}: {e}")
        raise e

    STAGE_NAME_4 = "Model Evaluation Stage"
    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME_4} started <<<<<<<")
        model_evaluation_pipeline = ModelEvaluationPipeline()
        model_evaluation_pipeline.initiate_model_evaluation()
        logger.info(f">>>>>>> Stage {STAGE_NAME_4} completed <<<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in stage {STAGE_NAME_4}: {e}")
        raise e


    
if __name__ == "__main__":
    main()


