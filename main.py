from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.textSummarizer.logging import logger

STAGE_NAME = "Data Ingestion Stage"
STAGE_NAME_2 = "Data Transformation Stage"
def main():
    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<")
        data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_training_pipeline.initiate_data_ingestion()
        logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(f"An error occurred in stage {STAGE_NAME}: {e}")
        raise e
    
    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME_2} started <<<<<<<")
        data_transformation_training_pipeline = DataTransformationTrainingPipeline()
        data_transformation_training_pipeline.initiate_data_transformation()
        logger.info(f">>>>>>> Stage {STAGE_NAME_2} completed <<<<<<<")

    except Exception as e:
        logger.exception(f"An error occurred in stage {STAGE_NAME_2}: {e}")
        raise e

    
if __name__ == "__main__":
    main()


