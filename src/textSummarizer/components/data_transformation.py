import os
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import DataTransformationConfig
from transformers import AutoTokenizer
from datasets import load_dataset

# DATA TRANSFORMATION COMPONENT

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        logger.info("Convirtiendo los ejemplos a features utilizando el tokenizer.")
        model_inputs = self.tokenizer(
            example_batch["dialogue"],
            max_length=1024,
            truncation=True,
            padding=True
        )

        labels = self.tokenizer(
            text_target=example_batch["summary"],
            max_length=128,
            truncation=True,
            padding=True
        )

        model_inputs["labels"] = labels["input_ids"]
        logger.info("Ejemplos convertidos a features exitosamente.")
        return model_inputs
    
    def convert(self):
        logger.info("Iniciando la etapa de transformación de datos.")
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.config.data_path / "train.csv"),
                "validation": str(self.config.data_path / "val.csv"),
                "test": str(self.config.data_path / "test.csv")
            }
        )
        dataset_samsun_pt = dataset.map(self.convert_examples_to_features, batched = True)
        dataset_samsun_pt.save_to_disk(str(self.config.root_dir / "samsum_dataset"))
        logger.info("Etapa de transformación de datos completada exitosamente.")