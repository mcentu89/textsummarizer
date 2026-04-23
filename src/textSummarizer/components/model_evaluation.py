from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.entity import ModelEvaluationConfig
import torch
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

import evaluate


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Divide el dataset en fragmentos más pequeños para procesarlos en batches."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer,
                                    batch_size=16, device="cuda",
                                    column_text="dialogue",
                                    column_summary="summary"):

        # Dividir en batches
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        model.to(device)
        model.eval()

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)
        ):

            # Tokenizar input
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # ✅ CORRECCIÓN AQUÍ (usar inputs reales)
            summaries = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                length_penalty=0.8,
                num_beams=8,
                max_length=128
            )

            # Decodificar resúmenes generados
            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for s in summaries
            ]

            # Limpiar espacios extra
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            # Añadir resultados a la métrica
            metric.add_batch(predictions=decoded_summaries, references=target_batch)

            score = metric.compute()
        
            return score
    
    def evaluate(self):
        # Cargar dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Cargar modelo y tokenizer
        model_path = Path(self.config.model_path)
        tokenizer_path = Path(self.config.tokenizer_path)

        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Cargar métrica ROUGE
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        
        rouge_metric = evaluate.load("rouge")

        # Calcular métrica en el dataset de test
        scores = self.calculate_metric_on_test_ds(
            dataset=dataset_samsum_pt['test'][0:10],
            metric=rouge_metric,
            model=model,
            tokenizer=tokenizer,
            batch_size=2,
            device=device,
            column_text="dialogue",
            column_summary="summary"
        )

        rouge_dict = {rn: scores[rn] for rn in rouge_names}

        # Guardar resultados en un archivo CSV
        df_scores = pd.DataFrame(rouge_dict, index=['pegasus'])
        df_scores.to_csv(self.config.metric_file_name, index=False)