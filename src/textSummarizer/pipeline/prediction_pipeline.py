from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline, AutoModelForSeq2SeqLM
import torch
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_evaluation_config()
        self.tokenizer = AutoTokenizer.from_pretrained(str(Path(self.config.tokenizer_path)))
        self.gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
        # Intentar cargar el modelo seq2seq y usar .generate() (más control), con fallback a pipeline
        self.model = None
        self.pipe = None
        self._init_exception = None
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(Path(self.config.model_path)))
            self.model.to(device)
            self._model_device = device
        except Exception as e_model:
            # fallback: intentar crear un pipeline como antes
            last_exc = e_model
            task_candidates = ["summarization", "text-to-text", "text2text-generation", "text-generation", "any-to-any"]
            for task_name in task_candidates:
                try:
                    self.pipe = pipeline(task_name, model=str(Path(self.config.model_path)), tokenizer=self.tokenizer)
                    self._pipeline_task = task_name
                    last_exc = None
                    break
                except Exception as e_pipe:
                    last_exc = e_pipe

            if last_exc is not None:
                self._init_exception = last_exc
        
    def predict(self, text):
        if hasattr(self, '_init_exception') and self._init_exception is not None:
            raise self._init_exception

        print("Dialogue:")
        print(text)

        # Si se cargó un modelo seq2seq, usar .generate() para mayor control
        if self.model is not None:
            device = getattr(self, '_model_device', 'cpu')
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='longest').to(device)
            gen_kwargs = dict(
                max_length=self.gen_kwargs.get('max_length'),
                num_beams=self.gen_kwargs.get('num_beams'),
                length_penalty=self.gen_kwargs.get('length_penalty')
            )
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # fallback al pipeline si el modelo no se cargó
            result = self.pipe(text, **self.gen_kwargs)
            # Manejar distintos formatos retornados por distintas versiones/architecturas
            output = None
            try:
                if isinstance(result, list) and len(result) > 0:
                    first = result[0]
                    if isinstance(first, dict):
                        for key in ("summary_text", "generated_text", "text", "output_text"):
                            if key in first:
                                output = first[key]
                                break
                        if output is None:
                            vals = [v for v in first.values() if isinstance(v, str) and v.strip()]
                            output = vals[0] if vals else str(first)
                    elif isinstance(first, str):
                        output = first
                    else:
                        output = str(first)
                elif isinstance(result, dict):
                    for key in ("summary_text", "generated_text", "text", "output_text"):
                        if key in result:
                            output = result[key]
                            break
                    if output is None:
                        output = str(result)
                else:
                    output = str(result)
            except Exception:
                output = str(result)

        print("\nModel Summary:")
        print(output)

        return output
    
