import os
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.entity import ModelTrainerConfig
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import torch


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        

        
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        #Training
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        dataset = load_from_disk(self.config.data_path)
        trainer_args = TrainingArguments(
            output_dir = self.config.root_dir, num_train_epochs = self.config.num_train_epochs, warmup_steps = self.config.warmup_steps,
            per_device_train_batch_size = self.config.per_device_train_batch_size, per_device_eval_batch_size = self.config.per_device_eval_batch_size,
            weight_decay = self.config.weight_decay, logging_steps = self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy, eval_steps = self.config.eval_steps, save_steps = int(float(self.config.save_steps)),
            gradient_accumulation_steps = self.config.gradient_accumulation_steps
        )
        
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dataset["train"].select(range(300)),
            eval_dataset=dataset["validation"],
            data_collator=seq2seq_data_collator
        )
        trainer.train()
        #Save the model
        trainer.save_model(os.path.join(self.config.root_dir, "pegasus_samsum_model"))
        #Save the tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))