import os

import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    DataCollatorWithPadding,
)
from evaluate import load

from src.config_bert import ModelConfig


class BERTfinetuning:
    def __init__(self, dataset, training_args):
        self.model = ModelConfig()
        self.model_name = self.model.model_name
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenized_datasets = dataset.map(
            self.tokenize_function, batched=True, num_proc=int(os.cpu_count() / 2)
        )  # 16-Core multi Processor
        config = AutoConfig.from_pretrained(self.model_name, num_labels=2)
        # ------------- Optional: Adjust dropout rates in the config if needed -------------
        # config.hidden_dropout_prob = self.model.dropout
        # config.attention_probs_dropout_prob = self.model.dropout
        # config.dropout = self.model.dropout
        # config.attention_dropout = self.model.dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config
        )
        self.training_args = training_args
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.trainer = Trainer(
            model=self.model,  # Pre-trained BERT model
            args=training_args,  # Training arguments
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["dev"],
            data_collator=self.data_collator,  # Efficient batching
            compute_metrics=self.compute_metrics,  # Custom metric
        )
        self.metric = load("f1")

    # Tokenization
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["clean_text"],
            #  padding="max_length", # Data Collator: dynamical padding
            truncation=True,
            max_length=128,
        )

    # Define a custom compute_metrics function
    def compute_metrics(
        self, eval_pred
    ):  # https://github.com/huggingface/evaluate/blob/main/metrics/f1/f1.py
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        model = ModelConfig()
        return self.metric.compute(
            predictions=predictions, references=labels, average=model.f1_avg
        )

    def test(self):
        test_results = self.trainer.predict(self.tokenized_datasets["test"])
        return print(test_results.metrics)

    def train(self):
        self.trainer.train()
