"""
transformer_model.py - Stage 2: BERT fine-tuned spam classifier.
Fixed for transformers >= 5.0 (PyTorch only, TF support was dropped)
"""

import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from src import config


class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=None):
        self.encodings = tokenizer(
            list(texts),
            max_length=max_length or config.TRANSFORMER["max_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"     : self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_bert_dataset(texts, labels, tokenizer, batch_size, shuffle=False, max_length=None):
    dataset = SpamDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_bert_model():
    cfg = config.TRANSFORMER
    print(f"Loading pretrained model: {cfg['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2
    )
    print("BERT model ready.")
    return model


def build_custom_bert_model():
    return build_bert_model()


def get_bert_callbacks(model_path=None):
    return []


def load_bert_tokenizer(model_name=None):
    name = model_name or config.TRANSFORMER["model_name"]
    return AutoTokenizer.from_pretrained(name)
