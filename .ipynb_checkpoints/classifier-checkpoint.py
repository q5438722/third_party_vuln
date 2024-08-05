import os, sys
import json, jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dataset(data_sets):
    dfs = [for path in data_sets]


# Load the pre-trained LLaMA model and tokenizer
model_name = 'meta-llama/Llama-2-7b-chat-hf'
model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Example data
data_sets = ['dataset/gentoo.csv', 'dataset/KDE.csv', 'dataset/mozilla.csv',
             'dataset/suse.csv', 'dataset/VSCode.csv']


texts = ["This is a positive sentence.", "This is a negative sentence."]
labels = [1, 0]  # 1 for positive, 0 for negative

# Create the dataset and dataloader
dataset = CustomDataset(texts, labels, tokenizer, max_length=1024)
# dataloader = DataLoader(dataset, batch_size=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('./fine-tuned-llama')
tokenizer.save_pretrained('./fine-tuned-llama')
