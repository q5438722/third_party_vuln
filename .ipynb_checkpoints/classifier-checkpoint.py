import os, sys
import numpy as np
import json, jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, Trainer, DataCollatorWithPadding, TrainingArguments, EvalPrediction
from transformers import LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = [str(item) for item in texts]
        self.labels = [1 if str(item) == '1' else 0 for item in labels]
        # print(self.labels[:10])
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print('padding', self.tokenizer.pad_token)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # print('encoding', {
        #     'input_ids': encoding['input_ids'].flatten(start_dim=0, end_dim=-1),
        #     'attention_mask': encoding['attention_mask'].flatten(start_dim=0, end_dim=-1),
        #     'labels': torch.tensor(label, dtype=torch.long)
        # })
        return {
            'input_ids': encoding['input_ids'].view(-1),
            'attention_mask': encoding['attention_mask'].view(-1),
            'labels': torch.tensor(label).float()
        }

def load_dataset(data_sets):
    dfs = [pd.read_csv(path) for path in data_sets]
    tot_df = pd.concat(dfs, axis=0).dropna()
    train, test = train_test_split(tot_df, test_size=0.2, random_state=42)
    return train, test

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load the pre-trained LLaMA model and tokenizer
model_name = 'meta-llama/Llama-2-7b-chat-hf'
model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=1)
# print(model)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
# print(model.config.pad_token_id)
# tokenizer.pad_token = tokenizer.special_tokens_map['eos_token']
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Example data
data_sets = ['dataset/gentoo.csv', 'dataset/KDE.csv', 'dataset/mozilla.csv',
             'dataset/suse.csv', 'dataset/VSCode.csv']
train, test = load_dataset(data_sets)

# texts = ["This is a positive sentence.", "This is a negative sentence."]
# labels = [1, 0]  # 1 for positive, 0 for negative

# Create the dataset and dataloader
train_dataset = CustomDataset(train['comment'].to_list(), train['label'].to_list(), tokenizer, max_length=512)
eval_dataset = CustomDataset(test['comment'].to_list(), test['label'].to_list(), tokenizer, max_length=512)
# dataloader = DataLoader(dataset, batch_size=2)


def compute_metrics(eval_pred: EvalPrediction):
    scores = sigmoid(eval_pred.predictions.reshape(-1, ))
    labels = eval_pred.label_ids.reshape(-1, )
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # while computing f1 = (2 * precision * recall) / (precision + recall), some element in (precision+recall) will be 0
    f1 = np_divide(2 * precision * recall, precision + recall)
    f1_idx = np.argmax(f1)
    f1_best = f1[f1_idx]

    auc = roc_auc_score(y_true=labels, y_score=scores)
    
    return {"auc": auc,
            "f1_best": f1_best,
            "f1_idx": f1_idx,
            "threshold": thresholds[f1_idx]
           }


# Define training arguments
training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=5,
    eval_strategy="epoch",
    eval_delay=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    optim="adamw_torch",
    fp16=True,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)
