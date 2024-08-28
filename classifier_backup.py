import os, sys
import numpy as np
import json, jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModel, Trainer, DataCollatorWithPadding, TrainingArguments, EvalPrediction
from transformers import LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments

from transformers import BertForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, Trainer, DataCollatorWithPadding, TrainingArguments, \
    EvalPrediction, HfArgumentParser

from sklearn.metrics import roc_auc_score, precision_recall_curve, matthews_corrcoef, accuracy_score
import numpy as np

def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))

def np_divide(a, b):
    # return 0 with divide by 0 while performing element wise divide for np.arrays
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0))

def f1_score(p, r):
    if p + r < 1e-5:
        return 0.0
    return 2 * p * r / (p + r)

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
            add_special_tokens=True,
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
    tot_df = pd.concat(dfs, axis=0)
    train, test = train_test_split(tot_df, test_size=0.2, random_state=42)
    return train, test

from transformers import AutoModel, GPTBigCodePreTrainedModel, GPTBigCodeModel
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType


class ModelClassifier(nn.Module):
    def __init__(self, name):
        super(ModelClassifier, self).__init__()
        self.num_labels = 1

        self.starcoder = AutoModel.from_pretrained(name)
        self.lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05
        )
        
        # Integrate LoRA into the model
        # self.starcoder = get_peft_model(self.starcoder, self.lora_config)

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(4096, 1)

        # self.init_weights()
        self.alpha = [1, 10]
        self.gamma = 2
        # self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # token_type_ids=None,
        position_ids=None,
        # head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else None
        outputs = self.starcoder(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print('wow')
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = BCEWithLogitsLoss()
                # print(logits.view(-1), labels.view(-1).float())
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
                alpha = self.alpha[0] * (1 - labels.view(-1).float()) + self.alpha[1] * labels.view(-1).float()
                loss = (alpha * loss).mean()
                # print('type', type(loss))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # print('loss: ', loss)
            # outputs = (loss,) + outputs
        # print(loss, logits)
        return SequenceClassifierOutput(loss=loss, logits=logits)  # (loss), logits, (hidden_states), (attentions)


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load the pre-trained LLaMA model and tokenizer
model_name = 'codellama/CodeLlama-7b-Instruct-hf'
model = ModelClassifier(model_name)
# print(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.pad_token_id
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
train_dataset = CustomDataset(train['comment'].to_list(), train['label'].to_list(), tokenizer, max_length=64)
eval_dataset = CustomDataset(test['comment'].to_list(), test['label'].to_list(), tokenizer, max_length=64)
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

# Fine-tune the model
trainer.train()
