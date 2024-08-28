from abc import ABC, abstractmethod
import os
import json
import torch
import openai
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils import load_config
torch.cuda.empty_cache()

class Embedding(ABC):
    @abstractmethod
    def get_embeddings(self, texts):
        pass


class LLAMA(Embedding):
    def __init__(self, config:dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['llama_model_path'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device='cpu'
        self.model = AutoModel.from_pretrained(self.config['llama_model_path'])

    def get_embeddings(self, texts, step=8):
        length = len(texts)
        embedding_list = []

        for st in tqdm(range(0, length, step)):
            end = min(st + step, length)
            inputs = self.tokenizer(texts[st:end], return_tensors="pt", max_length=2048, padding='max_length',
                                    add_special_tokens=True, truncation=True)

            # Move tensors to the same device as model
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

            self.model.to(self.device)
            # Perform inference and extract embeddings
            with torch.no_grad():  # Disable gradient computation for inference
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, -1, :].detach()

            # Move embeddings back to CPU for further processing if needed
            batch_embedding = embeddings.cpu()
            embedding_list.append(batch_embedding)
        total_embedding = torch.cat(embedding_list, dim=0)
        return total_embedding.numpy()


class OPENAI(Embedding):
    def __init__(self, config):
        self.config = config
        with open(self.config['openai_api_key_path'], 'r') as file:
            self.api_key = file.read().strip()
        openai.api_key = self.api_key

    def get_embedding(self, text):
        # Get a single text embedding
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data']['embedding']

    def get_embeddings(self, texts):
        # Get embeddings for a list of texts
        embeddings = []
        for text in texts:
            # For each text, get the embedding and append to the list
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings




# Example Usage
if __name__ == "__main__":
    config = load_config()

    text = "Example text for generating embeddings."

    # data_sets = ['dataset/thunderbird_featured.csv', 'dataset/KDE_featured.csv', 'dataset/firefox_featured.csv']
    data_sets = ['dataset/KDE_featured.csv', 'dataset/firefox_featured.csv']
        # LLAMA Embedding
    llama_embedder = LLAMA(config)

    for data_path in data_sets:
        df = pd.read_csv(data_path)
        df.comment =  df.comment.apply(lambda x: str(x))
        llama_embedding = llama_embedder.get_embeddings(df.comment.to_list())
        with open(f'{data_path}_embedding.json', 'w') as f:
            json.dump(llama_embedding.tolist(), f)

    print("LLAMA Embedding:", llama_embedding)

    # OPENAI Embedding
    # openai_embedder = OPENAI(config)
    # openai_embedding = openai_embedder.get_embedding(text)
    # print("OpenAI Embedding:", openai_embedding)