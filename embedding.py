from abc import ABC, abstractmethod
import os
import torch
import openai
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

    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", max_length=2048, padding='max_length',
                                add_special_tokens=True, truncation=True)

        # Move tensors to the same device as model
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        self.model.to(self.device)
        # Perform inference and extract embeddings
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, -1, :].detach()

        # Move embeddings back to CPU for further processing if needed
        return embeddings.cpu().numpy()



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

    # LLAMA Embedding
    llama_embedder = LLAMA(config)
    llama_embedding = llama_embedder.get_embedding(text)
    print("LLAMA Embedding:", llama_embedding)

    # OPENAI Embedding
    openai_embedder = OPENAI(config)
    openai_embedding = openai_embedder.get_embedding(text)
    print("OpenAI Embedding:", openai_embedding)