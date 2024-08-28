
import pandas as pd
import openai
import qdrant_client
from qdrant_client.http import models
from embedding import LLAMA,OPENAI
from utils import load_config
import numpy as np
class RAG:

    def __init__(self,embedding_model='LLAMA',embedding_size=4096,qdrant_path='data/qdrant_db', embeddings_dir='data/embeddings',batch_size=16):
        self.embeddings_dir = embeddings_dir
        self.qdrant_path = qdrant_path
        self.collection_name = 'comment_embeddings'
        self.embeddings_size=embedding_size
        self.batch_size=batch_size
        config = load_config()
        if embedding_model=='LLAMA':
            self.embedding_model=LLAMA(config)
        elif embedding_model=='OPENAI':
            self.embedding_model=OPENAI(config)

        self.client = qdrant_client.QdrantClient(path=self.qdrant_path)


    def initialize(self, csv_file):


        # Initialize Qdrant client


        # Create collection in Qdrant
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.embeddings_size, distance=models.Distance.COSINE),
            # Adjust vector size based on model used
        )

        # Load the CSV file containing XML paths

        dataset = pd.read_csv(csv_file)
        dataset_name=csv_file.split('.')[0]


        comments = dataset['comment'].tolist()
        bug_ids = [f"{dataset_name}-{i}" for i in dataset.bug_id.tolist()]

        vectors, payloads = [], []
        for i in range(0, len(comments), self.batch_size):
            batch_comments = comments[i:i+self.batch_size]
            batch_ids = bug_ids[i:i+self.batch_size]
            embeddings = self.embedding_model.get_embeddings(batch_comments)
            if embeddings is not None:
                vectors.extend(embeddings)
                payloads.extend([{"bug_id": bug_id} for bug_id in batch_ids])

        print(f"Uploading {len(vectors)} embeddings to Qdrant.")

        # Upload all embeddings in a batch
        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            payload=payloads
        )
        print("All embeddings have been processed and saved to Qdrant.")

    def initialize_with_embedding(self, csv_file, embedding_list):


        # Initialize Qdrant client


        # Create collection in Qdrant
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.embeddings_size, distance=models.Distance.COSINE),
            # Adjust vector size based on model used
        )

        # Load the CSV file containing XML paths

        dataset = pd.read_csv(csv_file)
        dataset_name=csv_file.split('.')[0]


        comments = dataset['comment'].tolist()
        payloads = [{"bug_id": f"{dataset_name}-{i}"}  for i in dataset.bug_id.tolist()]
        print(f"Uploading {len(vectors)} embeddings to Qdrant.")

        # Upload all embeddings in a batch
        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            payload=payloads
        )
        print("All embeddings have been processed and saved to Qdrant.")

    def query(self,comment, bug_id):
        similar_ids=self.find_similar_bugids(comment,bug_id)
        comments=[]
        for similar_id in similar_ids:
            comments.append(self.bugid_to_comment(similar_id['bug_id']))
        return comments

    def bugid_to_comment(self,bug_id):
        dataset,query_id=bug_id.split('-')
        df=pd.read_csv(dataset+'.csv')
        df.set_index('bug_id', inplace=True)
        comment = df.at[int(query_id), 'comment']

        if isinstance(comment, str):
            return comment

    def find_similar_bugids(self, comment, bug_id,k=3):
        """
        1. we search if the embedding of this bug_id is already in the database to avoid recalculation
        2. use the embedding to search for k most similar ones
        :return: k most similar comment as the queried comment
        """
        # Search in Qdrant database for existing embedding with the given bug_id
        points, next_offset = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="bug_id",
                        match=models.MatchValue(value=bug_id)
                    )
                ]
            ),
            with_vectors=True,  # Ensure that payload data is included in the response
            limit=1  # Limit the number of results
        )

        if points:

            print(f"Embedding for bug_id {bug_id} found in database.")
            embedding= np.array(points[0].vector)
        else:
            embedding = self.embedding_model.get_embeddings([comment])
            embedding=embedding[0]


        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=k  # Adjust the limit as needed
        )
        similar_bugs = [hit.payload for hit in hits]
        return similar_bugs

def add_symbol(comment):
    return f"```\n{comment}\n```"

def prompt_construction(comment, bug_id, rag):
    '''
        the target output follow's a llama-factory format:
        instruction: say something
        input: say something other
        output: the targeted output
    ''' 
    closest_comments=rag.query(comment, bug_id)
    closest_symbols = [add_symbol(comment) for comment in closest_comments]
    
    instruction_prompt = 'Please help identify the following bug is caused by a third-party library.'
    input_prompt = f'The bug to identify is {comment}.\n'
    output_prompt = f'Its similar bugs are {closest_symbols}'
    
    chat = {'instruction': instruction_prompt, 'input': input_prompt, 'output': output_prompt}
    return chat

if __name__=='__main__':

    rag = RAG(embedding_model='LLAMA')



    # Create Emebedding
    # csv_files = ['dataset/KDE_featured.csv','dataset/firefox_featured.csv','dataset/thunderbird_featured.csv']
    # for csv_file in csv_files:
    #     # rag = RAG(embedding_model='LLAMA')
    #     rag.initialize(csv_file)

    #Query similar embedding

    comments=rag.query('Hello World','dataset/KDE_featured-395283')

