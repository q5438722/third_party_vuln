
import pandas as pd
import openai
import qdrant_client
from qdrant_client.http import models
from embedding import LLAMA,OPENAI
from utils import load_config
import numpy as np
class RAG:

    def __init__(self,mode='query',embedding_model='LLAMA',embedding_size=4096,qdrant_path='data/qdrant_db', embeddings_dir='data/embeddings',batch_size=16):
        self.embeddings_dir = embeddings_dir
        self.qdrant_path = qdrant_path
        self.collection_name = 'comment_embeddings'
        self.embeddings_size=embedding_size
        self.batch_size=batch_size
        config = load_config()
        if mode=='train':
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

    def initialize_with_embedding(self, csv_file, vectors, partition=1.0):


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
        labels=dataset['label'].to_list()
        payloads = [{"bug_id": f"{dataset_name}-{i}"}  for i in dataset.bug_id.tolist()]

        self.id2comment = {f"{dataset_name}-{bug_id}": comment for bug_id, comment in 
                           zip(dataset['bug_id'].to_list(), comments)}
        self.id2label = {f"{dataset_name}-{bug_id}": label for bug_id, label in
                           zip(dataset['bug_id'].to_list(), labels)}


        unique_indxes = dataset.drop_duplicates(subset=['comment']).index.to_list()
        unique_vectors = [vec for idx, vec in enumerate(vectors) if idx in unique_indxes]
        unique_payloads = [pay for idx, pay in enumerate(payloads) if idx in unique_indxes]

        partition_length = int(len(unique_vectors) * partition)
        print(f"Uploading {partition_length} embeddings to Qdrant.")

        # Upload all embeddings in a batch
        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=unique_vectors[:partition_length],
            payload=unique_payloads[:partition_length]
        )
        print("All embeddings have been processed and saved to Qdrant.")

    def query(self,comment, bug_id, k=3):
        similar_ids=self.find_similar_bugids(comment,bug_id, k)
        bug_infos=[]
        for similar_id in similar_ids:
            # comments.append(self.bugid_to_comment(similar_id['bug_id']))
            bug_info=(self.id2comment[similar_id['bug_id']],self.id2label[similar_id['bug_id']])
            if bug_info[0] == comment:
                continue
            bug_infos.append(bug_info)
        return bug_infos[:k]

    def bugid_to_comment(self,bug_id):
        dataset,query_id=bug_id.split('-')
        df=pd.read_csv(dataset+'.csv')
        df.set_index('bug_id', inplace=True)
        comment = df.at[int(query_id), 'comment']

        if isinstance(comment, str):
            return comment

    def find_similar_bugids(self, comment, bug_id, k=3):
        """
        1. we search if the embedding of this bug_id is already in the database to avoid recalculation
        2. use the embedding to search for k most similar ones
        :return: k most similar comment as the queried comment
        """
        # Search in Qdrant database for existing embedding with the given bug_id
        # points, next_offset = self.client.scroll(
        #     collection_name=self.collection_name,
        #     scroll_filter=models.Filter(
        #         must=[
        #             models.FieldCondition(
        #                 key="bug_id",
        #                 match=models.MatchValue(value=bug_id)
        #             )
        #         ]
        #     ),
        #     with_vectors=True,  # Ensure that payload data is included in the response
        #     limit=1  # Limit the number of results
        # )

        # if points:

        #     print(f"Embedding for bug_id {bug_id} found in database.")
        #     embedding= np.array(points[0].vector)
        # else:
        #     embedding = self.embedding_model.get_embeddings([comment])
        #     embedding=embedding[0]
        embedding = self.embedding_model.get_embeddings([comment])
        embedding=embedding[0]


        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=k*2  # Adjust the limit as needed
        )
        similar_bugs = [hit.payload for hit in hits]
        return similar_bugs

    def add_symbol(self,comment, label=None):
        label_mapping={'1':'caused by a third-party issue','0': 'not caused by a third-party issue',
                       1:'caused by a third-party issue', 0: 'not caused by a third-party issue'}
        assert label is not None
        return f"The description of this bug is:\n```\n{comment}\n``` and it is {label_mapping[label]}.\n"

    def prompt_construction(self,comment, bug_id):
        '''
            the target output follow's a llama-factory format:
            instruction: say something
            input: say something other
            output: the targeted output
        '''
        closest_buginfos=self.query(comment, bug_id)
        closest_symbols = [self.add_symbol(comment,label) for comment,label in closest_buginfos]

        instruction_prompt = 'Please determine whether the following bug is caused by a third-party issue'
        input_prompt = f'The description of this bug is {comment}.\n'
        rag_prompt = f'Its similar bugs are {closest_symbols}.\n'
        
        label_mapping={'1':'caused by a third-party issue','0': 'not caused by a third-party issue',
                       1:'caused by a third-party issue', 0: 'not caused by a third-party issue'}
        label = self.id2label[bug_id]
        output_prompt = f'it is {label_mapping.get(label)}.\n'

        chat = {'instruction': instruction_prompt, 'input': input_prompt + rag_prompt, 'output': output_prompt}
        return chat

if __name__=='__main__':

    rag = RAG(embedding_model='LLAMA')



    # Create Emebedding
    csv_files = ['dataset/KDE_featured.csv','dataset/firefox_featured.csv','dataset/thunderbird_featured.csv']
    for csv_file in csv_files:
        # rag = RAG(embedding_model='LLAMA')
        rag.initialize(csv_file)

    #Query similar embedding

    comments=rag.query('Hello World','dataset/KDE_featured-395283')

