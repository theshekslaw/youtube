import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.config import config

class Retriever:
    def __init__(self, embedder, index, chunk_data):
        self.embedder = embedder  
        self.index = index  
        self.chunk_data = chunk_data  

    def generate_embedding(self, text):
        return self.embedder.generate_embedding(text)
    
    def retrieve(self, query, top_k=config.TOP_K):
        query_embedding = self.generate_embedding(query)
        query_embedding = np.array([query_embedding]).astype("float32")

        print(f"Query embedding shape: {query_embedding.shape}")  

        D, I = self.index.search(query_embedding, top_k)
        print(f"Retrieved IDs: {I}")  
        print(f"Retrieved Distances: {D}")  

        results = []
        for idx in I[0]:
            if idx != -1:
                print(f"Retrieving chunk for ID: {idx}") 
                chunk = self.chunk_data.get(idx)
                if chunk:
                    results.append(chunk)
                else:
                    print(f"⚠️ No chunk found for ID: {idx}")  

        print(f"Retrieved {len(results)} chunks")  
        return results

