import os
import re
import uuid
from transformers import AutoTokenizer
from src.config import config


class DocumentLoader:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)


    def load_documents(self, directory_path):
        documents = {}
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                doc_id = str(uuid.uuid4())
                documents[doc_id] = {"text": text, "filename": filename}
        return documents


    def chunk_text(self, text, chunk_size, chunk_overlap):
        chunks = []
        tokens = self.tokenizer.tokenize(text)
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
            chunks.append(chunk_text)
        return chunks
    
     
    def process_documents(self, directory_path):
        documents = self.load_documents(directory_path)
        all_chunks = {}
        for doc_id, doc in documents.items():
            text = doc["text"]
            filename = doc["filename"]
            chunks = self.chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                all_chunks[chunk_id] = {
                    "text": chunk,
                    "metadata": {"doc_id": doc_id, "filename": filename},
                }
        print(f"Loaded {len(all_chunks)} chunks from {directory_path}") 
        return all_chunks