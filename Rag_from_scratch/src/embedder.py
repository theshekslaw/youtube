import os
import faiss
import torch
import pickle
import hashlib
from transformers import AutoTokenizer, AutoModel
import numpy as np
import hashlib
from src.config import config


class Embedder:
    def __init__(self):
        self.model_name = config.EMBEDDING_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)


    def generate_embedding(self, text):
        self.model.eval()
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        return embeddings.cpu().numpy().flatten()


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def create_index(self, chunk_data):
        dimension = len(self.generate_embedding(list(chunk_data.values())[0]["text"]))
        index = faiss.IndexFlatL2(dimension)
        index_id_map = faiss.IndexIDMap(index)

        embeddings = []
        ids = []
        chunk_mapping = {}

        for chunk_id, chunk in chunk_data.items():
            embedding = self.generate_embedding(chunk["text"])
            embeddings.append(embedding)

            # 🔹 Ensure consistent ID hashing
            hashed_id = int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % (2**63)
            print(f"Storing Chunk ID: {chunk_id} -> Hashed ID: {hashed_id}")  # ✅ Debugging
            ids.append(hashed_id)

            # ✅ Store mapping using hashed_id as key
            chunk_mapping[hashed_id] = chunk  

        embeddings = np.array(embeddings).astype("float32")
        ids = np.array(ids).astype("int64")

        index_id_map.add_with_ids(embeddings, ids)
        faiss.write_index(index_id_map, os.path.join(config.EMBEDDINGS_DIR, "faiss_index.bin"))

        # ✅ Store the chunk mapping with hashed IDs
        with open(os.path.join(config.EMBEDDINGS_DIR, "chunk_mapping.pkl"), "wb") as f:
            pickle.dump(chunk_mapping, f)

        print(f"FAISS index now contains {index_id_map.ntotal} vectors.")  # ✅ Debugging

    def load_index(self):
            """Loads the FAISS index and chunk mapping"""
            index_path = os.path.join(config.EMBEDDINGS_DIR, "faiss_index.bin")
            mapping_path = os.path.join(config.EMBEDDINGS_DIR, "chunk_mapping.pkl")

            # 🔹 Load FAISS index
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index file not found: {index_path}")
            index = faiss.read_index(index_path)
            print(f"✅ Loaded FAISS index with {index.ntotal} vectors.")

            # 🔹 Load Chunk Mapping
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"Chunk mapping file not found: {mapping_path}")
            with open(mapping_path, "rb") as f:
                chunk_mapping = pickle.load(f)
            print(f"✅ Loaded chunk mapping with {len(chunk_mapping)} entries.")

            return index, chunk_mapping  