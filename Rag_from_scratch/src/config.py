import os

class Config:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")

    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    LLM_MODEL = "deepseek-ai/deepseek-llm-7b-chat"

    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 20

    TOP_K = 5 


os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)


config = Config()