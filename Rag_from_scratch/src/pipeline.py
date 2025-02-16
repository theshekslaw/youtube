from src.doc_loader import DocumentLoader
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator
from src.config import config

class Pipeline:
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.embedder = Embedder()

    def run(self, directory_path, query):
        chunk_data = self.document_loader.process_documents(directory_path)

        # Create and load FAISS index
        self.embedder.create_index(chunk_data)
        index, loaded_chunk_data = self.embedder.load_index()

        # Pass embedder, index, and chunk data correctly
        retriever = Retriever(self.embedder, index, loaded_chunk_data)
        relevant_chunks = retriever.retrieve(query)

        if relevant_chunks:
            context = "\n".join([chunk["text"] for chunk in relevant_chunks])
            generator = Generator()
            response = generator.generate_response(query, context)
        else:
            response = "I'm sorry, I couldn't find relevant information to answer your query."

        return response
