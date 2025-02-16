from transformers import pipeline
from src.config import config
from dotenv import load_dotenv 
import os 
load_dotenv()
class Generator:
    def __init__(self):
        self.api_key = os.getenv("hugging")
        self.model_name = "openai-community/gpt2-medium"  
        self.generator = pipeline("text-generation", 
            model=self.model_name, 
            token=self.api_key, 
            device_map="auto")
        
    def generate_response(self, query, context):
        prompt = f"""
        You are a helpful assistant. Use the following context to answer the user's question.
        Context: {context}
        Question: {query}
        Answer:
        """

        response = self.generator(prompt, max_new_tokens = 100, num_return_sequences = 1)[0]['generated_text']
        tokens = self.generator.tokenizer(prompt, return_tensors="pt")
        print(f"Tokenized input length: {tokens['input_ids'].shape[1]}")
        return response
