import os
from src.pipeline import Pipeline
from src.config import config

if __name__ == "__main__":
    query = "When did Abhishek Pandey start his internship at Ex Squared Pvt Ltd?"
    directory_path = config.DATA_DIR

   
    os.makedirs(directory_path, exist_ok=True)
   
    file1_path = os.path.join(directory_path, "doc1.txt")
    file2_path = os.path.join(directory_path, "doc2.txt")

    if not os.path.exists(file1_path):
        with open(file1_path, "w") as f:
            f.write("Toddlers throw tantrums for various reasons, including frustration, hunger, and tiredness.")
    if not os.path.exists(file2_path):
        with open(file2_path, "w") as f:
            f.write("Tantrums are a normal part of child development. They usually decrease as children learn to express themselves better.")

    pipeline = Pipeline()
    response = pipeline.run(directory_path, query)
    print(response)
