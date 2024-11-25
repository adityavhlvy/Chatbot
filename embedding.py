from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

def get_embedding_function():
    # Define local model path
    model_path = r"C:/LLM-Model/all-MiniLM-L6-v2"
    # model_path = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Use HuggingFaceEmbeddings with the loaded model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings