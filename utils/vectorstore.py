from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


def create_vectorstore(faiss_processed_docs , embedding_model):
    # Initialize SentenceTransformer embeddings using LangChain's HuggingFaceEmbeddings wrapper
    
    faiss_store = FAISS.from_documents(faiss_processed_docs, embedding_model)
    
    return  faiss_store

def load_vectorstore(embeddings):
    faiss_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    return faiss_store


def load_larger_vectorstore(embeddings):
    faiss_store = FAISS.load_local(
    "faiss_larger_index", embeddings, allow_dangerous_deserialization=True
    )
    return faiss_store


