from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


def create_vectorstore(faiss_processed_docs):
    # Initialize SentenceTransformer embeddings using LangChain's HuggingFaceEmbeddings wrapper
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_store = FAISS.from_documents(faiss_processed_docs, embedding_model)
    return  faiss_store