from processing import loading_dataset
from processing import document_processing

from utils import embedding, vectorstore



def create_and_store_vectorstore():
    corpus , _ = loading_dataset.load_docs()

    documents = document_processing.process_documents(corpus)
        
    faiss_processed_docs = document_processing.create_faiss_docs(documents)
    embeddings = embedding.small_embedding_model
    faiss_vectorstore = vectorstore.create_vectorstore(faiss_processed_docs , embedding_model= embeddings)

    faiss_vectorstore.save_local('faiss_index')

    
    return None

def load_retriever():
    embeddings = embedding.small_embedding_model
    loaded_retriever = vectorstore.load_vectorstore(embeddings)
    return loaded_retriever

