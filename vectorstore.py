from utils import retriever
from processing import loading_dataset
from processing import document_processing


corpus , queries = loading_dataset.load_docs()

documents = document_processing.process_documents(corpus)

faiss_processed_docs = document_processing.create_faiss_docs(documents)

faiss_vectorstore = retriever.create_vectorstore(faiss_processed_docs)


