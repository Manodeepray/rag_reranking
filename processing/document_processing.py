from tqdm import tqdm
from langchain.schema import Document


def process_documents(corpus):
    documents = []
    for i in tqdm(range(len(corpus['_id']))):
      faiss_doc = {}
      faiss_doc['id'] = corpus['_id'][i]
      faiss_doc['page_content'] = corpus['text'][i]
      faiss_doc['metadata'] = corpus['title'][i]
      documents.append(faiss_doc)
    
    return documents

def create_faiss_docs(documents):
    
  faiss_processed_docs = []

  for i in tqdm(range(len(documents))):
    d = documents[i]
    metadata = d["metadata"] if isinstance(d["metadata"], dict) else {"description": d["metadata"]}
    text = Document(page_content=d["page_content"], metadata=metadata, id=d["id"])
    faiss_processed_docs.append(text)

  return faiss_processed_docs


