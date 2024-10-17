from retriever import load_retriever
from processing import loading_dataset
from transformers import  AutoModelForSequenceClassification, AutoTokenizer
import torch
from langchain.schema import Document
from langchain_nvidia_ai_endpoints.reranking import NVIDIARerank
from artifacts import keys
def get_top_k_passages():
    loaded_retriever = load_retriever()

    corpus , queries = loading_dataset.load_docs()

    query = queries['text'][20]
    answer = corpus['text'][20]
    top_k_retrieved = loaded_retriever.similarity_search(query, k=5)
    answers = {}
    answers['query'] = query
    answers['ground truth'] = answer
    answers['top_k_retrieved'] = top_k_retrieved
    print(answers)
    return top_k_retrieved , query

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rerank(self, query: str, passages: list[Document], top_k: int = 5):
        # Create a list to hold passages with their relevance scores
        reranked_passages = []
        
        # For each passage, compute relevance score
        for passage in passages:
            inputs = self.tokenizer(query, passage.page_content, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                print("output" , outputs)
                score = outputs.logits[0].item()  # Get relevance score for passage
            
            # Append passage and score as tuple
            reranked_passages.append((passage, score))
        
        # Sort by relevance score in descending order
        reranked_passages.sort(key=lambda x: x[1], reverse=True)

        # Return top-k reranked passages
        return [passage for passage, score in reranked_passages[:top_k]]

def smaller_model_reranking__small():
    top_k_retrieved , query = get_top_k_passages()
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Rerank the retrieved passages
    reranked_passages = reranker.rerank(query, top_k_retrieved, top_k=5)

    print("\nquery",query,"\n")
    # Print reranked results
    for idx, passage in enumerate(reranked_passages):
        print(f"Rank {idx + 1}: {passage.page_content}")
    return None

def smaller_model_reranking_large():
    top_k_retrieved , query = get_top_k_passages()
    #reranker = CrossEncoderReranker(model_name="nvidia/nv-rerankqa-mistral-4b-v3")
        # Rerank the retrieved passages

    client = NVIDIARerank(
        model="nvidia/nv-rerankqa-mistral-4b-v3",
        api_key=keys.NVIDIA_API_KEY
    )
    response = client.compress_documents(
        query=query,
        documents=top_k_retrieved
    )
    #reranked_passages = reranker.rerank(query, top_k_retrieved, top_k=5)

    print("\nquery",query,"\n")
    print(response)
    # Print reranked results
    for idx, passage in enumerate(response):
        print(f"Rank {idx + 1}: {passage.page_content}")
    return None

smaller_model_reranking_large()