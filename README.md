# Multi-Stage Text Retrieval for Question Answering Systems

## Project Overview

This project focuses on implementing a **multi-stage text retrieval pipeline** for **question-answering (QA)** tasks, integrating both **embedding-based candidate retrieval** and **ranking-based reranking**. The project benchmarks various models to assess their impact on retrieval accuracy and aims to provide insights into trade-offs between model size, accuracy, and computational performance.

The main goal is to implement and evaluate **multi-stage retrieval pipelines** that can retrieve and rank relevant passages from large corpora, specifically tailored for question-answering systems. The project uses publicly available datasets from the **BEIR benchmark** and state-of-the-art models for candidate retrieval and passage reranking.

---

## Task Overview

This project follows the steps outlined below:

### 1. **Literature Review and Understanding**

- **Paper Reference**: [Understanding Ranking Models for Question Answering Systems](https://www.arxiv.org/abs/2409.07691)
- A comprehensive review of the paper helps in understanding how **multi-stage text retrieval** works and the role of **embedding models** and **ranking models** in improving the accuracy of question-answering systems.
- Key concepts include:
  - The importance of **retrieving relevant candidate passages** (using embedding models).
  - **Ranking models** and how they enhance retrieval accuracy by reranking passages based on query relevance.

### 2. **Dataset Preparation**

- **Datasets**: Publicly available Q&A datasets from the **BEIR benchmark**, such as:
  - **Natural Questions (NQ)**
  - **HotpotQA**
  - **FiQA**
- The datasets are preprocessed to ensure they are tokenized and split into smaller chunks/passages, optimizing them for efficient retrieval and reranking.
- **Preprocessing steps** include chunking long documents into passages and tokenizing them to fit into the retrieval model input format.

### 3. **Implement the Retrieval Pipeline**

#### Stage 1 - **Candidate Retrieval**

- Two embedding models are selected, each serving different requirements in terms of size and performance:
  - A **small embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (lightweight, efficient, and commercially usable).
  - A **large embedding model**: `nvidia/nv-embedqa-e5-v5` (more computationally intensive but with improved retrieval accuracy).
- These models are used for retrieving the **top-k relevant passages** for a given query.

#### Stage 2 - **Reranking**

- Two **ranking models** are chosen to reorder the retrieved passages based on relevance:
  - **Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2` (a compact model specifically designed for reranking tasks in QA).
  - **Larger Reranker**: `nvidia/nv-rerankqa-mistral-4b-v3` (a powerful reranking model, but heavier on resources).
- The reranking models evaluate and reorder the candidate passages to improve accuracy based on the query-passage relevance scores.

### 4. **Benchmarking and Evaluation**

- The pipeline is evaluated using the **NDCG@10** metric, which assesses the quality of the ranked passages by measuring the relevance of the top-10 retrieved passages.
- **Comparisons**:
  - Retrieval accuracy is compared with and without reranking models.
  - Various combinations of embedding and reranking models are tested to analyze the trade-offs between **model size** and **retrieval accuracy**.
  - The project highlights how larger models often enhance accuracy but at the cost of increased computational resources, while smaller models provide faster results with slight accuracy trade-offs.

---

## Project Architecture

1. **Input Query**: A user provides a natural language question.
2. **Candidate Retrieval**: The embedding models retrieve the most relevant passages from the corpus.
3. **Passage Reranking**: Cross-encoder reranking models reorder the retrieved passages based on their relevance to the query.
4. **Evaluation and Output**: The top reranked passages are evaluated using NDCG@10 and returned.

---

## Libraries Used

The project utilizes a set of core libraries for machine learning, natural language processing, and retrieval tasks:

- **Hugging Face Transformers**:

  - Provides pre-trained transformer models for embedding and cross-encoder tasks.
  - Used for loading the `sentence-transformers`, `cross-encoder`, and `nvidia` models.
  - Install via: `pip install transformers`

- **Sentence-Transformers**:

  - Facilitates the use of embedding models like `all-MiniLM-L6-v2`.
  - Install via: `pip install sentence-transformers`

- **FAISS (Facebook AI Similarity Search)**:

  - A library for efficient similarity search, used for fast nearest neighbor search over embeddings.
  - Install via: `pip install faiss-cpu` (or `faiss-gpu` for GPU support)

- **PyTorch**:

  - The backbone for loading and running models, including cross-encoders for reranking.
  - Install via: `pip install torch`

- **Datasets** (Hugging Face Datasets):

  - Used for loading and processing BEIR benchmark datasets like NQ, HotpotQA, and FiQA.
  - Install via: `pip install datasets`

- **Scikit-learn**:

  - For metrics such as **NDCG@10**, to evaluate retrieval and ranking performance.
  - Install via: `pip install scikit-learn`

- **TQDM**:
  - For progress bars to monitor the retrieval and ranking processes.
  - Install via: `pip install tqdm`

---

## Project Goals

1. **Understand Ranking Models**:

   - A key focus is on understanding how ranking models impact the retrieval accuracy in question-answering systems. By reranking the retrieved passages, the system improves the precision of the final answers.

2. **Benchmark Different Models**:

   - The project aims to compare and benchmark different **embedding** and **ranking models** in terms of **accuracy**, **performance**, and **computational cost**.
   - The analysis will provide insights into the trade-offs between model size, retrieval accuracy, and performance.

3. **Multi-Stage Retrieval Pipeline**:
   - The pipeline aims to improve the relevance of passages in question-answering tasks by leveraging both embedding models and ranking models in a structured two-stage process.

---

## Results and Insights

- **Impact of Reranking**: Preliminary results show that adding a reranking stage significantly improves retrieval accuracy, with larger models providing higher accuracy at the expense of speed and resource usage.
- **Trade-offs**: Smaller models like `MiniLM` perform efficiently in candidate retrieval, but reranking with larger models like `Mistral` adds considerable accuracy to the final output.

---

## Future Work

- **End-to-End Pipeline**: Extend the project to include answer generation using a generative model to form a complete **Retrieval-Augmented Generation (RAG)** system.
- **Further Benchmarking**: Benchmark more models and refine the preprocessing steps to optimize passage retrieval across different QA datasets.
- **Scalability**: Explore the deployment of this multi-stage pipeline for large-scale real-time question-answering applications.

---

## License and Usage

This project is developed for educational and research purposes. The code and models are not intended for public use or distribution without proper authorization.

---

## Contact Information

For further inquiries or collaboration opportunities, please contact [manodeepray1@gmail.com].

---

## Acknowledgments

This project leverages open-source models and datasets from the HuggingFace repository and the BEIR benchmark.

---
