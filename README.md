<div align="center">

# 🔥 RAG vs Non-RAG Chatbot 
### 🛠️ A Comparative Framework for Optimized LLM Query Handling

[![python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-31012/)
[![langchain](https://img.shields.io/badge/langchain-0.1.4-brightgreen)](https://www.langchain.com/)
[![chroma](https://img.shields.io/badge/chroma-0.4.15-orange)](https://www.trychroma.com/)
[![FAISS](https://img.shields.io/badge/faiss-1.7.4-yellow)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Flan--T5-red)](https://huggingface.co/google/flan-t5-large)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

[Architecture](#architecture)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Setup](#setup)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Usage](#usage)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Results](#results)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Contributing](#contributing)

</div>

---

## 🚀 Overview

The **RAG vs Non-RAG Chatbot** framework demonstrates the performance and accuracy differences between:
- **RAG (Retrieval-Augmented Generation)**: Enhances LLM responses with contextually relevant external knowledge. Utilizes **ChromaDB** for vector storage and **FAISS** for retrieval.
- **Non-RAG**: Direct LLM-based answering without retrieval. Uses **Flan-T5** hosted on **Hugging Face**.

This project enables side-by-side comparisons of:
- Response accuracy
- Latency and efficiency
- Contextual relevance with/without retrieval

---

## ⚙️ Architecture

<div align="center">
<img src="./assets/architecture.png" width="80%">
</div>

### 🛠️ **Components**
1. **RAG Pipeline**
   - Embedding Model: `fastembed` for text embedding.
   - Vector Store: `ChromaDB` with `FAISS` backend.
   - Retrieval: Query similarity search with top-k matching.
   - LLM: `ChatOllama` for generating responses based on retrieved context.
   
2. **Non-RAG Pipeline**
   - Direct querying with `Flan-T5` hosted on Hugging Face.
   - No external knowledge retrieval.

---

## 🛠️ Setup

### ✅ **Prerequisites**
- Python `3.10+`
- Install required dependencies:
```bash
pip install -r requirements.txt
