<div align="center">

# 🔥 RAG vs Non-RAG Chatbot 
### 🛠️ A Comparative Framework for Optimized LLM Query Handling

[![python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-31012/)
[![langchain](https://img.shields.io/badge/langchain-0.1.4-brightgreen)](https://www.langchain.com/)
[![chroma](https://img.shields.io/badge/chroma-0.4.15-orange)](https://www.trychroma.com/)
[![FAISS](https://img.shields.io/badge/faiss-1.7.4-yellow)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Flan--T5-red)](https://huggingface.co/google/flan-t5-large)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

[Architecture](#architecture)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Setup](#setup)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Environment Variables](#environment-variables)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Usage](#usage)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Testing](#testing)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Results](#results)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Contributing](#contributing)

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
- Clone the repository:

```bash
git clone https://github.com/Vvslaxman/rag-norag
cd rag-vs-nonrag-chatbot


# RAG vs Non-RAG Chatbot

## 📦 Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following variables:

```ini

# Hugging Face API Token
HUGGINGFACEHUB_API_TOKEN=<your_huggingface_api_token>

# ChromaDB Configuration
CHROMA_DB_HOST=localhost
CHROMA_DB_PORT=8000
CHROMA_COLLECTION=rag_data

# FAISS Configuration
FAISS_INDEX_PATH=./faiss_index

# FastEmbed Configuration
FASTEMBED_MODEL=all-MiniLM-L6-v2
```

## 🚀 Usage

### ✅ Start the Chatbot

To start the chatbot, run:

```bash
streamlit run app.py
```

### ✅ Interacting with the Chatbot

Once running:

- Open your browser and navigate to `http://localhost:5000`
- Enter a query in the chatbot UI
- Select either RAG or Non-RAG mode
- Compare the results side by side

## 🧪 Testing

### ✅ Run Unit Tests

To ensure everything is working properly, run the testing script:

```bash
python test.py
```

## 🛠️ Folder Structure

```
📁 RAG-vs-NonRAG-Chatbot
 ├── 📁 assets             # Architecture diagrams, images
 ├── 📁 data               # Sample documents for RAG
 ├── 📁 models             # Pre-trained LLM models
 ├── 📁 src                # Source code
 │      ├── rag_pipeline.py
 │      ├── non_rag_pipeline.py
 │      └── app.py
 ├── .env                  # Environment variables
 ├── benchmark.py          # Benchmarking script
 ├── requirements.txt      # Dependencies
 ├── README.md             # Project documentation
 └── LICENSE               # License file
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add new feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Create a Pull Request.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📬 Contact

For any issues or questions, feel free to open an issue or reach out at:

- **GitHub**: [Vvslaxman](https://github.com/Vvslaxman)
- **Email**: vvslaxman14@gmail.com
