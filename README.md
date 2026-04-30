# 🧠 Multimodal RAG System (Local, Hybrid, Vision-Enhanced)

A **production-style Multimodal RAG (Retrieval-Augmented Generation) system** that can ingest PDFs with **text, tables, and images**, and answer questions using advanced retrieval and reasoning techniques.

---

## 🚀 Key Features

- 📄 Smart PDF parsing using Unstructured  
- 🧠 AI-enhanced chunk summaries (text + image aware)  
- 🔍 Hybrid search (semantic + keyword)  
- 🔁 Multi-query retrieval for better recall  
- 🧮 Reciprocal Rank Fusion (RRF)  
- 🎯 MiniLM reranker for precision  
- 🖼️ LLaVA for visual reasoning  
- 🧾 Structured JSON debug output  
- 💻 Fully local (no API required)  

---

## 🏗️ Architecture

```text
User Query
   ↓
Multi-Query Expansion
   ↓
Hybrid Retrieval
   → Vector Search (MMR)
   → BM25 Keyword Search
   ↓
RRF (Reciprocal Rank Fusion)
   ↓
MiniLM Reranker
   ↓
Final Top Chunks
   ↓
LLaVA Image Analysis
   ↓
llama3.2 Final Answer
```
---

## ⚙️ Tech Stack

| Component    | Technology            |
|--------------|----------------------|
| LLM          | Ollama (llama3.2)    |
| Vision Model | LLaVA                |
| Embeddings   | mxbai-embed-large    |
| Vector DB    | ChromaDB             |
| Reranker     | MiniLM (CrossEncoder)|
| Retrieval    | BM25 + Vector + MMR  |
| PDF Parsing  | Unstructured         |
| Framework    | LangChain            |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

---

## 🤖 Download Models (Ollama)

```bash
ollama pull llama3.2
ollama pull llava
ollama pull mxbai-embed-large
```

## 📄 Dataset / Knowledge Source

This project uses the Transformer research paper:

📘 **Attention Is All You Need**  
📁 Located in: `docs/attention-is-all-you-need.pdf`
